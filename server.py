"""Content Analyzer — FastAPI backend."""

import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from database import (
    init_db, upsert_bookmark, save_analysis, get_all_bookmarks,
    get_analysis, get_all_categories, get_stats, get_bookmarks_grouped,
    search_analyzed_bookmarks, save_brief, get_all_briefs, delete_brief,
    get_brief, update_brief,
    save_oauth_pkce, pop_oauth_pkce, get_oauth_tokens, set_oauth_tokens,
    clear_oauth_tokens,
)
from analyzer import (
    analyze_single_async, generate_markdown_report, fetch_web_content,
    generate_brief_async, chat_about_content_async, chat_about_brief_async,
    _exa_crawl_async,
)
from x_api import (
    extract_tweet_id, fetch_tweet_by_id, fetch_tweet_via_fxtwitter,
    fetch_all_bookmarks, generate_pkce_pair, get_oauth2_auth_url,
    exchange_oauth2_code, refresh_oauth2_token,
)

# ─── Config: env vars first, fallback to secrets.toml for local dev ───

def _load_secrets() -> dict:
    path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if not path.exists():
        return {}
    config = {}
    current_section = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("["):
            current_section = line.strip("[]")
            config[current_section] = {}
        elif "=" in line and current_section:
            key, val = line.split("=", 1)
            config[current_section][key.strip()] = val.strip().strip('"')
    return config

_secrets = _load_secrets()
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or _secrets.get("anthropic", {}).get("api_key", "")
X_BEARER = os.environ.get("X_BEARER_TOKEN") or _secrets.get("x_api", {}).get("bearer_token", "")
X_CLIENT_ID = os.environ.get("X_CLIENT_ID") or _secrets.get("x_api", {}).get("client_id", "")
EXA_KEY = os.environ.get("EXA_API_KEY") or _secrets.get("exa", {}).get("api_key", "")
APIFY_TOKEN = os.environ.get("APIFY_API_TOKEN") or _secrets.get("apify", {}).get("api_token", "")

_IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("DATABASE_URL"))

_OAUTH_DIR = Path("/tmp" if _IS_SERVERLESS else str(Path(__file__).parent / ".oauth"))
if not _IS_SERVERLESS:
    _OAUTH_DIR.mkdir(exist_ok=True)
_VERIFIER_FILE = _OAUTH_DIR / "verifier"
_TOKEN_FILE = _OAUTH_DIR / "token.json"

_X_TOKEN_ENV = os.environ.get("X_USER_TOKEN", "")


def _save_tokens(access: str, refresh: str = ""):
    saved = False
    try:
        set_oauth_tokens("x", access, refresh or "")
        verify = get_oauth_tokens("x")
        if verify.get("access_token"):
            saved = True
        else:
            print(f"[OAUTH] DB save succeeded but verify read-back empty")
    except Exception as e:
        print(f"[OAUTH] DB save error: {e}")
    try:
        _TOKEN_FILE.write_text(json.dumps({"access_token": access, "refresh_token": refresh}))
        if not saved:
            saved = True
    except Exception as e:
        print(f"[OAUTH] File save error: {e}")
    if not saved:
        print("[OAUTH] WARNING: tokens could not be persisted anywhere!")
    return saved

def _load_tokens() -> dict:
    if _X_TOKEN_ENV:
        return {"access_token": _X_TOKEN_ENV, "refresh_token": ""}
    try:
        db_tok = get_oauth_tokens("x")
        if db_tok.get("access_token"):
            return db_tok
    except Exception as e:
        print(f"[OAUTH] DB load error: {e}")
    if _TOKEN_FILE.exists():
        try:
            data = json.loads(_TOKEN_FILE.read_text())
            tok = data if isinstance(data, dict) else {"access_token": data, "refresh_token": ""}
            if tok.get("access_token"):
                try:
                    set_oauth_tokens("x", tok["access_token"], tok.get("refresh_token", ""))
                except Exception:
                    pass
                return tok
        except Exception as e:
            print(f"[OAUTH] File load error: {e}")
    return {}

def _clear_tokens():
    try:
        clear_oauth_tokens("x")
    except Exception as e:
        print(f"[OAUTH] DB clear error: {e}")
    try:
        _TOKEN_FILE.unlink(missing_ok=True)
    except Exception:
        pass

def _get_user_token() -> str:
    return _load_tokens().get("access_token", "")

def _try_refresh() -> str:
    tokens = _load_tokens()
    rt = tokens.get("refresh_token", "")
    if not rt or not X_CLIENT_ID:
        return ""
    result = refresh_oauth2_token(X_CLIENT_ID, rt)
    if result.get("error"):
        return ""
    _save_tokens(result["access_token"], result.get("refresh_token", rt))
    return result["access_token"]


# ─── App ───

app = FastAPI(title="Content Analyzer")
try:
    init_db()
except Exception as _init_err:
    print(f"INIT_DB ERROR: {_init_err}")


@app.get("/api/health")
def health():
    from database import _SB, _PG, SUPABASE_URL
    token_status = "unknown"
    try:
        tok = _load_tokens()
        token_status = "has_token" if tok.get("access_token") else "no_token"
    except Exception as e:
        token_status = f"error: {e}"
    key_preview = ANTHROPIC_KEY[:15] + "..." if ANTHROPIC_KEY else "NOT SET"
    if _SB:
        try:
            from database import get_stats
            s = get_stats()
            return {"status": "ok", "mode": "supabase_rest", "stats": s, "oauth": token_status, "anthropic_key": key_preview}
        except Exception as e:
            return {"status": "error", "mode": "supabase_rest", "error": str(e), "oauth": token_status, "anthropic_key": key_preview}
    try:
        from database import get_connection
        conn = get_connection()
        conn.close()
        return {"status": "ok", "mode": "pg" if _PG else "sqlite", "oauth": token_status}
    except Exception as e:
        return {"status": "error", "error": str(e), "oauth": token_status}


@app.get("/api/debug/oauth")
def debug_oauth():
    """Test OAuth token storage + PKCE round-trip."""
    results = {}
    try:
        results["1_load_before"] = _load_tokens()
    except Exception as e:
        results["1_load_before"] = f"error: {e}"
    try:
        set_oauth_tokens("_test", "test_token_123", "test_refresh_456")
        results["2_write"] = "ok"
    except Exception as e:
        results["2_write"] = f"error: {e}"
    try:
        read_back = get_oauth_tokens("_test")
        results["3_read_back"] = read_back
        results["3_success"] = read_back.get("access_token") == "test_token_123"
    except Exception as e:
        results["3_read_back"] = f"error: {e}"
    try:
        clear_oauth_tokens("_test")
        results["4_cleanup"] = "ok"
    except Exception as e:
        results["4_cleanup"] = f"error: {e}"
    try:
        test_state = "debug_test_state_12345"
        test_verifier = "debug_test_verifier_abc"
        save_oauth_pkce(test_state, test_verifier)
        results["5_pkce_save"] = "ok"
        popped = pop_oauth_pkce(test_state)
        results["6_pkce_pop"] = popped
        results["6_pkce_match"] = popped == test_verifier
    except Exception as e:
        results["5_pkce_error"] = f"error: {e}"
    return results


@app.get("/api/debug/analyses")
def debug_analyses(tweet_id: str = ""):
    """Diagnostic: teste le cycle write/read analyses pour un tweet_id."""
    from database import _SB, _sb, _analysis_fields
    results = {}

    if not _SB:
        return {"error": "Not in Supabase mode"}

    # 1. Liste quelques tweet_ids existants si aucun fourni
    if not tweet_id:
        try:
            bm_resp = _sb.table("bookmarks").select("tweet_id").limit(5).execute()
            results["hint"] = "Pass ?tweet_id=XXX to test a specific bookmark"
            results["sample_tweet_ids"] = [b["tweet_id"] for b in (bm_resp.data or [])]
        except Exception as e:
            results["bookmarks_error"] = str(e)
        return results

    # 2. Lecture directe table analyses
    try:
        direct = _sb.table("analyses").select("*").eq("tweet_id", tweet_id).execute()
        results["direct_read_count"] = len(direct.data or [])
        results["direct_read"] = direct.data[:2] if direct.data else None
    except Exception as e:
        results["direct_read_error"] = str(e)

    # 3. Lecture via embedded PostgREST (la requête utilisée par get_bookmarks_grouped)
    try:
        fields = _analysis_fields()
        embedded = (
            _sb.table("bookmarks")
            .select(f"tweet_id, analyses({fields})")
            .eq("tweet_id", tweet_id)
            .execute()
        )
        bm_data = embedded.data[0] if embedded.data else {}
        results["embedded_analyses"] = bm_data.get("analyses", [])
        results["embedded_count"] = len(bm_data.get("analyses", []))
    except Exception as e:
        results["embedded_read_error"] = str(e)

    # 4. Test write → read (insert test puis delete)
    try:
        test_row = {
            "tweet_id": tweet_id,
            "tags": "[]",
            "summary": "__debug_test__",
            "claims_check": "",
            "red_flags": "",
            "actionable": "",
            "full_analysis_md": "# Debug test",
            "analyzed_at": "2099-01-01T00:00:00",
            "embedding_vector": "[]",
            "verdict": "MIXED",
        }
        insert_resp = _sb.table("analyses").insert(test_row).execute()
        results["test_insert"] = "ok" if insert_resp.data else "empty_response"
        results["test_insert_data"] = insert_resp.data[:1] if insert_resp.data else None

        # Read back
        readback = _sb.table("analyses").select("id, tweet_id, summary").eq("tweet_id", tweet_id).eq("summary", "__debug_test__").execute()
        results["test_readback"] = readback.data

        # Cleanup
        if readback.data:
            for row in readback.data:
                _sb.table("analyses").delete().eq("id", row["id"]).execute()
            results["test_cleanup"] = "ok"
    except Exception as e:
        results["test_write_error"] = str(e)

    # 5. Diagnostic
    direct_ok = results.get("direct_read_count", 0) > 0
    embedded_ok = results.get("embedded_count", 0) > 0
    if direct_ok and embedded_ok:
        results["diagnosis"] = "OK — analyses exist and PostgREST join works"
    elif direct_ok and not embedded_ok:
        results["diagnosis"] = "BUG — analyses exist in DB but PostgREST embedded query fails (FK missing or RLS issue)"
    elif not direct_ok:
        results["diagnosis"] = "BUG — no analyses found for this tweet_id (save_analysis() not persisting)"
    return results




@app.get("/api/debug/anthropic")
async def debug_anthropic():
    """Test minimal Anthropic API call from Vercel."""
    import traceback
    try:
        from anthropic import AsyncAnthropic
        aclient = AsyncAnthropic(api_key=ANTHROPIC_KEY)
        response = await aclient.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say hi"}],
        )
        return {"status": "ok", "response": response.content[0].text}
    except Exception as e:
        return {"status": "error", "error": str(e), "type": type(e).__name__,
                "traceback": traceback.format_exc()}


def _parse_tags(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return []

def _parse_metrics(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {}


class APIOverloadedError(Exception):
    pass


async def _analyze_and_save(bm: dict) -> dict:
    content = bm["content"]
    if re.match(r'^https?://\S+$', content.strip()):
        enriched = fetch_tweet_via_fxtwitter(bm["tweet_id"])
        if enriched and enriched.get("content") and enriched["content"] != content:
            content = enriched["content"]
            bm["content"] = content
            upsert_bookmark(tweet_id=bm["tweet_id"], author_username=bm.get("author_username", ""),
                            author_name=bm.get("author_name", ""), content=content,
                            created_at=bm.get("created_at", ""), url=bm.get("url", ""))
    try:
        a = await analyze_single_async(content=content, author=bm.get("author_username", ""),
                                       api_key=ANTHROPIC_KEY, url=bm.get("url", ""), exa_key=EXA_KEY)
    except Exception as e:
        if "overloaded" in str(e).lower() or "529" in str(e):
            raise APIOverloadedError("overloaded")
        raise
    try:
        save_analysis(tweet_id=bm["tweet_id"], tags=a.get("tags", []),
                      summary=a.get("summary", ""), claims_check=a.get("claims_check", ""),
                      red_flags=a.get("red_flags", ""), actionable=a.get("actionable", ""),
                      full_md=generate_markdown_report(bm, a), verdict=a.get("verdict", ""),
                      translation=a.get("translation"), repo_analysis=a.get("repo_analysis"),
                      conclusion=a.get("conclusion"), eli5=a.get("eli5"),
                      web_research_summary=a.get("web_research_summary"),
                      feature_ideas=a.get("feature_ideas"))
    except Exception as e:
        print(f"[SAVE ERROR] tweet {bm['tweet_id']}: {e}")
        raise
    return a


# ═══════════ HTML ═══════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    tpl = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(tpl.read_text())


# ═══════════ API ═══════════

@app.get("/api/data")
async def api_data():
    groups, all_bm = get_bookmarks_grouped()
    stats = get_stats()

    def _serialize_bm(bm):
        analysis = bm.get("_analysis") or {}
        m = _parse_metrics(bm.get("metrics", {}))
        return {
            "tweet_id": bm.get("tweet_id", ""),
            "author_username": bm.get("author_username", ""),
            "author_name": bm.get("author_name", ""),
            "content": bm.get("content", "")[:500],
            "created_at": bm.get("created_at", ""),
            "fetched_at": bm.get("fetched_at", ""),
            "url": bm.get("url", ""),
            "likes": m.get("likes", 0) or m.get("like_count", 0) or 0,
            "retweets": m.get("retweets", 0) or m.get("retweet_count", 0) or 0,
            "verdict": (analysis.get("verdict") or "").upper().strip(),
            "summary": analysis.get("summary") or "",
            "tags": _parse_tags(analysis.get("tags", [])),
            "actionable": analysis.get("actionable") or "",
            "translation": analysis.get("translation") or "",
            "eli5": analysis.get("eli5") or "",
            "conclusion": analysis.get("conclusion") or "",
            "repo_analysis": analysis.get("repo_analysis") or "",
            "web_research_summary": analysis.get("web_research_summary") or "",
            "claims_check": analysis.get("claims_check") or "",
            "red_flags": analysis.get("red_flags") or "",
            "feature_ideas": analysis.get("feature_ideas") or "",
            "has_analysis": bool(analysis.get("full_analysis_md")),
            "full_md": analysis.get("full_analysis_md") or "",
        }

    return {
        "stats": stats,
        "counts": {
            "signal": len(groups["signal"]),
            "mixed": len(groups["mixed"]),
            "noise": len(groups["noise"]),
            "unanalyzed": len(groups["unanalyzed"]),
            "total": len(all_bm),
        },
        "bookmarks": [_serialize_bm(bm) for bm in all_bm],
        "connected": bool(_get_user_token()),
    }


@app.get("/api/oauth/url")
async def oauth_url():
    if not X_CLIENT_ID:
        return {"error": "No client_id configured"}
    import secrets as sec
    v, ch = generate_pkce_pair()
    state = sec.token_urlsafe(16)
    db_saved = False
    try:
        save_oauth_pkce(state, v)
        db_saved = True
    except Exception as e:
        print(f"[OAUTH] save_pkce error: {e}")
    try:
        _VERIFIER_FILE.write_text(v)
    except Exception:
        pass
    print(f"[OAUTH] Generated auth URL — state={state[:8]}... db_saved={db_saved}")
    url = get_oauth2_auth_url(X_CLIENT_ID, ch, state)
    return {"url": url}


@app.post("/api/oauth/callback")
async def oauth_callback(request: Request):
    """Client-side OAuth callback — exchanges code for tokens in a single request."""
    body = await request.json()
    code = body.get("code", "")
    state = body.get("state", "")
    if not code or not X_CLIENT_ID:
        return JSONResponse({"error": "Missing code or client_id"}, status_code=400)
    verifier = None
    try:
        verifier = pop_oauth_pkce(state)
    except Exception as e:
        print(f"[OAUTH] pop_pkce DB error: {e}")
    if not verifier and _VERIFIER_FILE.exists():
        try:
            verifier = _VERIFIER_FILE.read_text().strip()
            _VERIFIER_FILE.unlink(missing_ok=True)
        except Exception as e:
            print(f"[OAUTH] verifier file read error: {e}")
    if not verifier:
        print(f"[OAUTH] No verifier found for state={state}")
        return JSONResponse({"error": "PKCE verifier not found — please try connecting again"}, status_code=400)
    token_result = exchange_oauth2_code(X_CLIENT_ID, code, verifier)
    if token_result.get("error"):
        print(f"[OAUTH] Token exchange error: {token_result['error']}")
        return JSONResponse({"error": token_result["error"]}, status_code=400)
    saved = _save_tokens(
        token_result["access_token"],
        token_result.get("refresh_token", ""),
    )
    return {"connected": True, "saved": saved}


@app.post("/api/oauth/disconnect")
async def oauth_disconnect():
    _clear_tokens()
    return {"ok": True}


@app.post("/api/import-bookmarks")
def import_bookmarks():
    token = _get_user_token()
    if not token:
        return JSONResponse({"error": "Not connected"}, status_code=401)
    result = fetch_all_bookmarks(token)
    if result.get("error") and "401" in str(result["error"]):
        new_token = _try_refresh()
        if new_token:
            result = fetch_all_bookmarks(new_token)
        else:
            _clear_tokens()
            return JSONResponse({"error": "Session expired"}, status_code=401)
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=400)
    enriched = 0
    for bm in result["bookmarks"]:
        content = bm.get("content", "")
        if re.match(r'^https?://\S+$', content.strip()):
            fx = fetch_tweet_via_fxtwitter(bm["tweet_id"])
            if fx and fx.get("content") and fx["content"] != content:
                bm["content"] = fx["content"]
                enriched += 1
        upsert_bookmark(**{k: v for k, v in bm.items() if k != "error"})
    return {"imported": len(result["bookmarks"]), "enriched": enriched}


@app.post("/api/enrich-articles")
def enrich_articles():
    """Backfill: fetch real content for bookmarks that only have a URL."""
    all_bm = get_all_bookmarks()
    url_only = [b for b in all_bm if re.match(r'^https?://\S+$', (b.get("content") or "").strip())]
    enriched, failed = 0, 0
    errors = []
    for bm in url_only:
        try:
            fx = fetch_tweet_via_fxtwitter(bm["tweet_id"])
            if fx and fx.get("content") and fx["content"] != bm["content"]:
                upsert_bookmark(tweet_id=bm["tweet_id"], author_username=bm.get("author_username", ""),
                                author_name=bm.get("author_name", ""), content=fx["content"],
                                created_at=bm.get("created_at", ""), url=bm.get("url", ""))
                enriched += 1
            else:
                failed += 1
                if len(errors) < 3:
                    errors.append(f"{bm['tweet_id']}: fx={fx is not None}, content={bool(fx and fx.get('content'))}")
        except Exception as e:
            failed += 1
            if len(errors) < 3:
                errors.append(f"{bm['tweet_id']}: {type(e).__name__}: {str(e)[:100]}")
    return {"total_url_only": len(url_only), "enriched": enriched, "failed": failed, "sample_errors": errors}


@app.post("/api/analyze")
async def api_analyze(
    urls: str = Form(""),
    files: list[UploadFile] = File(None),
):
    items = []
    for line in (urls or "").strip().split("\n"):
        u = line.strip()
        if u and u.startswith("http"):
            items.append(("url", u))
    for f in (files or []):
        content = f.file.read().decode("utf-8", errors="replace")
        items.append(("file", f.filename, content))

    results = []
    for item in items:
        try:
            if item[0] == "url":
                url = item[1]
                tweet_id = extract_tweet_id(url)
                if tweet_id:
                    tweet = None
                    if X_BEARER:
                        tweet = fetch_tweet_by_id(tweet_id, X_BEARER)
                        if tweet and tweet.get("error"):
                            tweet = None
                    # Always try FxTwitter if no tweet or content is short
                    # (FxTwitter extracts X Article text that the official API doesn't)
                    if not tweet or len(tweet.get("content", "")) < 200:
                        fx = fetch_tweet_via_fxtwitter(tweet_id)
                        if fx and len(fx.get("content", "")) > len((tweet or {}).get("content", "")):
                            tweet = fx
                    if tweet:
                        upsert_bookmark(**{k: v for k, v in tweet.items() if k != "error"})
                        a = await _analyze_and_save(tweet)
                        results.append({"url": url, "verdict": a.get("verdict", ""), "summary": a.get("summary", "")})
                else:
                    # Try Exa crawl first (handles Medium, paywalls, JS-rendered pages)
                    content_text = ""
                    if EXA_KEY:
                        import httpx as _httpx
                        async with _httpx.AsyncClient() as _client:
                            content_text = await _exa_crawl_async(url, EXA_KEY, _client, max_chars=6000, timeout=15)
                    # Fallback to direct fetch
                    if not content_text:
                        web = fetch_web_content(url)
                        if not web.get("error"):
                            content_text = web.get("content", "")
                    if content_text:
                        fid = "web_" + hashlib.md5(url.encode()).hexdigest()[:12]
                        domain = url.split("/")[2] if len(url.split("/")) > 2 else url
                        upsert_bookmark(tweet_id=fid, author_username=domain,
                                        author_name=domain, content=content_text,
                                        created_at=datetime.utcnow().isoformat(), url=url,
                                        metrics={})
                        bm = {"tweet_id": fid, "content": content_text,
                              "author_username": domain, "url": url}
                        a = await _analyze_and_save(bm)
                        results.append({"url": url, "verdict": a.get("verdict", ""), "summary": a.get("summary", "")})
                    else:
                        results.append({"url": url, "error": "Impossible de récupérer le contenu de cette URL"})
            elif item[0] == "file":
                fname, content = item[1], item[2]
                fid = "file_" + hashlib.md5(content[:500].encode()).hexdigest()[:12]
                upsert_bookmark(tweet_id=fid, author_username="upload", author_name=fname,
                                content=content[:8000], created_at=datetime.utcnow().isoformat(),
                                url=f"file://{fname}", metrics={})
                bm = {"tweet_id": fid, "content": content[:8000], "author_username": fname, "url": ""}
                a = await _analyze_and_save(bm)
                results.append({"file": fname, "verdict": a.get("verdict", ""), "summary": a.get("summary", "")})
        except APIOverloadedError:
            results.append({"error": "overloaded"})
            break
        except Exception as e:
            results.append({"error": str(e)})
    return {"results": results}


@app.post("/api/reanalyze/{tweet_id}")
async def api_reanalyze(tweet_id: str):
    import asyncio
    from database import get_bookmark
    loop = asyncio.get_event_loop()
    bm = get_bookmark(tweet_id)
    if not bm:
        return JSONResponse({"error": "Not found"}, status_code=404)
    # Always re-fetch via FxTwitter — extracts X Articles, quote tweets, cards
    # Cost: ~1s vs ~15s for Claude call, worth it for better content
    fx = await loop.run_in_executor(None, fetch_tweet_via_fxtwitter, tweet_id)
    if fx and len(fx.get("content", "")) > len(bm.get("content", "")):
        bm["content"] = fx["content"]
        upsert_bookmark(**{k: v for k, v in fx.items() if k != "error"})
    try:
        a = await _analyze_and_save(bm)
        return {"verdict": a.get("verdict"), "summary": a.get("summary")}
    except APIOverloadedError:
        return JSONResponse({"error": "overloaded"}, status_code=529)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[API] reanalyze failed for {tweet_id}: {e}\n{tb}")
        return JSONResponse({"error": str(e), "traceback": tb}, status_code=500)


@app.post("/api/reanalyze-batch")
async def api_reanalyze_batch(tweet_ids: list[str]):
    import asyncio
    from database import get_bookmark
    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(10)

    async def _one(tid):
        async with sem:
            bm = get_bookmark(tid)
            if not bm:
                return {"tweet_id": tid, "error": "not found"}
            # Re-fetch if content is too short (just a URL)
            # Run sync FxTwitter in thread to avoid blocking event loop
            # Always re-fetch — extracts articles, quote tweets, cards
            fx = await loop.run_in_executor(None, fetch_tweet_via_fxtwitter, tid)
            if fx and len(fx.get("content", "")) > len(bm.get("content", "")):
                bm["content"] = fx["content"]
                upsert_bookmark(tweet_id=bm["tweet_id"], author_username=bm.get("author_username", ""),
                                author_name=bm.get("author_name", ""), content=fx["content"],
                                created_at=bm.get("created_at", ""), url=bm.get("url", ""))
            try:
                a = await _analyze_and_save(bm)
                return {"tweet_id": tid, "verdict": a.get("verdict"), "summary": a.get("summary", "")}
            except APIOverloadedError:
                return {"tweet_id": tid, "error": "overloaded"}
            except Exception as e:
                print(f"[BATCH] {tid} failed: {e}")
                return {"tweet_id": tid, "error": str(e)}

    results = await asyncio.gather(*[_one(tid) for tid in tweet_ids])
    return {"results": list(results)}


@app.post("/api/brief/generate")
async def api_generate_brief(request_text: str = Form(""), min_quality: int = Form(2)):
    if not request_text.strip():
        return JSONResponse({"error": "Empty request"}, status_code=400)
    relevant = search_analyzed_bookmarks(request_text, min_verdict_score=min_quality)
    brief = await generate_brief_async(request_text, relevant, ANTHROPIC_KEY, exa_key=EXA_KEY)
    if not brief.get("error"):
        save_brief(request_text.strip(), brief)
    return brief


@app.get("/api/briefs")
async def api_briefs():
    briefs = get_all_briefs()
    return [{"id": b["id"], "request": b["request"], "project_name": b.get("project_name", ""),
             "created_at": b["created_at"], "data": b.get("_parsed", {})} for b in briefs]


@app.delete("/api/briefs/{brief_id}")
async def api_delete_brief(brief_id: int):
    delete_brief(brief_id)
    return {"ok": True}


@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    tweet_id = body.get("tweet_id", "")
    history = body.get("history", [])
    if not message:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    if not ANTHROPIC_KEY:
        return JSONResponse({"error": "Anthropic API key not configured"}, status_code=500)
    from database import get_bookmark
    bm = get_bookmark(tweet_id) if tweet_id else None
    analysis = get_analysis(tweet_id) if tweet_id else None
    if not bm:
        return JSONResponse({"error": "Bookmark not found"}, status_code=404)
    try:
        reply = await chat_about_content_async(
            message=message,
            bookmark=bm,
            analysis=analysis or {},
            history=history,
            api_key=ANTHROPIC_KEY,
            exa_key=EXA_KEY,
            apify_key=APIFY_TOKEN,
        )
        return {"reply": reply}
    except Exception as e:
        print(f"[CHAT] Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/brief/chat")
async def api_brief_chat(request: Request):
    body = await request.json()
    brief_id = body.get("brief_id")
    message = body.get("message", "").strip()
    history = body.get("history", [])
    if not message or not brief_id:
        return JSONResponse({"error": "Missing message or brief_id"}, status_code=400)
    if not ANTHROPIC_KEY:
        return JSONResponse({"error": "Anthropic API key not configured"}, status_code=500)
    brief = get_brief(brief_id)
    if not brief:
        return JSONResponse({"error": "Brief not found"}, status_code=404)
    brief_data = brief.get("_parsed", {})

    # Search bookmarks for context if the message seems to need it
    bookmarks_context = ""
    search_keywords = message.split()[:5]
    query = " ".join(search_keywords)
    if len(query) > 5:
        relevant = search_analyzed_bookmarks(query, min_verdict_score=1)
        if relevant:
            parts = []
            for bm in relevant[:8]:
                parts.append(
                    f"[{(bm.get('verdict') or '').upper()}] @{bm.get('author_username', '')}: "
                    f"{(bm.get('summary') or '')[:200]} | "
                    f"Actions: {(bm.get('actionable') or '')[:150]} | "
                    f"URL: {bm.get('url', '')}"
                )
            bookmarks_context = "\n".join(parts)

    try:
        result = await chat_about_brief_async(
            message=message,
            brief_data=brief_data,
            history=history,
            api_key=ANTHROPIC_KEY,
            exa_key=EXA_KEY,
            bookmarks_context=bookmarks_context,
        )
        # If Claude returned brief updates, apply them
        if result.get("updated_fields"):
            updated_data = {**brief_data, **result["updated_fields"]}
            update_brief(brief_id, updated_data)
            result["updated_brief"] = updated_data
        return result
    except Exception as e:
        print(f"[BRIEF_CHAT] Error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.put("/api/briefs/{brief_id}")
async def api_update_brief(brief_id: int, request: Request):
    body = await request.json()
    brief = get_brief(brief_id)
    if not brief:
        return JSONResponse({"error": "Brief not found"}, status_code=404)
    current_data = brief.get("_parsed", {})
    updated_data = {**current_data, **body}
    update_brief(brief_id, updated_data)
    return {"ok": True, "data": updated_data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)

"""Persistence layer — SQLite (local), PostgreSQL (psycopg2), or Supabase REST (Vercel)."""

import os
import json
from datetime import datetime

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")

_SB = bool(SUPABASE_URL and SUPABASE_KEY)
_PG = bool(DATABASE_URL) and not _SB

if _SB:
    from supabase import create_client
    _sb = create_client(SUPABASE_URL, SUPABASE_KEY)
elif _PG:
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3
    from pathlib import Path
    _ON_VERCEL = bool(os.environ.get("VERCEL"))
    DB_PATH = (
        Path("/tmp/bookmarks.db")
        if _ON_VERCEL
        else Path(__file__).parent / "data" / "bookmarks.db"
    )

_ANALYSIS_FIELDS = (
    "id, tweet_id, tags, summary, claims_check, red_flags, actionable, "
    "full_analysis_md, analyzed_at, verdict, translation, repo_analysis, "
    "conclusion, eli5, web_research_summary"
)
_ANALYSIS_FIELDS_WITH_FEATURES = _ANALYSIS_FIELDS + ", feature_ideas"


# ═══════════════════════════════════════════════════════════════════════
#  SUPABASE REST MODE  (HTTPS — works from Vercel, no IPv4/IPv6 issue)
# ═══════════════════════════════════════════════════════════════════════

if _SB:

    _has_feature_ideas = None

    def _analysis_fields():
        global _has_feature_ideas
        if _has_feature_ideas is True:
            return _ANALYSIS_FIELDS_WITH_FEATURES
        if _has_feature_ideas is False:
            return _ANALYSIS_FIELDS
        try:
            _sb.table("analyses").select("feature_ideas").limit(1).execute()
            _has_feature_ideas = True
            return _ANALYSIS_FIELDS_WITH_FEATURES
        except Exception:
            _has_feature_ideas = False
            return _ANALYSIS_FIELDS

    _oauth_table_ok = None

    def _check_oauth_table():
        global _oauth_table_ok
        if _oauth_table_ok is not None:
            return _oauth_table_ok
        try:
            _sb.table("oauth_tokens").select("provider").limit(1).execute()
            _oauth_table_ok = True
        except Exception:
            _oauth_table_ok = False
            print("[INIT] oauth_tokens table missing — using categories fallback")
        return _oauth_table_ok

    def init_db():
        _check_oauth_table()

    def get_connection():
        return None

    def upsert_bookmark(tweet_id, author_username, author_name, content,
                         created_at, url, metrics=None, media_urls=None):
        row = {
            "tweet_id": tweet_id,
            "author_username": author_username,
            "author_name": author_name,
            "content": content,
            "created_at": created_at,
            "url": url,
            "metrics": json.dumps(metrics or {}),
            "media_urls": json.dumps(media_urls or []),
            "fetched_at": datetime.utcnow().isoformat(),
        }
        # Only set fetched_at for NEW bookmarks, don't overwrite for existing
        try:
            _sb.table("bookmarks").insert(row).execute()
        except Exception:
            # Already exists — update without changing fetched_at
            row.pop("fetched_at")
            row.pop("tweet_id")
            _sb.table("bookmarks").update(row).eq("tweet_id", tweet_id).execute()

    def save_analysis(tweet_id, tags, summary, claims_check, red_flags,
                      actionable, full_md, embedding=None, verdict="",
                      translation=None, repo_analysis=None, conclusion=None,
                      eli5=None, web_research_summary=None, feature_ideas=None):
        row = {
            "tweet_id": tweet_id,
            "tags": json.dumps(tags),
            "summary": summary,
            "claims_check": claims_check,
            "red_flags": red_flags,
            "actionable": actionable,
            "full_analysis_md": full_md,
            "analyzed_at": datetime.utcnow().isoformat(),
            "embedding_vector": json.dumps(embedding or []),
            "verdict": verdict or "",
            "translation": translation,
            "repo_analysis": repo_analysis,
            "conclusion": conclusion,
            "eli5": eli5,
            "web_research_summary": web_research_summary,
        }
        if feature_ideas:
            row["feature_ideas"] = feature_ideas
        try:
            _sb.table("analyses").insert(row).execute()
        except Exception as e:
            print(f"[DB] save_analysis 1st insert failed for {tweet_id}: {e}")
            row.pop("feature_ideas", None)
            try:
                _sb.table("analyses").insert(row).execute()
            except Exception as e2:
                print(f"[DB] save_analysis FAILED for {tweet_id}: {e2}")
                raise
        for tag in tags:
            try:
                existing = (
                    _sb.table("categories")
                    .select("name, count")
                    .eq("name", tag)
                    .maybe_single()
                    .execute()
                )
                has_existing = existing and existing.data
            except Exception:
                has_existing = False
            if has_existing:
                _sb.table("categories").update({
                    "count": (existing.data.get("count") or 0) + 1
                }).eq("name", tag).execute()
            else:
                try:
                    _sb.table("categories").upsert({
                        "name": tag, "color": None, "count": 1,
                        "created_at": datetime.utcnow().isoformat(),
                    }, on_conflict="name").execute()
                except Exception:
                    pass

    def get_all_bookmarks():
        resp = (
            _sb.table("bookmarks")
            .select(f"*, analyses({_analysis_fields()})")
            .order("created_at", desc=True)
            .execute()
        )
        rows = []
        for bm in resp.data:
            analyses = bm.pop("analyses", [])
            if analyses:
                latest = max(analyses, key=lambda a: a.get("analyzed_at") or "")
                bm["tags"] = latest.get("tags")
                bm["summary"] = latest.get("summary")
                bm["full_analysis_md"] = latest.get("full_analysis_md")
                bm["analyzed_at"] = latest.get("analyzed_at")
            rows.append(bm)
        return rows

    def get_bookmark(tweet_id):
        try:
            resp = (
                _sb.table("bookmarks")
                .select("*")
                .eq("tweet_id", tweet_id)
                .maybe_single()
                .execute()
            )
            return resp.data if resp else None
        except Exception:
            return None

    def get_analysis(tweet_id):
        try:
            resp = (
                _sb.table("analyses")
                .select("*")
                .eq("tweet_id", tweet_id)
                .order("analyzed_at", desc=True)
                .limit(1)
                .maybe_single()
                .execute()
            )
            return resp.data if resp else None
        except Exception:
            return None

    def get_all_categories():
        resp = _sb.table("categories").select("*").order("count", desc=True).execute()
        return [c for c in (resp.data or []) if not c.get("name", "").startswith("_")]

    def get_all_analyses_with_bookmarks():
        resp = (
            _sb.table("analyses")
            .select(
                "tweet_id, tags, summary, full_analysis_md, analyzed_at, "
                "embedding_vector, bookmarks(author_username, author_name, "
                "content, url, created_at, metrics)"
            )
            .order("analyzed_at", desc=True)
            .execute()
        )
        rows = []
        for a in resp.data:
            bm = a.pop("bookmarks", {}) or {}
            rows.append({**bm, **a})
        return rows

    def get_stats():
        bm = _sb.table("bookmarks").select("tweet_id", count="exact").limit(0).execute()
        an_resp = _sb.table("analyses").select("tweet_id").execute()
        distinct_analyzed = len({a["tweet_id"] for a in an_resp.data})
        cat = _sb.table("categories").select("name", count="exact").limit(0).execute()
        return {
            "total_bookmarks": bm.count or 0,
            "total_analyzed": distinct_analyzed,
            "total_categories": cat.count or 0,
        }

    def bookmark_exists(tweet_id):
        try:
            resp = (
                _sb.table("bookmarks")
                .select("tweet_id")
                .eq("tweet_id", tweet_id)
                .maybe_single()
                .execute()
            )
            return resp is not None and resp.data is not None
        except Exception:
            return False

    def analysis_exists(tweet_id):
        try:
            resp = (
                _sb.table("analyses")
                .select("tweet_id")
                .eq("tweet_id", tweet_id)
                .limit(1)
                .maybe_single()
                .execute()
            )
            return resp is not None and resp.data is not None
        except Exception:
            return False

    def save_brief(request, brief_data):
        _sb.table("briefs").insert({
            "request": request,
            "project_name": brief_data.get("project_name", ""),
            "brief_json": json.dumps(brief_data, ensure_ascii=False),
            "created_at": datetime.utcnow().isoformat(),
        }).execute()

    def get_all_briefs():
        resp = _sb.table("briefs").select("*").order("created_at", desc=True).execute()
        results = []
        for d in resp.data:
            try:
                d["_parsed"] = json.loads(d.get("brief_json", "{}"))
            except Exception:
                d["_parsed"] = {}
            results.append(d)
        return results

    def delete_brief(brief_id):
        _sb.table("briefs").delete().eq("id", brief_id).execute()

    def get_brief(brief_id):
        try:
            resp = _sb.table("briefs").select("*").eq("id", brief_id).maybe_single().execute()
            if resp and resp.data:
                d = resp.data
                try:
                    d["_parsed"] = json.loads(d.get("brief_json", "{}"))
                except Exception:
                    d["_parsed"] = {}
                return d
        except Exception:
            pass
        return None

    def update_brief(brief_id, brief_data):
        _sb.table("briefs").update({
            "brief_json": json.dumps(brief_data, ensure_ascii=False),
            "project_name": brief_data.get("project_name", ""),
        }).eq("id", brief_id).execute()

    def search_analyzed_bookmarks(query, min_verdict_score=1):
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.split() if len(w.strip()) > 2]
        if not keywords:
            return []
        resp = (
            _sb.table("analyses")
            .select(f"*, bookmarks(*)")
            .neq("full_analysis_md", "")
            .order("analyzed_at", desc=True)
            .execute()
        )
        verdict_scores = {"SIGNAL": 3, "MIXED": 2, "NOISE": 1}
        results = []
        for a in resp.data:
            bm = a.pop("bookmarks", {}) or {}
            r = {**bm, **a}
            v = (r.get("verdict") or "").upper().strip()
            v_score = verdict_scores.get(v, 0)
            if v_score < min_verdict_score:
                continue
            searchable = " ".join([
                (r.get(f) or "") for f in
                ("content", "summary", "tags", "actionable", "conclusion",
                 "repo_analysis", "eli5", "web_research_summary")
            ]).lower()
            hits = sum(1 for kw in keywords if kw in searchable)
            if hits > 0:
                r["_relevance"] = hits * v_score
                results.append(r)
        results.sort(key=lambda x: x["_relevance"], reverse=True)
        return results[:20]

    def get_bookmarks_grouped():
        resp = (
            _sb.table("bookmarks")
            .select(f"*, analyses({_analysis_fields()})")
            .order("created_at", desc=True)
            .execute()
        )
        groups = {"signal": [], "mixed": [], "noise": [], "unanalyzed": []}
        all_bm = []
        for bm in resp.data:
            analyses = bm.pop("analyses", [])
            analysis = None
            if analyses:
                analysis = max(analyses, key=lambda a: a.get("analyzed_at") or "")
                bm["tags"] = analysis.get("tags")
                bm["summary"] = analysis.get("summary")
                bm["full_analysis_md"] = analysis.get("full_analysis_md")
                bm["analyzed_at"] = analysis.get("analyzed_at")
            bm["_analysis"] = analysis
            all_bm.append(bm)

            if not analysis or not analysis.get("full_analysis_md"):
                groups["unanalyzed"].append(bm)
            else:
                v = (analysis.get("verdict") or "").upper().strip()
                if not v:
                    md = analysis.get("full_analysis_md", "")
                    if "SIGNAL" in md:
                        v = "SIGNAL"
                    elif "NOISE" in md:
                        v = "NOISE"
                    else:
                        v = "MIXED"
                if v == "SIGNAL":
                    groups["signal"].append(bm)
                elif v == "NOISE":
                    groups["noise"].append(bm)
                else:
                    groups["mixed"].append(bm)
        return groups, all_bm

    def save_oauth_pkce(state: str, verifier: str):
        if not state:
            return
        try:
            key = f"_pkce_{state[:20]}"
            _sb.table("categories").upsert({
                "name": key,
                "color": json.dumps({"s": state, "v": verifier}),
                "count": 0,
                "created_at": datetime.utcnow().isoformat(),
            }, on_conflict="name").execute()
            print(f"[PKCE] saved verifier for state={state[:8]}...")
        except Exception as e:
            print(f"[PKCE] save error: {e}")

    def pop_oauth_pkce(state: str):
        if not state:
            return None
        try:
            key = f"_pkce_{state[:20]}"
            r = _sb.table("categories").select("color").eq("name", key).maybe_single().execute()
            if not r or not r.data or not r.data.get("color"):
                print(f"[PKCE] no verifier found for state={state[:8]}...")
                return None
            data = json.loads(r.data["color"])
            v = data.get("v")
            _sb.table("categories").delete().eq("name", key).execute()
            print(f"[PKCE] popped verifier for state={state[:8]}...")
            return v
        except Exception as e:
            print(f"[PKCE] pop error: {e}")
            return None

    def get_oauth_tokens(provider: str = "x"):
        if _check_oauth_table():
            try:
                r = (
                    _sb.table("oauth_tokens")
                    .select("access_token, refresh_token")
                    .eq("provider", provider)
                    .maybe_single()
                    .execute()
                )
                if r and r.data and r.data.get("access_token"):
                    return {
                        "access_token": r.data.get("access_token") or "",
                        "refresh_token": r.data.get("refresh_token") or "",
                    }
            except Exception as e:
                print(f"[OAUTH-DB] get_oauth_tokens error: {e}")
        try:
            key = f"_oauth_{provider}"
            r = _sb.table("categories").select("color").eq("name", key).maybe_single().execute()
            if r and r.data and r.data.get("color"):
                data = json.loads(r.data["color"])
                return {"access_token": data.get("a", ""), "refresh_token": data.get("r", "")}
        except Exception as e:
            print(f"[OAUTH-DB] categories fallback read error: {e}")
        return {}

    def set_oauth_tokens(provider: str, access_token: str, refresh_token: str = ""):
        if _check_oauth_table():
            try:
                _sb.table("oauth_tokens").upsert({
                    "provider": provider,
                    "access_token": access_token,
                    "refresh_token": refresh_token or "",
                    "updated_at": datetime.utcnow().isoformat(),
                }, on_conflict="provider").execute()
                print(f"[OAUTH-DB] set via oauth_tokens table: ok")
                return
            except Exception as e:
                print(f"[OAUTH-DB] oauth_tokens upsert error: {e}")
        try:
            key = f"_oauth_{provider}"
            payload = json.dumps({"a": access_token, "r": refresh_token or ""})
            _sb.table("categories").upsert({
                "name": key,
                "color": payload,
                "count": 0,
                "created_at": datetime.utcnow().isoformat(),
            }, on_conflict="name").execute()
            print(f"[OAUTH-DB] set via categories fallback: ok")
        except Exception as e:
            print(f"[OAUTH-DB] categories fallback write error: {e}")
            raise

    def clear_oauth_tokens(provider: str = "x"):
        if _check_oauth_table():
            try:
                _sb.table("oauth_tokens").delete().eq("provider", provider).execute()
            except Exception:
                pass
        try:
            _sb.table("categories").delete().eq("name", f"_oauth_{provider}").execute()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
#  PSYCOPG2 / SQLITE MODE  (existing implementation, unchanged)
# ═══════════════════════════════════════════════════════════════════════

else:

    def _q(sql):
        return sql.replace("?", "%s") if _PG else sql

    def get_connection():
        if _PG:
            conn = psycopg2.connect(DATABASE_URL, options="-c statement_timeout=30000")
            conn.autocommit = False
            return conn
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _fetchall(conn, sql, params=()):
        if _PG:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(_q(sql), params)
            rows = cur.fetchall()
            cur.close()
            return rows
        return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def _fetchone(conn, sql, params=()):
        if _PG:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(_q(sql), params)
            row = cur.fetchone()
            cur.close()
            return row
        r = conn.execute(sql, params).fetchone()
        return dict(r) if r else None

    def _execute(conn, sql, params=()):
        if _PG:
            cur = conn.cursor()
            cur.execute(_q(sql), params)
            cur.close()
        else:
            conn.execute(sql, params)

    def init_db():
        conn = get_connection()
        if _PG:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bookmarks (
                    tweet_id TEXT PRIMARY KEY,
                    author_username TEXT, author_name TEXT,
                    content TEXT, created_at TEXT, url TEXT,
                    metrics TEXT, media_urls TEXT, fetched_at TEXT,
                    source_type TEXT DEFAULT 'x'
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id SERIAL PRIMARY KEY,
                    tweet_id TEXT NOT NULL REFERENCES bookmarks(tweet_id),
                    tags TEXT, summary TEXT, claims_check TEXT,
                    red_flags TEXT, actionable TEXT, full_analysis_md TEXT,
                    analyzed_at TEXT, embedding_vector TEXT,
                    verdict TEXT DEFAULT '',
                    translation TEXT, repo_analysis TEXT,
                    conclusion TEXT, eli5 TEXT, web_research_summary TEXT,
                    feature_ideas TEXT
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_analyses_tweet ON analyses(tweet_id)")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS categories (
                    name TEXT PRIMARY KEY, color TEXT,
                    count INTEGER DEFAULT 0, created_at TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS briefs (
                    id SERIAL PRIMARY KEY,
                    request TEXT NOT NULL, project_name TEXT,
                    brief_json TEXT NOT NULL, created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS oauth_pkce (
                    state TEXT PRIMARY KEY,
                    verifier TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS oauth_tokens (
                    provider TEXT PRIMARY KEY,
                    access_token TEXT,
                    refresh_token TEXT,
                    updated_at TEXT
                )
            """)
            cur.close()
            conn.commit()
        else:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS bookmarks (
                    tweet_id TEXT PRIMARY KEY, author_username TEXT,
                    author_name TEXT, content TEXT, created_at TEXT,
                    url TEXT, metrics TEXT, media_urls TEXT, fetched_at TEXT
                );
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tweet_id TEXT NOT NULL, tags TEXT, summary TEXT,
                    claims_check TEXT, red_flags TEXT, actionable TEXT,
                    full_analysis_md TEXT, analyzed_at TEXT, embedding_vector TEXT,
                    FOREIGN KEY (tweet_id) REFERENCES bookmarks(tweet_id)
                );
                CREATE TABLE IF NOT EXISTS categories (
                    name TEXT PRIMARY KEY, color TEXT,
                    count INTEGER DEFAULT 0, created_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_analyses_tweet ON analyses(tweet_id);
                CREATE TABLE IF NOT EXISTS briefs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request TEXT NOT NULL, project_name TEXT,
                    brief_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oauth_pkce (
                    state TEXT PRIMARY KEY,
                    verifier TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS oauth_tokens (
                    provider TEXT PRIMARY KEY,
                    access_token TEXT,
                    refresh_token TEXT,
                    updated_at TEXT
                );
            """)
            for col, typedef in [("verdict", "TEXT DEFAULT ''"), ("translation", "TEXT"),
                                  ("repo_analysis", "TEXT"), ("conclusion", "TEXT"),
                                  ("eli5", "TEXT"), ("web_research_summary", "TEXT"),
                                  ("feature_ideas", "TEXT")]:
                try:
                    conn.execute(f"ALTER TABLE analyses ADD COLUMN {col} {typedef}")
                    conn.commit()
                except sqlite3.OperationalError:
                    pass
            try:
                conn.execute("ALTER TABLE bookmarks ADD COLUMN source_type TEXT DEFAULT 'x'")
                conn.commit()
            except sqlite3.OperationalError:
                pass
        conn.close()

    def upsert_bookmark(tweet_id, author_username, author_name, content,
                         created_at, url, metrics=None, media_urls=None):
        conn = get_connection()
        _execute(conn, """
            INSERT INTO bookmarks (tweet_id, author_username, author_name, content,
                                   created_at, url, metrics, media_urls, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tweet_id) DO UPDATE SET
                content = EXCLUDED.content, metrics = EXCLUDED.metrics,
                fetched_at = EXCLUDED.fetched_at
        """, (tweet_id, author_username, author_name, content, created_at, url,
              json.dumps(metrics or {}), json.dumps(media_urls or []),
              datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

    def save_analysis(tweet_id, tags, summary, claims_check, red_flags,
                      actionable, full_md, embedding=None, verdict="",
                      translation=None, repo_analysis=None, conclusion=None,
                      eli5=None, web_research_summary=None, feature_ideas=None):
        conn = get_connection()
        _execute(conn, """
            INSERT INTO analyses (tweet_id, tags, summary, claims_check, red_flags,
                                  actionable, full_analysis_md, analyzed_at, embedding_vector,
                                  verdict, translation, repo_analysis, conclusion, eli5,
                                  web_research_summary, feature_ideas)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tweet_id, json.dumps(tags), summary, claims_check, red_flags,
              actionable, full_md, datetime.utcnow().isoformat(),
              json.dumps(embedding or []), verdict or "", translation,
              repo_analysis, conclusion, eli5, web_research_summary,
              feature_ideas))
        for tag in tags:
            _execute(conn, """
                INSERT INTO categories (name, color, count, created_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(name) DO UPDATE SET count = count + 1
            """, (tag, None, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

    def get_all_bookmarks():
        conn = get_connection()
        rows = _fetchall(conn, """
            SELECT b.*, a.tags, a.summary, a.full_analysis_md, a.analyzed_at
            FROM bookmarks b
            LEFT JOIN (
                SELECT tweet_id, tags, summary, full_analysis_md, analyzed_at,
                       ROW_NUMBER() OVER (PARTITION BY tweet_id ORDER BY analyzed_at DESC) AS rn
                FROM analyses
            ) a ON b.tweet_id = a.tweet_id AND a.rn = 1
            ORDER BY b.created_at DESC
        """)
        conn.close()
        return rows

    def get_bookmark(tweet_id):
        conn = get_connection()
        row = _fetchone(conn, "SELECT * FROM bookmarks WHERE tweet_id = ?", (tweet_id,))
        conn.close()
        return row

    def get_analysis(tweet_id):
        conn = get_connection()
        row = _fetchone(conn,
            "SELECT * FROM analyses WHERE tweet_id = ? ORDER BY analyzed_at DESC LIMIT 1",
            (tweet_id,))
        conn.close()
        return row

    def get_all_categories():
        conn = get_connection()
        rows = _fetchall(conn, "SELECT * FROM categories ORDER BY count DESC")
        conn.close()
        return rows

    def get_all_analyses_with_bookmarks():
        conn = get_connection()
        rows = _fetchall(conn, """
            SELECT b.tweet_id, b.author_username, b.author_name, b.content, b.url,
                   b.created_at, b.metrics,
                   a.tags, a.summary, a.full_analysis_md, a.analyzed_at, a.embedding_vector
            FROM analyses a JOIN bookmarks b ON a.tweet_id = b.tweet_id
            ORDER BY a.analyzed_at DESC
        """)
        conn.close()
        return rows

    def get_stats():
        conn = get_connection()
        total_bm = _fetchone(conn, "SELECT COUNT(*) as c FROM bookmarks")["c"]
        total_an = _fetchone(conn, "SELECT COUNT(DISTINCT tweet_id) as c FROM analyses")["c"]
        total_cat = _fetchone(conn, "SELECT COUNT(*) as c FROM categories")["c"]
        conn.close()
        return {"total_bookmarks": total_bm, "total_analyzed": total_an,
                "total_categories": total_cat}

    def bookmark_exists(tweet_id):
        conn = get_connection()
        row = _fetchone(conn, "SELECT 1 as x FROM bookmarks WHERE tweet_id = ?", (tweet_id,))
        conn.close()
        return row is not None

    def analysis_exists(tweet_id):
        conn = get_connection()
        row = _fetchone(conn, "SELECT 1 as x FROM analyses WHERE tweet_id = ?", (tweet_id,))
        conn.close()
        return row is not None

    def save_brief(request, brief_data):
        conn = get_connection()
        _execute(conn, """
            INSERT INTO briefs (request, project_name, brief_json, created_at)
            VALUES (?, ?, ?, ?)
        """, (request, brief_data.get("project_name", ""),
              json.dumps(brief_data, ensure_ascii=False),
              datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

    def get_all_briefs():
        conn = get_connection()
        rows = _fetchall(conn, "SELECT * FROM briefs ORDER BY created_at DESC")
        conn.close()
        results = []
        for d in rows:
            try:
                d["_parsed"] = json.loads(d.get("brief_json", "{}"))
            except Exception:
                d["_parsed"] = {}
            results.append(d)
        return results

    def delete_brief(brief_id):
        conn = get_connection()
        _execute(conn, "DELETE FROM briefs WHERE id = ?", (brief_id,))
        conn.commit()
        conn.close()

    def get_brief(brief_id):
        conn = get_connection()
        row = _fetchone(conn, "SELECT * FROM briefs WHERE id = ?", (brief_id,))
        conn.close()
        if not row:
            return None
        try:
            row["_parsed"] = json.loads(row.get("brief_json", "{}"))
        except Exception:
            row["_parsed"] = {}
        return row

    def update_brief(brief_id, brief_data):
        conn = get_connection()
        _execute(conn, """
            UPDATE briefs SET brief_json = ?, project_name = ? WHERE id = ?
        """, (json.dumps(brief_data, ensure_ascii=False),
              brief_data.get("project_name", ""), brief_id))
        conn.commit()
        conn.close()

    def search_analyzed_bookmarks(query, min_verdict_score=1):
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.split() if len(w.strip()) > 2]
        if not keywords:
            return []
        conn = get_connection()
        rows = _fetchall(conn, """
            SELECT b.*, a.tags, a.summary, a.actionable, a.verdict, a.conclusion,
                   a.eli5, a.repo_analysis, a.full_analysis_md, a.web_research_summary,
                   a.claims_check, a.red_flags, a.translation, a.analyzed_at
            FROM bookmarks b JOIN analyses a ON b.tweet_id = a.tweet_id
            WHERE a.full_analysis_md IS NOT NULL AND a.full_analysis_md != ''
            ORDER BY a.analyzed_at DESC
        """)
        conn.close()
        verdict_scores = {"SIGNAL": 3, "MIXED": 2, "NOISE": 1}
        results = []
        for r in rows:
            v = (r.get("verdict") or "").upper().strip()
            v_score = verdict_scores.get(v, 0)
            if v_score < min_verdict_score:
                continue
            searchable = " ".join([(r.get(f) or "") for f in
                ("content", "summary", "tags", "actionable", "conclusion",
                 "repo_analysis", "eli5", "web_research_summary")]).lower()
            hits = sum(1 for kw in keywords if kw in searchable)
            if hits > 0:
                r["_relevance"] = hits * v_score
                results.append(r)
        results.sort(key=lambda x: x["_relevance"], reverse=True)
        return results[:20]

    def get_bookmarks_grouped():
        all_bm = get_all_bookmarks()
        groups = {"signal": [], "mixed": [], "noise": [], "unanalyzed": []}
        for bm in all_bm:
            analysis = get_analysis(bm["tweet_id"]) if bm.get("analyzed_at") else None
            bm["_analysis"] = analysis
            if not analysis or not analysis.get("full_analysis_md"):
                groups["unanalyzed"].append(bm)
            else:
                v = (analysis.get("verdict") or "").upper().strip()
                if not v:
                    md = analysis.get("full_analysis_md", "")
                    if "SIGNAL" in md:
                        v = "SIGNAL"
                    elif "NOISE" in md:
                        v = "NOISE"
                    else:
                        v = "MIXED"
                if v == "SIGNAL":
                    groups["signal"].append(bm)
                elif v == "NOISE":
                    groups["noise"].append(bm)
                else:
                    groups["mixed"].append(bm)
        return groups, all_bm

    def save_oauth_pkce(state: str, verifier: str):
        if not state:
            return
        conn = get_connection()
        _execute(conn, """
            INSERT INTO oauth_pkce (state, verifier, created_at) VALUES (?, ?, ?)
            ON CONFLICT(state) DO UPDATE SET
                verifier = EXCLUDED.verifier, created_at = EXCLUDED.created_at
        """, (state, verifier, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

    def pop_oauth_pkce(state: str):
        if not state:
            return None
        conn = get_connection()
        row = _fetchone(conn, "SELECT verifier FROM oauth_pkce WHERE state = ?", (state,))
        if row:
            _execute(conn, "DELETE FROM oauth_pkce WHERE state = ?", (state,))
        conn.commit()
        conn.close()
        return row["verifier"] if row else None

    def get_oauth_tokens(provider: str = "x"):
        conn = get_connection()
        row = _fetchone(
            conn,
            "SELECT access_token, refresh_token FROM oauth_tokens WHERE provider = ?",
            (provider,),
        )
        conn.close()
        if not row:
            return {}
        return {
            "access_token": row.get("access_token") or "",
            "refresh_token": row.get("refresh_token") or "",
        }

    def set_oauth_tokens(provider: str, access_token: str, refresh_token: str = ""):
        conn = get_connection()
        _execute(conn, """
            INSERT INTO oauth_tokens (provider, access_token, refresh_token, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(provider) DO UPDATE SET
                access_token = EXCLUDED.access_token,
                refresh_token = EXCLUDED.refresh_token,
                updated_at = EXCLUDED.updated_at
        """, (provider, access_token, refresh_token or "", datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()

    def clear_oauth_tokens(provider: str = "x"):
        conn = get_connection()
        _execute(conn, "DELETE FROM oauth_tokens WHERE provider = ?", (provider,))
        conn.commit()
        conn.close()

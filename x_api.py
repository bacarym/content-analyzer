"""X (Twitter) API v2 integration — OAuth 2.0 PKCE for bookmarks."""

import re
import os
import hashlib
import base64
import requests
from typing import Optional
from urllib.parse import urlencode


REDIRECT_URI = os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:8501")
SCOPES = "bookmark.read users.read tweet.read offline.access"


def extract_tweet_id(url: str) -> Optional[str]:
    patterns = [
        r'(?:twitter\.com|x\.com)/\w+/status/(\d+)',
        r'^(\d+)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return None


# ─── OAuth 2.0 PKCE ───

def generate_pkce_pair() -> tuple[str, str]:
    code_verifier = base64.urlsafe_b64encode(os.urandom(40)).rstrip(b"=").decode()
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return code_verifier, code_challenge


def get_oauth2_auth_url(client_id: str, code_challenge: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return f"https://twitter.com/i/oauth2/authorize?{urlencode(params)}"


def exchange_oauth2_code(client_id: str, code: str,
                         code_verifier: str) -> dict:
    resp = requests.post(
        "https://api.twitter.com/2/oauth2/token",
        data={
            "code": code,
            "grant_type": "authorization_code",
            "client_id": client_id,
            "redirect_uri": REDIRECT_URI,
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    if resp.status_code != 200:
        return {"error": f"Token exchange error {resp.status_code}: {resp.text}"}
    return resp.json()


def refresh_oauth2_token(client_id: str, refresh_token: str) -> dict:
    resp = requests.post(
        "https://api.twitter.com/2/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    if resp.status_code != 200:
        return {"error": f"Refresh error {resp.status_code}: {resp.text}"}
    return resp.json()


# ─── Tweet fetching ───

def _bearer_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def fetch_tweet_by_id(tweet_id: str, bearer_token: str) -> Optional[dict]:
    url = f"https://api.x.com/2/tweets/{tweet_id}"
    params = {
        "tweet.fields": "created_at,public_metrics,author_id,conversation_id,entities",
        "expansions": "author_id",
        "user.fields": "name,username,profile_image_url",
    }
    resp = requests.get(url, headers=_bearer_headers(bearer_token),
                        params=params, timeout=15)
    if resp.status_code != 200:
        return {"error": f"X API error {resp.status_code}: {resp.text}"}
    data = resp.json()
    tweet = data.get("data", {})
    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
    author = users.get(tweet.get("author_id"), {})
    return _format_tweet(tweet, author)


def _extract_article_text(data: dict) -> str:
    """Extract full text from X article (long-form tweet) via FxTwitter."""
    article = data.get("article", {})
    if not article:
        return ""
    blocks = article.get("content", {}).get("blocks", [])
    if not blocks:
        return article.get("preview_text", "")
    title = article.get("title", "")
    parts = [title] if title else []
    for block in blocks:
        text = block.get("text", "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def fetch_tweet_via_fxtwitter(tweet_id: str) -> Optional[dict]:
    url = f"https://api.fxtwitter.com/status/{tweet_id}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json().get("tweet", {})
        # Use article content if available, otherwise regular text
        content = _extract_article_text(data) or data.get("text", "")
        return {
            "tweet_id": str(data.get("id", tweet_id)),
            "author_username": data.get("author", {}).get("screen_name", "unknown"),
            "author_name": data.get("author", {}).get("name", "Unknown"),
            "content": content,
            "created_at": data.get("created_at", ""),
            "url": data.get("url", f"https://x.com/i/status/{tweet_id}"),
            "metrics": {
                "likes": data.get("likes", 0),
                "retweets": data.get("retweets", 0),
                "replies": data.get("replies", 0),
            },
            "media_urls": [m.get("url") for m in data.get("media", {}).get("all", []) if m.get("url")],
        }
    except Exception:
        return None


# ─── Bookmarks (requires OAuth 2.0 user token) ───

def fetch_bookmarks(user_token: str, max_results: int = 100,
                    pagination_token: str = None) -> dict:
    headers = _bearer_headers(user_token)
    me_resp = requests.get("https://api.x.com/2/users/me",
                           headers=headers, timeout=10)
    if me_resp.status_code != 200:
        return {"error": f"Auth error {me_resp.status_code}: {me_resp.text}",
                "bookmarks": [], "next_token": None}
    user_id = me_resp.json()["data"]["id"]

    url = f"https://api.x.com/2/users/{user_id}/bookmarks"
    params = {
        "tweet.fields": "created_at,public_metrics,author_id,conversation_id,entities",
        "expansions": "author_id",
        "user.fields": "name,username,profile_image_url",
        "max_results": min(max_results, 100),
    }
    if pagination_token:
        params["pagination_token"] = pagination_token
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    if resp.status_code != 200:
        return {"error": f"Bookmarks API error {resp.status_code}: {resp.text}",
                "bookmarks": [], "next_token": None}
    data = resp.json()
    tweets = data.get("data", [])
    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
    next_token = data.get("meta", {}).get("next_token")
    bookmarks = [_format_tweet(t, users.get(t.get("author_id"), {})) for t in tweets]
    return {"bookmarks": bookmarks, "next_token": next_token, "error": None}


def fetch_all_bookmarks(user_token: str, progress_callback=None) -> dict:
    all_bookmarks = []
    next_token = None
    page = 0
    while True:
        result = fetch_bookmarks(user_token, max_results=100,
                                 pagination_token=next_token)
        if result.get("error"):
            if page == 0:
                return {"error": result["error"], "bookmarks": []}
            break
        all_bookmarks.extend(result["bookmarks"])
        page += 1
        if progress_callback:
            progress_callback(len(all_bookmarks), page)
        next_token = result.get("next_token")
        if not next_token:
            break
    return {"bookmarks": all_bookmarks, "error": None}


def _format_tweet(tweet: dict, author: dict) -> dict:
    metrics = tweet.get("public_metrics", {})
    tweet_id = tweet.get("id", "")
    username = author.get("username", "unknown")
    # Expand t.co URLs to real URLs using entities from X API v2
    content = tweet.get("text", "")
    for ue in tweet.get("entities", {}).get("urls", []):
        tco = ue.get("url", "")
        expanded = ue.get("expanded_url", "")
        if tco and expanded:
            content = content.replace(tco, expanded)
    return {
        "tweet_id": tweet_id,
        "author_username": username,
        "author_name": author.get("name", "Unknown"),
        "content": content,
        "created_at": tweet.get("created_at", ""),
        "url": f"https://x.com/{username}/status/{tweet_id}",
        "metrics": {
            "likes": metrics.get("like_count", 0),
            "retweets": metrics.get("retweet_count", 0),
            "replies": metrics.get("reply_count", 0),
            "views": metrics.get("impression_count", 0),
        },
        "media_urls": [],
    }

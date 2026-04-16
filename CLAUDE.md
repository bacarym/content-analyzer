# Content Analyzer

## Description

App d'analyse de bookmarks X avec Claude + recherche autonome Exa. Trie Signal vs Noise dans les bookmarks tech/AI de Bacary.

- **Status:** Live sur Vercel — content-analyzer-xi.vercel.app
- **Repo:** github.com/bacarym/content-analyzer

## Stack

- **Backend:** FastAPI (Python 3.12) — `server.py`
- **Frontend:** Alpine.js + Tailwind CSS — `templates/index.html` (SPA single-file)
- **Database:** Supabase (PostgreSQL) — `database.py`
- **Analyse:** Claude Sonnet (AsyncAnthropic + prompt caching) — `analyzer.py`
- **Recherche:** Exa API (search + crawl, full async parallel) — `analyzer.py`
- **Auth X:** OAuth 2.0 PKCE — `x_api.py` + `server.py`
- **Déploiement:** Vercel serverless — `vercel.json` + `api/index.py`

## Architecture clé

- `server.py` — FastAPI routes, OAuth flow, API endpoints
- `analyzer.py` — Pipeline d'analyse async (URL resolution, GitHub context, Exa research, Claude analysis). Toute l'I/O en parallèle via `asyncio.gather` + `httpx`
- `database.py` — Couche Supabase (bookmarks, analyses, OAuth tokens/PKCE)
- `x_api.py` — API X v2 + fallback FxTwitter
- `templates/index.html` — SPA Alpine.js complète (filtres, chat, briefs, detail view)

## Patterns importants

- **Date filter:** bouton "OK" + `$refs` pour lire les valeurs DOM (les events natifs des date pickers ne sont PAS fiables avec Alpine.js — ne jamais utiliser `x-model` ou `@input`/`@change` sur `type="date"`)
- **`dateFrom`/`dateTo` sont des props top-level** du composant Alpine (pas imbriquées) pour garantir la réactivité
- **Prompt caching:** `ANALYSIS_SYSTEM_PROMPT` et `BRIEF_SYSTEM_PROMPT` wrappés avec `cache_control: {"type": "ephemeral"}`
- **Verdict strict:** 2+ red flags → NOISE, marketing exagéré → NOISE, scepticisme par défaut
- **Batch re-analyze:** progress bar partagée via `analyzing` + `analyzeProgress`
- **OAuth persistence:** tokens dans Supabase `oauth_tokens` table, PKCE dans `oauth_pkce`

## Règles

- Toujours répondre en français
- Ne jamais utiliser `rm -rf` → utiliser `trash`
- Workflow obligatoire: Analyze → Plan → Execute → Verify

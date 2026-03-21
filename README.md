# Content Analyzer

Analyse, catégorise et explore tes bookmarks X avec Claude + recherche autonome Exa.

**Live** : [content-analyzer-xi.vercel.app](https://content-analyzer-xi.vercel.app)

## Features

- **Bookmarks X** — Connexion OAuth 2.0 PKCE, fetch automatique des bookmarks
- **Analyse Claude** — Résumé ELI5, claims check, red flags, web research, conclusion avec verdict Signal/Mixed/Noise
- **Recherche autonome Exa** — Crawl GitHub repos, recherche ciblée de critiques, sentiment communautaire, crédibilité auteur
- **Feature Ideas** — Claude suggère des idées de features ou projets basés sur le contenu analysé
- **Chat contextuel** — Chat avec Claude dans le contexte d'un contenu analysé
- **Briefs** — Génération de briefs de recherche à partir de bookmarks sélectionnés
- **Filtres** — Par verdict (Signal/Mixed/Noise), topic, plage de dates, recherche texte
- **Re-analyse batch** — Re-analyser tout ou seulement les résultats filtrés, avec progress bar

## Stack

| Couche      | Techno                                         |
| ----------- | ---------------------------------------------- |
| Backend     | FastAPI (Python 3.12)                          |
| Frontend    | Alpine.js + Tailwind CSS                       |
| Database    | Supabase (PostgreSQL)                          |
| Analyse     | Claude Sonnet (Anthropic API) + prompt caching |
| Recherche   | Exa API (search + crawl)                       |
| Auth X      | OAuth 2.0 PKCE                                 |
| Déploiement | Vercel (serverless)                            |

## Architecture

```
content-analyzer/
├── server.py           # FastAPI backend — routes, OAuth, endpoints API
├── analyzer.py         # Pipeline d'analyse Claude + recherche Exa (full async)
├── database.py         # Couche persistance Supabase (bookmarks, analyses, OAuth)
├── x_api.py            # Intégration API X v2 + fallback FxTwitter
├── similarity.py       # Détection doublons TF-IDF
├── templates/
│   └── index.html      # SPA Alpine.js + Tailwind (rendu côté serveur)
├── api/
│   └── index.py        # Entry point Vercel serverless
├── sql/
│   └── oauth_tables.sql # Tables OAuth pour Supabase
├── vercel.json         # Config déploiement Vercel
└── requirements.txt
```

## Setup local

### Prérequis

- Python 3.10+
- Clé API Anthropic
- Clé API Exa (pour la recherche autonome)
- Projet Supabase (URL + service key)
- App OAuth X (client ID + secret)

### Installation

```bash
pip install -r requirements.txt
```

### Variables d'environnement

```bash
ANTHROPIC_API_KEY=sk-ant-...
EXA_API_KEY=...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
X_CLIENT_ID=...
X_CLIENT_SECRET=...
X_REDIRECT_URI=https://your-domain/api/oauth/callback
```

### Lancement

```bash
uvicorn server:app --reload
```

L'app s'ouvre sur `http://localhost:8000`.

## Déploiement Vercel

Le projet est configuré pour Vercel via `vercel.json`. Toutes les routes passent par `api/index.py` qui importe `server.py`.

### Setup Supabase (une seule fois)

Exécuter `sql/oauth_tables.sql` dans le SQL Editor Supabase pour créer les tables `oauth_pkce` et `oauth_tokens`.

## Notes techniques

- L'analyse utilise **Claude Sonnet** avec **prompt caching** pour réduire la latence
- Toute la recherche Exa (search, crawl, GitHub) tourne en **parallèle async** via `asyncio.gather`
- La recherche ciblée inclut : critiques du projet, sentiment Reddit/HN, crédibilité auteur
- Le verdict est **strict** : 2+ red flags → NOISE, marketing exagéré → NOISE
- Le filtre date utilise un **bouton "OK"** + `$refs` (les events natifs des date pickers ne sont pas fiables avec Alpine.js)

---

_Built for Bacary · 2026_

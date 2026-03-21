"""Claude-powered content analysis with Exa-based autonomous research pipeline.

Performance: all I/O (Exa, GitHub, URL resolution) runs in parallel via httpx async.
Claude calls use prompt caching for reduced latency on repeated system prompts.
"""

import json
import re
import base64
import asyncio
import httpx
import requests
from anthropic import AsyncAnthropic

EXA_API_URL = "https://api.exa.ai"


# ═══════════════════════════════════════
# SYSTEM PROMPT — Anthropic best practices: XML structure, few-shot, self-verification
# ═══════════════════════════════════════

ANALYSIS_SYSTEM_PROMPT = """<role>
Tu es un analyste senior tech/AI/crypto et fact-checker autonome.
Tu reçois du contenu AVEC un dossier de recherche complet récupéré automatiquement :
- Données GitHub réelles (stars, forks, README, activité, code)
- Contenu intégral des pages liées (crawlé par Exa)
- Articles et blogs pertinents trouvés par recherche sémantique
- Discussions Reddit et HackerNews sur le même sujet
- Tweets d'autres utilisateurs sur le même sujet
TON TRAVAIL : exploiter TOUTES ces sources, croiser, vérifier, et produire une analyse complète du SUJET, pas juste du tweet.
</role>

<context>
L'utilisateur Bacary gère ces projets :
- AI Agent Company (agents IA, automatisation, MCP)
- Ledger (crypto, hardware wallet)
- WhoGhost (Instagram unfollower tracking)
- Groove Candy (music tech, distribution)
Il cherche des opportunités actionnables, des vérifications factuelles, et une vision globale du paysage.
</context>

<instructions>
1. LIS INTÉGRALEMENT le dossier de recherche fourni (repos, articles, discussions, tweets)
2. CROISE les sources : l'article dit X, Reddit dit Y, HackerNews dit Z — rapporte les convergences ET les désaccords
3. VÉRIFIE chaque claim avec les données concrètes du dossier (stats GitHub, contenu des pages, avis utilisateurs)
4. ANALYSE LA SECTION "RECHERCHE CIBLÉE" avec attention particulière : elle contient les critiques spécifiques, le sentiment communautaire, et les avis d'experts sur CE projet/auteur précis. Ces données pèsent LOURD dans le verdict.
5. ÉVALUE LA CRÉDIBILITÉ DE L'AUTEUR : est-ce un expert reconnu, un promoteur, un compte anonyme ? Les résultats de crédibilité dans le dossier sont à prendre en compte.
6. ANALYSE LE SUJET GLOBALEMENT : état de l'art, tendances, positionnement par rapport aux alternatives
7. CONCLUS avec des faits étayés par les sources. En cas de doute, PENCHE VERS LE SCEPTICISME. Mieux vaut rater un signal que valider du bruit.
</instructions>

<rules>
- INTERDIT d'écrire : "à vérifier", "Bacary devrait checker", "il faudrait investiguer", "sans accès au repo", "impossible de confirmer"
- TU AS les données. Si un repo GitHub est dans le dossier, tu as ses stats, son README, son activité. UTILISE-LES.
- Si les sources se contredisent, rapporte les deux positions factuellement
- Cite tes sources : "Selon l'article de [titre] sur [site]...", "Les utilisateurs Reddit mentionnent que..."
- Chaque affirmation dans claims_check est étayée par une donnée concrète du dossier
- L'analyse couvre le SUJET GLOBAL et son écosystème, pas seulement le contenu source
- Réponds en français (tags en anglais). Traduis intégralement si la source n'est ni FR ni EN.
</rules>

<output_format>
JSON valide uniquement. Aucun markdown, aucun backtick, aucun texte avant ou après.

{
  "tags": ["tag1", "tag2"],
  "summary": "Résumé factuel en 2-3 phrases couvrant le contenu ET ce que disent les autres sources.",
  "translation": "Traduction FR complète si contenu ni FR ni EN. Sinon null.",
  "claims_check": "Vérification FACTUELLE de chaque claim avec citation des sources du dossier. Format : 'Claim: X → Vérifié/Infirmé : [preuve concrète du dossier]'",
  "red_flags": "Red flags CONSTATÉS avec preuves tirées du dossier. 'Aucun' si RAS.",
  "actionable": "Actions CONCRÈTES pour les projets de Bacary. Pas 'explorer' mais 'Intégrer X dans Y parce que Z'. Si rien de pertinent, le dire.",
  "feature_ideas": "2-5 idées CONCRÈTES de features pour les projets existants de Bacary (AI Agent Company, Ledger, WhoGhost, Groove Candy) OU idées de nouveaux projets/produits inspirés par ce contenu. Format : '• [Projet] Feature/Idée : description en 1-2 phrases avec le WHY'. null si aucune idée pertinente.",
  "repo_analysis": "Analyse technique complète si repo GitHub dans le dossier (stats réelles, stack, qualité, activité). Sinon null.",
  "web_research_summary": "SYNTHÈSE DES SOURCES EXTERNES : Que disent les articles trouvés ? Quel consensus sur Reddit/HN ? Y a-t-il des critiques récurrentes ? Des alternatives mentionnées ?",
  "conclusion": "ANALYSE GLOBALE du sujet : état de l'art, positionnement vs alternatives, nouveauté réelle, opportunité de monétisation. Appuyée sur les sources du dossier.",
  "eli5": "1-2 phrases pour un non-technique. Zéro jargon.",
  "verdict": "SIGNAL | NOISE | MIXED — voir <verdict_criteria>"
}
</output_format>

<verdict_criteria>
SIGNAL (haute valeur, actionnable immédiatement) — TOUTES ces conditions réunies :
- Claims principaux VÉRIFIÉS avec preuves concrètes MULTIPLES et indépendantes
- Red flags ABSENTS (zéro) ou purement cosmétiques (typo, formatage)
- Opportunité d'action concrète et réaliste pour les projets de Bacary
- Consensus CLAIREMENT positif dans les sources externes (Reddit, HN, articles)
- Personnes réputées/experts confirmant la valeur du contenu
- Si repo : activement maintenu, communauté réelle, pas de signaux d'abandon

MIXED (intéressant mais avec réserves significatives) — UN seul critère suffit :
- UN claim partiellement vérifié ou source unique sans corroboration
- UN SEUL red flag non-cosmétique (exagération, claim douteux, biais)
- Avis partagés dans les sources (pas de consensus clair)
- Opportunité existante mais conditionnelle ou risquée
- Si repo : jeune, peu de communauté, ou dépendances risquées

NOISE (faible valeur, à ignorer) — UN seul critère suffit :
- Claims principaux INFIRMÉS ou invérifiables
- DEUX red flags ou plus de nature structurelle (même non-majeurs individuellement)
- Sources externes majoritairement négatives ou sceptiques
- Aucune opportunité actionnable concrète
- Contenu promotionnel, thread viral sans substance, opinion sans données, buzzwords
- Chiffres de revenus/performance non vérifiés par source indépendante
- Langage marketing exagéré ("killed", "game-changer", "10x", "revolutionary")
- Si repo : abandonné, pas de vrai code, vanity metrics
- Auteur avec historique de contenu promotionnel/hype

RÈGLE STRICTE NOISE : Dès que red_flags contient 2+ flags (même mineurs), le verdict est NOISE sauf si les claims sont solidement vérifiés par des sources indépendantes multiples.
RÈGLE STRICTE MIXED : Si red_flags contient UN flag structurel, le verdict est MIXED maximum, JAMAIS SIGNAL.
RÈGLE STRICTE PROMO : Un contenu promotionnel (langage superlatif, claims de revenus, comparaisons exagérées) sans preuves indépendantes = NOISE systématiquement.
RÈGLE STRICTE HYPE : Si le dossier de recherche ciblé (section critiques/sentiment) contient des avis majoritairement sceptiques ou des contre-arguments solides = NOISE.
RÈGLE STRICTE EXPERTS : L'avis de personnes réputées (développeurs connus, analystes reconnus, fondateurs établis) pèse PLUS qu'un thread viral anonyme. Si les experts sont sceptiques, NOISE même si le thread a des likes.
</verdict_criteria>

<examples>
<example>
<input>Tweet: "OpenRAG : plateforme RAG complète qui package Langflow, Docling et OpenSearch. github.com/langflow-ai/openrag"</input>
<research_dossier>
REPO GITHUB: langflow-ai/openrag — Stars: 1200 | Forks: 89 | Python | MIT | Last push: 2026-03-14
README: Complete RAG platform combining Langflow for workflow building, Docling for document ingestion, OpenSearch for retrieval...
ARTICLES: [Medium] "OpenRAG vs LangChain: A comparison" — L'auteur préfère OpenRAG pour le visual builder mais note le manque de docs...
REDDIT: [r/MachineLearning] 45 upvotes — "Anyone tried OpenRAG? The Langflow integration is nice but OpenSearch setup is painful"
HACKERNEWS: "Show HN: OpenRAG" — 23 comments, mostly positive about the concept, concerns about vendor lock-in with Langflow
</research_dossier>
<output>
{
  "tags": ["AI", "RAG", "Open Source", "Developer Tools"],
  "summary": "OpenRAG est une plateforme RAG open-source (MIT) combinant Langflow, Docling et OpenSearch. Le repo est actif (1200 stars, dernier push il y a 3 jours). Les retours communautaires sont globalement positifs sur le concept mais pointent des difficultés de setup OpenSearch et un manque de documentation.",
  "translation": null,
  "claims_check": "Claim: 'plateforme RAG complète' → Confirmé : le repo intègre effectivement Langflow (workflow builder), Docling (ingestion documents IBM), et OpenSearch (moteur de recherche). Claim: 'package ensemble' → Partiellement confirmé : le README montre l'intégration mais selon Reddit r/MachineLearning, le setup OpenSearch reste 'painful' et nécessite une configuration manuelle significative.",
  "red_flags": "Aucun red flag majeur. Points d'attention : le terme 'plateforme complète' est optimiste selon les retours utilisateurs — le setup OpenSearch demande du travail. Documentation jugée insuffisante par l'article Medium comparatif.",
  "actionable": "Pertinent pour AI Agent Company : le visual builder Langflow pourrait accélérer le prototypage de pipelines RAG pour les agents. Tester l'intégration Docling pour l'ingestion de documents clients. Ne pas déployer en production avant résolution des problèmes de setup documentés sur Reddit.",
  "feature_ideas": "• [AI Agent Company] RAG-as-a-Service : proposer un service managé de pipeline RAG basé sur l'architecture OpenRAG (Langflow + Docling + OpenSearch) pour les clients enterprise qui veulent interroger leurs docs internes.\n• [AI Agent Company] Agent Builder Visual : intégrer Langflow comme UI de construction d'agents pour permettre aux non-devs de créer des workflows d'agents.\n• [Nouveau projet] DocuSearch : SaaS vertical de recherche documentaire pour cabinets d'avocats/comptables, basé sur Docling + OpenSearch, facturé au volume de documents indexés.",
  "repo_analysis": "Stack : Python, MIT License. 1200 stars, 89 forks (ratio 13:1, sain). Dernier push 2026-03-14, projet actif. Dépend de Langflow (DataStax), Docling (IBM), OpenSearch. Architecture modulaire selon le README. Manque : tests CI visibles, documentation API détaillée.",
  "web_research_summary": "L'article Medium 'OpenRAG vs LangChain' positionne OpenRAG comme plus accessible grâce au visual builder mais moins flexible que LangChain pour les cas avancés. Sur Reddit (45 upvotes), le consensus est positif sur le concept mais critique sur le setup OpenSearch. Sur HackerNews (23 commentaires), les développeurs apprécient l'approche mais s'inquiètent du vendor lock-in avec Langflow (propriété DataStax). Alternatives mentionnées : Haystack, RAGFlow, Dify.",
  "conclusion": "Le marché RAG est en pleine consolidation (LangChain, LlamaIndex, Haystack, Dify). OpenRAG se différencie par l'intégration visuelle Langflow mais cette dépendance à DataStax est un risque. Le projet est jeune mais actif. Pas d'opportunité de monétisation directe, mais les patterns d'architecture (Docling + OpenSearch) sont réutilisables pour AI Agent Company.",
  "eli5": "C'est un outil tout-en-un pour créer des systèmes qui cherchent et utilisent des documents pour répondre à des questions. Prometteur mais encore jeune.",
  "verdict": "MIXED"
}
</output>
</example>

<example>
<input>Tweet: "Founders: stop paying $5K/month for marketing teams. This AI agent does it all for $50/month. 30x cheaper. We went from $0 to $12K MRR in 2 weeks."</input>
<research_dossier>
ARTICLES: [Starter Story] "How Jenny Dabot built an AI marketing agent" — success story, no independent verification, affiliate links present
REDDIT: [r/Entrepreneur] 12 upvotes — "sounds too good to be true", "where's the proof?", one user: "From all things that didn't happen, this didn't happen the most"
HACKERNEWS: User smt88: "These AI slop factories are ruining the internet" — 45 points
</research_dossier>
<output>
{
  "tags": ["AI Marketing Automation", "Startup Tools", "Autonomous Agents"],
  "summary": "Thread promotionnel sur un agent marketing IA à $50/mois remplaçant des équipes à $5K. Les claims de revenus ($12K MRR en 2 semaines) ne sont étayés par aucune source indépendante. Reddit et HackerNews sont majoritairement sceptiques.",
  "translation": null,
  "claims_check": "Claim: '$0 to $12K MRR in 2 weeks' → Non vérifiable : aucune source indépendante ne confirme ce chiffre. L'article Starter Story est un format publirédactionnel avec liens affiliés. Claim: '30x cheaper' → Comparaison trompeuse : compare un salaire d'équipe full-time à un outil SaaS limité, pas le même périmètre de service.",
  "red_flags": "Effet Dunning-Kruger potentiel : success stories non vérifiables. Scepticisme fort sur Reddit ('From all things that didn't happen, this didn't happen the most'). HackerNews critique ces 'AI slop factories'. Format publirédactionnel avec liens affiliés. Chiffres de revenus invérifiables.",
  "actionable": "Aucune action recommandée. Le concept d'agent marketing IA est pertinent pour AI Agent Company mais ce thread spécifique est du contenu promotionnel, pas une référence technique fiable.",
  "feature_ideas": null,
  "repo_analysis": null,
  "web_research_summary": "L'article Starter Story est un publirédactionnel, pas un article journalistique. Sur Reddit r/Entrepreneur, le scepticisme domine (12 upvotes seulement, commentaires négatifs). Sur HackerNews, l'utilisateur smt88 (45 points) critique fondamentalement ces outils comme des 'AI slop factories'.",
  "conclusion": "Le marché de l'automatisation marketing IA est réel mais ce contenu est du marketing, pas de l'information. Les claims de revenus sont invérifiables, les sources sont biaisées (affiliés), et la communauté technique est sceptique. Le sujet mérite veille mais pas via cette source.",
  "eli5": "Quelqu'un dit que son robot marketing rapporte gros, mais personne d'indépendant ne le confirme et beaucoup de gens doutent en ligne.",
  "verdict": "NOISE"
}
</output>
</example>
</examples>

<self_verification>
Avant de répondre, vérifie :
1. As-tu cité des sources concrètes du dossier dans claims_check ?
2. As-tu synthétisé les avis Reddit/HN/articles dans web_research_summary ?
3. As-tu INTÉGRÉ les résultats de la recherche ciblée (critiques, sentiment, crédibilité) dans ton analyse ?
4. As-tu analysé le SUJET GLOBAL (pas juste le tweet) dans conclusion ?
5. N'as-tu écrit AUCUNE formule de type "à vérifier" ou "Bacary devrait" ?
6. As-tu utilisé les données GitHub réelles (stars, dates, etc.) si disponibles ?
7. VERDICT CHECK STRICT : Compte le nombre de red flags listés. Si 2 ou plus → ton verdict DOIT être NOISE sauf preuves indépendantes massives.
8. VERDICT CHECK STRICT : Si red_flags contient même 1 flag structurel, JAMAIS SIGNAL. Rétrograde à MIXED minimum.
9. VERDICT CHECK : Si claims_check contient des claims infirmés ou non vérifiables → NOISE, pas MIXED.
10. VERDICT CHECK : Si les sources externes sont sceptiques OU si la recherche ciblée révèle des critiques → NOISE.
11. VERDICT CHECK : Le contenu utilise-t-il du langage marketing exagéré ? Si oui et pas de preuves indépendantes → NOISE.
12. VERDICT CHECK : L'auteur est-il crédible selon le dossier ? Un promoteur récurrent sans expertise technique = NOISE.
</self_verification>"""


# ═══════════════════════════════════════
# CACHED SYSTEM PROMPT (for Anthropic prompt caching)
# ═══════════════════════════════════════

ANALYSIS_SYSTEM_CACHED = [{
    "type": "text",
    "text": ANALYSIS_SYSTEM_PROMPT,
    "cache_control": {"type": "ephemeral"}
}]


# ═══════════════════════════════════════
# URL RESOLUTION & EXTRACTION (sync helpers — no I/O)
# ═══════════════════════════════════════

def _extract_urls(text: str) -> list[str]:
    return re.findall(r'https?://[^\s\)\]>"]+', text)


def _extract_github_repos(urls: list[str]) -> list[tuple[str, str]]:
    repos = []
    for url in urls:
        m = re.match(r'https?://github\.com/([^/]+)/([^/\s?#]+)', url)
        if m:
            owner, repo = m.group(1), m.group(2).rstrip('/')
            if owner not in ("features", "settings", "marketplace", "explore", "topics"):
                repos.append((owner, repo))
    return repos


def _build_search_query(content: str, author: str) -> str:
    cleaned = re.sub(r'https?://\S+', '', content)
    cleaned = re.sub(r'@\w+', '', cleaned)
    cleaned = re.sub(r'[#]', '', cleaned)
    words = cleaned.split()[:20]
    query = " ".join(words).strip()
    if len(query) < 10:
        query = f"{author} {query}"
    return query[:150]


def _format_exa_results(results: list[dict], source_label: str) -> str:
    if not results:
        return ""
    parts = []
    for r in results:
        title = r.get("title", "Sans titre")
        url = r.get("url", "")
        summary = r.get("summary", "")
        text = r.get("text", "")[:500] if not summary else ""
        highlights = r.get("highlights", [])
        hl_text = " | ".join(highlights[:2]) if highlights else ""

        entry = f"[{source_label}] {title}\nURL: {url}"
        if summary:
            entry += f"\nRésumé: {summary}"
        if hl_text:
            entry += f"\nExtraits clés: {hl_text}"
        if text and not summary:
            entry += f"\nContenu: {text}"
        parts.append(entry)
    return "\n\n".join(parts)


def _parse_claude_json(raw: str) -> dict:
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return json.loads(raw.strip())


# ═══════════════════════════════════════
# ASYNC I/O — all network calls run in parallel
# ═══════════════════════════════════════

async def _resolve_url_async(url: str, client: httpx.AsyncClient) -> str:
    if not any(short in url for short in ("t.co/", "bit.ly/", "tinyurl.com/", "ow.ly/")):
        return url
    try:
        r = await client.head(url, follow_redirects=True, timeout=5)
        final = str(r.url)
        return final if final != url else url
    except Exception:
        return url


async def _resolve_shortened_urls_async(urls: list[str],
                                         client: httpx.AsyncClient) -> list[str]:
    tasks = [_resolve_url_async(u, client) for u in urls]
    return list(await asyncio.gather(*tasks))


async def _fetch_github_context_async(owner: str, repo: str,
                                       client: httpx.AsyncClient) -> str:
    gh = {"Accept": "application/vnd.github.v3+json"}
    parts = []
    try:
        meta_resp, readme_resp = await asyncio.gather(
            client.get(f"https://api.github.com/repos/{owner}/{repo}",
                       headers=gh, timeout=8),
            client.get(f"https://api.github.com/repos/{owner}/{repo}/readme",
                       headers=gh, timeout=8),
        )
        if meta_resp.status_code == 200:
            d = meta_resp.json()
            parts.append(f"Repo: {d.get('full_name')} — {d.get('description', 'N/A')}")
            parts.append(f"Stars: {d.get('stargazers_count', 0)} | Forks: {d.get('forks_count', 0)} | Open issues: {d.get('open_issues_count', 0)}")
            parts.append(f"Language: {d.get('language', 'N/A')} | License: {(d.get('license') or {}).get('spdx_id', 'N/A')}")
            parts.append(f"Created: {d.get('created_at', '?')[:10]} | Last push: {d.get('pushed_at', '?')[:10]}")
            parts.append(f"Archived: {d.get('archived', False)} | Topics: {', '.join(d.get('topics', []))}")
            parts.append(f"Default branch: {d.get('default_branch', 'main')} | Size: {d.get('size', 0)} KB")
            parts.append(f"Watchers: {d.get('subscribers_count', 0)} | Network: {d.get('network_count', 0)}")
        if readme_resp.status_code == 200:
            content = base64.b64decode(
                readme_resp.json().get("content", "")
            ).decode("utf-8", errors="replace")
            parts.append(f"\n--- README (extrait) ---\n{content[:4000]}")
    except Exception:
        pass
    return "\n".join(parts) if parts else ""


async def _exa_search_async(query: str, exa_key: str, client: httpx.AsyncClient,
                             num_results: int = 5, include_domains: list = None,
                             category: str = None, light: bool = False) -> list[dict]:
    """light=True returns only summaries (faster for secondary sources)."""
    contents = {"summary": {"query": query}}
    if not light:
        contents["text"] = {"maxCharacters": 1500}
        contents["highlights"] = {"numSentences": 3, "query": query}

    body = {"query": query, "numResults": num_results,
            "type": "neural", "contents": contents}
    if include_domains:
        body["includeDomains"] = include_domains
    if category:
        body["category"] = category
    try:
        r = await client.post(f"{EXA_API_URL}/search",
                              headers={"x-api-key": exa_key,
                                       "Content-Type": "application/json"},
                              json=body, timeout=10)
        if r.status_code == 200:
            return r.json().get("results", [])
    except Exception:
        pass
    return []


async def _exa_crawl_async(url: str, exa_key: str,
                            client: httpx.AsyncClient,
                            max_chars: int = 2500) -> str:
    try:
        r = await client.post(f"{EXA_API_URL}/contents",
                              headers={"x-api-key": exa_key,
                                       "Content-Type": "application/json"},
                              json={"urls": [url],
                                    "text": {"maxCharacters": max_chars}},
                              timeout=10)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                return results[0].get("text", "")
    except Exception:
        pass
    return ""


# ═══════════════════════════════════════
# ASYNC RESEARCH PIPELINE — all sources fetched in parallel
# ═══════════════════════════════════════

def _extract_project_names(content: str, urls: list[str],
                           github_repos: list[tuple[str, str]]) -> list[str]:
    """Extract project/product names for targeted searches."""
    names = set()
    for owner, repo in github_repos:
        names.add(repo)
        names.add(f"{owner}/{repo}")
    for url in urls:
        m = re.search(r'(?:github\.com|gitlab\.com)/[^/]+/([^/\s?#]+)', url)
        if m:
            names.add(m.group(1).rstrip('/'))
    capitalized = re.findall(r'\b([A-Z][a-zA-Z0-9]{2,}(?:[A-Z][a-z]+)*)\b', content)
    stopwords = {"The", "This", "That", "And", "But", "For", "With", "From",
                 "How", "What", "Why", "Just", "Here", "Now", "New", "Our",
                 "Your", "Get", "Use", "All", "Any", "One", "Two"}
    for w in capitalized:
        if w not in stopwords and len(w) > 2:
            names.add(w)
    return list(names)[:5]


async def _research_topic_async(content: str, author: str,
                                 urls: list[str], exa_key: str,
                                 client: httpx.AsyncClient,
                                 github_repos: list[tuple[str, str]] = None) -> str:
    query = _build_search_query(content, author)
    project_names = _extract_project_names(
        content, urls, github_repos or [])

    # ── Core searches (generic topic) ──
    articles_fut = _exa_search_async(query, exa_key, client, num_results=5)
    reddit_fut = _exa_search_async(query, exa_key, client, num_results=3,
                                    include_domains=["reddit.com"], light=True)
    hn_fut = _exa_search_async(query, exa_key, client, num_results=3,
                                include_domains=["news.ycombinator.com"], light=True)
    tweets_fut = _exa_search_async(query, exa_key, client, num_results=3,
                                    category="tweet", light=True)

    # ── Targeted searches (repo/project-specific) ──
    targeted_futs = []
    targeted_labels = []

    for name in project_names[:3]:
        targeted_futs.append(_exa_search_async(
            f'"{name}" review criticism problems limitations issues',
            exa_key, client, num_results=3, light=True))
        targeted_labels.append(f"Critique: {name}")

        targeted_futs.append(_exa_search_async(
            f'"{name}" opinion',
            exa_key, client, num_results=3,
            include_domains=["reddit.com", "news.ycombinator.com"], light=True))
        targeted_labels.append(f"Sentiment: {name}")

    # ── Author credibility search ──
    if author and author not in ("unknown", "upload"):
        targeted_futs.append(_exa_search_async(
            f"@{author} credibility reputation reliable",
            exa_key, client, num_results=2, category="tweet", light=True))
        targeted_labels.append(f"Crédibilité: @{author}")

    # ── Counter-argument search ──
    counter_query = _build_search_query(content, "")
    targeted_futs.append(_exa_search_async(
        f"{counter_query} criticism skepticism debunk overrated hype",
        exa_key, client, num_results=3, light=True))
    targeted_labels.append("Contre-arguments")

    # ── URL crawls ──
    skip_domains = ("github.com", "x.com", "twitter.com", "t.co")
    crawl_urls = [u for u in urls[:3] if not any(d in u for d in skip_domains)]
    crawl_futs = [_exa_crawl_async(u, exa_key, client) for u in crawl_urls]

    # ── Run ALL in parallel ──
    n_core = 4
    n_targeted = len(targeted_futs)
    all_results = await asyncio.gather(
        articles_fut, reddit_fut, hn_fut, tweets_fut,
        *targeted_futs, *crawl_futs,
        return_exceptions=True,
    )

    def _safe(idx):
        r = all_results[idx]
        return r if not isinstance(r, Exception) else []

    articles = _safe(0)
    reddit = _safe(1)
    hn = _safe(2)
    tweets = _safe(3)
    targeted_results = [_safe(n_core + i) for i in range(n_targeted)]
    crawled = all_results[n_core + n_targeted:]

    # ── Assemble dossier ──
    sections = []
    fmt = _format_exa_results(articles, "Article")
    if fmt:
        sections.append(f"=== ARTICLES & BLOGS TROUVÉS (recherche sémantique Exa) ===\n{fmt}")
    fmt = _format_exa_results(reddit, "Reddit")
    if fmt:
        sections.append(f"=== DISCUSSIONS REDDIT ===\n{fmt}")
    fmt = _format_exa_results(hn, "HackerNews")
    if fmt:
        sections.append(f"=== DISCUSSIONS HACKERNEWS ===\n{fmt}")
    fmt = _format_exa_results(tweets, "Tweet")
    if fmt:
        sections.append(f"=== TWEETS SIMILAIRES ===\n{fmt}")

    # ── Targeted results section ──
    targeted_parts = []
    for i, label in enumerate(targeted_labels):
        fmt = _format_exa_results(targeted_results[i], label)
        if fmt:
            targeted_parts.append(fmt)
    if targeted_parts:
        sections.append(
            f"=== RECHERCHE CIBLÉE (critiques, sentiment, crédibilité) ===\n"
            + "\n\n".join(targeted_parts))

    for i, text in enumerate(crawled):
        if isinstance(text, str) and len(text) > 100:
            sections.append(f"=== PAGE CRAWLÉE: {crawl_urls[i]} ===\n{text[:2500]}")

    return "\n\n".join(sections)


# ═══════════════════════════════════════
# CORE ANALYSIS — async with parallel I/O + prompt caching
# ═══════════════════════════════════════

async def analyze_single_async(content: str, author: str, api_key: str,
                                url: str = "", exa_key: str = "") -> dict:
    async with httpx.AsyncClient() as client:
        raw_urls = list(set(_extract_urls(content) + ([url] if url else [])))

        resolve_task = _resolve_shortened_urls_async(raw_urls, client)
        gh_repos = _extract_github_repos(raw_urls)
        gh_tasks = [_fetch_github_context_async(o, r, client)
                    for o, r in gh_repos[:2]]

        resolved_urls, *gh_results = await asyncio.gather(
            resolve_task, *gh_tasks, return_exceptions=True,
        )

        if isinstance(resolved_urls, Exception):
            resolved_urls = raw_urls
        unique_urls = list(dict.fromkeys(
            resolved_urls if isinstance(resolved_urls, list) else raw_urls
        ))

        dossier_parts = []
        for i, (owner, repo) in enumerate(gh_repos[:2]):
            ctx = gh_results[i] if not isinstance(gh_results[i], Exception) else ""
            if ctx:
                dossier_parts.append(
                    f"=== REPO GITHUB VÉRIFIÉ: {owner}/{repo} ===\n{ctx}")

        if exa_key:
            exa_research = await _research_topic_async(
                content, author, unique_urls, exa_key, client,
                github_repos=gh_repos)
            if exa_research:
                dossier_parts.append(exa_research)

    dossier = "\n\n".join(dossier_parts)

    user_msg = f"""<content>
Contenu de @{author} :
{content}

URL source: {url}
</content>"""

    if dossier:
        user_msg += f"""

<research_dossier>
Dossier de recherche compilé automatiquement. TOUTES ces données sont réelles et vérifiées.

{dossier}
</research_dossier>

Tu as ci-dessus un dossier complet. Base CHAQUE vérification sur ces données. Cite les sources."""
    else:
        user_msg += "\n\nAucune recherche externe n'a abouti. Analyse sur la base du contenu seul et indique clairement les limites."

    aclient = AsyncAnthropic(api_key=api_key)
    response = await aclient.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=ANALYSIS_SYSTEM_CACHED,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    try:
        return _parse_claude_json(raw)
    except json.JSONDecodeError:
        return {
            "tags": ["Uncategorized"], "summary": raw[:500],
            "claims_check": "Erreur de parsing", "red_flags": "N/A",
            "actionable": "N/A", "verdict": "MIXED", "_raw": raw,
        }


def analyze_single(content: str, author: str, api_key: str,
                   url: str = "", exa_key: str = "") -> dict:
    """Sync wrapper — used when no event loop is running."""
    return asyncio.run(
        analyze_single_async(content, author, api_key, url, exa_key))


# ═══════════════════════════════════════
# BATCH ANALYSIS — concurrent with semaphore
# ═══════════════════════════════════════

CATEGORIZE_BATCH_PROMPT = """Tu es un système de catégorisation. Pour chaque tweet, assigne 1-4 tags pertinents.

Réponds UNIQUEMENT en JSON valide : un tableau d'objets avec "tweet_id" et "tags".

Exemple :
[
  {"tweet_id": "123", "tags": ["AI", "LLM", "Open Source"]},
  {"tweet_id": "456", "tags": ["Crypto", "DeFi"]}
]

Tags possibles (non exhaustif, tu peux en créer) : AI, LLM, Crypto, Web3, DeFi, Product Management, Startup, Open Source, Hardware, Design, Music Tech, Marketing, Developer Tools, Security, Data, Robotics, Gaming, AR/VR, Fintech, E-commerce, Social Media, Regulation, Research, Infrastructure.
"""


async def categorize_batch_async(tweets: list, api_key: str) -> list:
    aclient = AsyncAnthropic(api_key=api_key)
    tweets_text = "\n\n".join([
        f"[ID: {t['tweet_id']}] @{t['author_username']}: {t['content'][:300]}"
        for t in tweets
    ])
    response = await aclient.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=CATEGORIZE_BATCH_PROMPT,
        messages=[{"role": "user", "content": tweets_text}],
    )
    raw = response.content[0].text.strip()
    try:
        return _parse_claude_json(raw)
    except json.JSONDecodeError:
        return []


def categorize_batch(tweets: list, api_key: str) -> list:
    return asyncio.run(categorize_batch_async(tweets, api_key))


async def analyze_batch_async(tweets: list, api_key: str,
                               exa_key: str = "",
                               concurrency: int = 3) -> list:
    sem = asyncio.Semaphore(concurrency)

    async def _one(tweet):
        async with sem:
            result = await analyze_single_async(
                content=tweet["content"],
                author=tweet["author_username"],
                api_key=api_key,
                url=tweet.get("url", ""),
                exa_key=exa_key,
            )
            result["tweet_id"] = tweet["tweet_id"]
            return result

    return list(await asyncio.gather(*[_one(t) for t in tweets]))


def analyze_batch(tweets: list, api_key: str,
                  progress_callback=None, exa_key: str = "") -> list:
    return asyncio.run(analyze_batch_async(tweets, api_key, exa_key))


# ═══════════════════════════════════════
# WEB CONTENT FETCHER (sync — simple scraper, rarely called)
# ═══════════════════════════════════════

def fetch_web_content(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (content-analyzer bot)"}
    try:
        if "reddit.com" in url and not url.endswith(".json"):
            json_url = url.rstrip("/") + ".json"
            r = requests.get(json_url,
                             headers={**headers, "Accept": "application/json"},
                             timeout=15)
            if r.status_code == 200:
                data = r.json()
                post = data[0]["data"]["children"][0]["data"]
                title = post.get("title", "")
                body = post.get("selftext", "")
                author = post.get("author", "")
                score = post.get("score", 0)
                comments_raw = []
                if len(data) > 1:
                    for c in data[1]["data"]["children"][:5]:
                        cd = c.get("data", {})
                        if cd.get("body"):
                            comments_raw.append(
                                f"u/{cd.get('author','?')}: {cd['body'][:300]}")
                content = (f"{title}\n\n{body}\n\n--- Top commentaires ---\n"
                           + "\n".join(comments_raw))
                return {"content": content[:6000],
                        "author_username": f"u/{author}",
                        "author_name": author, "title": title,
                        "metrics": {"score": score}, "source_type": "reddit"}

        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        html = r.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        title_m = re.search(r'<title>([^<]+)</title>', html)
        title = title_m.group(1).strip() if title_m else url
        return {"content": text[:6000], "author_username": url.split("/")[2],
                "author_name": title[:80], "title": title,
                "metrics": {}, "source_type": "web"}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════
# MARKDOWN REPORT (sync — pure string formatting)
# ═══════════════════════════════════════

def generate_markdown_report(tweet: dict, analysis: dict) -> str:
    verdict_emoji = {
        "SIGNAL": "🟢", "NOISE": "🔴", "MIXED": "🟡"
    }.get(analysis.get("verdict", "MIXED"), "⚪")

    tags_str = " · ".join([f"`{t}`" for t in analysis.get("tags", [])])
    translation = analysis.get('translation')
    translation_section = f"\n## Traduction FR\n\n{translation}\n" if translation else ""
    repo_analysis = analysis.get('repo_analysis')
    repo_section = f"\n## Analyse du repo\n\n{repo_analysis}\n" if repo_analysis else ""
    web_research = analysis.get('web_research_summary')
    web_section = f"\n## Recherche web (Exa)\n\n{web_research}\n" if web_research else ""

    md = f"""# {verdict_emoji} Analyse — @{tweet.get('author_username', 'unknown')}

**Source:** [{tweet.get('url', '')}]({tweet.get('url', '')})
**Date:** {tweet.get('created_at', 'N/A')}
**Verdict:** {analysis.get('verdict', 'N/A')} {verdict_emoji}
**Tags:** {tags_str}

---

## Contenu original

> {tweet.get('content', '').replace(chr(10), chr(10) + '> ')}
{translation_section}
---

## Résumé

{analysis.get('summary', 'N/A')}
{repo_section}{web_section}
## Vérification des claims

{analysis.get('claims_check', 'N/A')}

## Red flags

{analysis.get('red_flags', 'Aucun')}

## Actions concrètes

{analysis.get('actionable', 'N/A')}

## Conclusion

{analysis.get('conclusion', 'N/A')}

---

*Analysé par Content Analyzer · Claude Sonnet · Recherche Exa*
"""
    return md


# ═══════════════════════════════════════
# BRIEF GENERATOR — async
# ═══════════════════════════════════════

BRIEF_SYSTEM_PROMPT = """<role>
Tu es un architecte logiciel senior et expert en prompt engineering.
Tu génères des briefs techniques et des meta-prompts pour lancer des projets dans Cursor IDE ou Claude Code.
</role>

<context>
L'utilisateur Bacary décrit ce qu'il veut construire.
Tu reçois aussi une liste de ressources pertinentes extraites de ses bookmarks analysés (articles, repos, outils, discussions).
Ces ressources ont été évaluées : SIGNAL = haute valeur, MIXED = intéressant, NOISE = faible valeur.
</context>

<instructions>
1. Analyse la demande de l'utilisateur pour identifier : objectif, stack technique implicite, contraintes
2. Examine les ressources pertinentes trouvées dans ses bookmarks
3. Génère un brief structuré ET un meta-prompt prêt à coller dans Cursor/Claude Code
</instructions>

<output_format>
Réponds en JSON avec ces champs :

{
  "project_name": "Nom court du projet/feature",
  "objective": "Objectif en 1-2 phrases claires",
  "relevant_resources": [
    {"title": "...", "url": "...", "why": "Pourquoi cette resource est pertinente", "verdict": "SIGNAL/MIXED"}
  ],
  "tech_stack_suggestion": "Stack recommandée basée sur la demande + les ressources trouvées",
  "architecture_notes": "Notes d'architecture clés (2-3 phrases)",
  "cursor_prompt": "Le META-PROMPT complet, prêt à coller dans Cursor IDE. Doit être autonome, détaillé, et inclure les liens/repos pertinents comme contexte. Format Markdown.",
  "claude_code_prompt": "Variante du prompt optimisée pour Claude Code (plus concis, orienté terminal).",
  "risks": "Risques identifiés à partir des red flags des analyses",
  "next_steps": ["Étape 1", "Étape 2", "Étape 3"]
}
</output_format>

<rules>
- Le cursor_prompt doit être AUTONOME : quelqu'un qui le colle dans Cursor doit pouvoir commencer immédiatement
- Inclus les URLs des repos/articles pertinents directement dans le prompt
- Sois concret : pas de "il faudrait explorer", mais "Utilise X parce que Y"
- Le prompt doit mentionner les contraintes techniques identifiées dans les analyses (red flags, limites)
- Réponds en français sauf le code et les prompts techniques (anglais OK)
- JSON valide uniquement, sans markdown, sans backticks, sans préambule
</rules>"""

BRIEF_SYSTEM_CACHED = [{
    "type": "text",
    "text": BRIEF_SYSTEM_PROMPT,
    "cache_control": {"type": "ephemeral"}
}]


async def generate_brief_async(user_request: str, relevant_bookmarks: list,
                                api_key: str, exa_key: str = "") -> dict:
    resources_context = ""
    for bm in relevant_bookmarks[:10]:
        tags = bm.get("tags", "")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except Exception:
                tags = []
        tag_str = ", ".join(tags[:3]) if isinstance(tags, list) else ""
        verdict = (bm.get("verdict") or "").upper()
        resources_context += f"""
--- Resource [{verdict}] ---
Source: {bm.get('url', 'N/A')}
Auteur: {bm.get('author_username', '')}
Tags: {tag_str}
Résumé: {bm.get('summary', '')[:300]}
Actions: {(bm.get('actionable') or '')[:300]}
Conclusion: {(bm.get('conclusion') or '')[:200]}
Repo: {(bm.get('repo_analysis') or '')[:200]}
Recherche web: {(bm.get('web_research_summary') or '')[:200]}
Red flags: {(bm.get('red_flags') or '')[:150]}
"""

    extra_research = ""
    if exa_key:
        async with httpx.AsyncClient() as client:
            results = await _exa_search_async(
                user_request, exa_key, client, num_results=3)
        if results:
            extra_research = "\n--- Recherche Exa complémentaire ---\n"
            for r in results:
                extra_research += (
                    f"[{r.get('title','')}] {r.get('url','')}: "
                    f"{r.get('summary','')[:200]}\n")

    user_msg = f"""<user_request>
{user_request}
</user_request>

<bookmarks_resources>
{len(relevant_bookmarks)} ressources pertinentes trouvées dans les bookmarks analysés :
{resources_context}
</bookmarks_resources>
{extra_research}
Génère le brief et les prompts."""

    aclient = AsyncAnthropic(api_key=api_key)
    response = await aclient.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=BRIEF_SYSTEM_CACHED,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    try:
        return _parse_claude_json(raw)
    except json.JSONDecodeError:
        return {"error": raw[:500]}


def generate_brief(user_request: str, relevant_bookmarks: list,
                   api_key: str, exa_key: str = "") -> dict:
    return asyncio.run(
        generate_brief_async(user_request, relevant_bookmarks, api_key, exa_key))


# ═══════════════════════════════════════
# CHAT ABOUT CONTENT — async
# ═══════════════════════════════════════

CHAT_SYSTEM_PROMPT = """<role>
Tu es un assistant de recherche expert en tech, AI, crypto et business.
Tu as accès au contenu complet d'un bookmark analysé et à son analyse détaillée.
Tu réponds en français, de manière concise et actionnable.
</role>

<rules>
- Réponds en français
- Sois concis mais complet (pas de blabla)
- Si l'utilisateur demande une recherche complémentaire, utilise les résultats fournis
- Cite tes sources quand tu te bases sur le dossier de recherche
- Formate avec **gras** pour les points clés et `code` pour les termes techniques
- Pas de markdown headers (#), utilise **gras** pour structurer
</rules>

<context>
L'utilisateur Bacary gère ces projets :
- AI Agent Company (agents IA, automatisation, MCP)
- Ledger (crypto, hardware wallet)
- WhoGhost (Instagram unfollower tracking)
- Groove Candy (music tech, distribution)
</context>"""


async def chat_about_content_async(message: str, bookmark: dict,
                                    analysis: dict, history: list,
                                    api_key: str, exa_key: str = "") -> str:
    context_parts = []
    content = bookmark.get("content", "")
    author = bookmark.get("author_username", "")
    url = bookmark.get("url", "")

    context_parts.append(f"Contenu de @{author} :\n{content}")
    if url:
        context_parts.append(f"URL: {url}")

    if analysis:
        for field in ("summary", "translation", "eli5", "conclusion",
                      "claims_check", "red_flags", "actionable",
                      "repo_analysis", "web_research_summary"):
            val = analysis.get(field)
            if val and val not in ("N/A", "Aucun", "null"):
                context_parts.append(f"{field}: {val}")
        verdict = analysis.get("verdict", "")
        if verdict:
            context_parts.append(f"Verdict: {verdict}")
        tags = analysis.get("tags", "")
        if tags:
            context_parts.append(f"Tags: {tags}")

    exa_supplement = ""
    research_triggers = ("recherche", "cherche", "find", "search", "explore",
                         "alternative", "compétiteur", "concurrent", "approfondi")
    if exa_key and any(t in message.lower() for t in research_triggers):
        async with httpx.AsyncClient() as client:
            query = f"{message} {content[:200]}"
            results = await _exa_search_async(query, exa_key, client,
                                              num_results=5)
        if results:
            exa_supplement = _format_exa_results(
                results, "Recherche complémentaire")

    context = "\n\n".join(context_parts)

    messages = []
    messages.append({
        "role": "user",
        "content": (f"<bookmark_context>\n{context}\n</bookmark_context>\n\n"
                    "Réponds 'OK, j'ai le contexte.' pour confirmer.")
    })
    messages.append({"role": "assistant", "content": "OK, j'ai le contexte."})

    for h in history[:-1]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})

    user_msg = message
    if exa_supplement:
        user_msg += (f"\n\n<recherche_web>\n{exa_supplement}\n</recherche_web>\n"
                     "Utilise ces résultats de recherche pour enrichir ta réponse.")
    messages.append({"role": "user", "content": user_msg})

    aclient = AsyncAnthropic(api_key=api_key)
    response = await aclient.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        system=CHAT_SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text.strip()


def chat_about_content(message: str, bookmark: dict, analysis: dict,
                       history: list, api_key: str, exa_key: str = "") -> str:
    return asyncio.run(
        chat_about_content_async(message, bookmark, analysis, history,
                                 api_key, exa_key))

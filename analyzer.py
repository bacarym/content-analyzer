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

# Reusable HTTP clients — avoids TCP handshake overhead on each call
_httpx_client = None
_anthropic_clients = {}


def _get_httpx_client():
    global _httpx_client
    if _httpx_client is None or _httpx_client.is_closed:
        _httpx_client = httpx.AsyncClient(timeout=5)
    return _httpx_client


def _get_anthropic_client(api_key: str):
    if api_key not in _anthropic_clients:
        _anthropic_clients[api_key] = AsyncAnthropic(api_key=api_key)
    return _anthropic_clients[api_key]


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
- TU AS les données. Si un repo GitHub est dans le dossier, tu as ses stats, README, issues récentes, commits récents, ET la structure du code. UTILISE TOUT. Analyse les issues pour voir les vrais problèmes. Analyse les commits pour voir si le développement est actif. Analyse la structure pour voir si c'est du vrai code ou un projet vide.
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
  "repo_analysis": "Analyse technique APPROFONDIE si repo GitHub dans le dossier : stats réelles, stack, qualité du code (structure des fichiers), activité des commits (fréquence, auteurs), issues ouvertes (vrais bugs vs feature requests), maturité du projet. REGARDE les issues pour identifier les vrais problèmes rapportés par les utilisateurs. Sinon null.",
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
    import re
    raw = raw.strip()
    # Strip markdown code blocks
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    # Find JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]
    # Fix unescaped newlines inside JSON strings
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try fixing common issues: unescaped newlines in strings
        fixed = re.sub(r'(?<=": ")(.*?)(?="[,\n\r\s]*["}])',
                       lambda m: m.group(0).replace('\n', '\\n'),
                       raw, flags=re.DOTALL)
        return json.loads(fixed)


# ═══════════════════════════════════════
# ASYNC I/O — all network calls run in parallel
# ═══════════════════════════════════════

async def _resolve_url_async(url: str, client: httpx.AsyncClient) -> str:
    if not any(short in url for short in ("t.co/", "bit.ly/", "tinyurl.com/", "ow.ly/")):
        return url
    try:
        r = await client.head(url, follow_redirects=True, timeout=3)
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
    base = f"https://api.github.com/repos/{owner}/{repo}"
    parts = []
    try:
        # Fetch meta, readme, issues, commits, tree ALL in parallel
        meta_resp, readme_resp, issues_resp, commits_resp, tree_resp = await asyncio.gather(
            client.get(base, headers=gh, timeout=3),
            client.get(f"{base}/readme", headers=gh, timeout=3),
            client.get(f"{base}/issues?per_page=5&state=all&sort=updated", headers=gh, timeout=3),
            client.get(f"{base}/commits?per_page=3", headers=gh, timeout=3),
            client.get(f"{base}/contents", headers=gh, timeout=3),
            return_exceptions=True,
        )
        if not isinstance(meta_resp, Exception) and meta_resp.status_code == 200:
            d = meta_resp.json()
            parts.append(f"Repo: {d.get('full_name')} — {d.get('description', 'N/A')}")
            parts.append(f"Stars: {d.get('stargazers_count', 0)} | Forks: {d.get('forks_count', 0)} | Open issues: {d.get('open_issues_count', 0)}")
            parts.append(f"Language: {d.get('language', 'N/A')} | License: {(d.get('license') or {}).get('spdx_id', 'N/A')}")
            parts.append(f"Created: {d.get('created_at', '?')[:10]} | Last push: {d.get('pushed_at', '?')[:10]}")
            parts.append(f"Archived: {d.get('archived', False)} | Topics: {', '.join(d.get('topics', []))}")
            parts.append(f"Size: {d.get('size', 0)} KB | Watchers: {d.get('subscribers_count', 0)}")
        if not isinstance(readme_resp, Exception) and readme_resp.status_code == 200:
            content = base64.b64decode(
                readme_resp.json().get("content", "")
            ).decode("utf-8", errors="replace")
            parts.append(f"\n--- README (extrait) ---\n{content[:3000]}")
        # Recent issues — shows real community activity and problems
        if not isinstance(issues_resp, Exception) and issues_resp.status_code == 200:
            issues = issues_resp.json()[:5]
            if issues:
                issue_lines = []
                for iss in issues:
                    state = "🟢" if iss.get("state") == "open" else "🔴"
                    comments = iss.get("comments", 0)
                    issue_lines.append(f"  {state} #{iss.get('number')} [{comments} comments] {iss.get('title','')[:80]}")
                parts.append(f"\n--- ISSUES RÉCENTES ---\n" + "\n".join(issue_lines))
        # Recent commits — shows development activity
        if not isinstance(commits_resp, Exception) and commits_resp.status_code == 200:
            commits = commits_resp.json()[:5]
            if commits:
                commit_lines = []
                for c in commits:
                    date = c.get("commit", {}).get("author", {}).get("date", "")[:10]
                    msg = c.get("commit", {}).get("message", "").split("\n")[0][:80]
                    author = c.get("commit", {}).get("author", {}).get("name", "")
                    commit_lines.append(f"  {date} [{author}] {msg}")
                parts.append(f"\n--- COMMITS RÉCENTS ---\n" + "\n".join(commit_lines))
        # File tree — shows project structure and real code
        if not isinstance(tree_resp, Exception) and tree_resp.status_code == 200:
            files = tree_resp.json()
            if isinstance(files, list):
                file_names = [f.get("name", "") for f in files[:20]]
                parts.append(f"\n--- STRUCTURE PROJET ---\n{', '.join(file_names)}")
    except Exception:
        pass
    return "\n".join(parts) if parts else ""


async def _exa_search_async(query: str, exa_key: str, client: httpx.AsyncClient,
                             num_results: int = 5, include_domains: list = None,
                             category: str = None, light: bool = False) -> list[dict]:
    """light=True returns only summaries (faster for secondary sources)."""
    contents = {"summary": {"query": query}}
    if not light:
        contents["text"] = {"maxCharacters": 1200}
        contents["highlights"] = {"numSentences": 2, "query": query}

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
                              json=body, timeout=3)
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
                              timeout=3)
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

    # ── 4 core searches only (all parallel, ~4s max) ──
    articles_fut = _exa_search_async(query, exa_key, client, num_results=3)
    reddit_fut = _exa_search_async(query, exa_key, client, num_results=3,
                                    include_domains=["reddit.com"], light=True)
    hn_fut = _exa_search_async(query, exa_key, client, num_results=3,
                                include_domains=["news.ycombinator.com"], light=True)
    counter_query = _build_search_query(content, "")
    counter_fut = _exa_search_async(
        f"{counter_query} criticism skepticism problems limitations",
        exa_key, client, num_results=3, light=True)

    articles, reddit, hn, counter = await asyncio.gather(
        articles_fut, reddit_fut, hn_fut, counter_fut,
        return_exceptions=True,
    )

    def _safe(r):
        return r if not isinstance(r, Exception) else []

    # ── Assemble dossier ──
    sections = []
    fmt = _format_exa_results(_safe(articles), "Article")
    if fmt:
        sections.append(f"=== ARTICLES & BLOGS ===\n{fmt}")
    fmt = _format_exa_results(_safe(reddit), "Reddit")
    if fmt:
        sections.append(f"=== REDDIT ===\n{fmt}")
    fmt = _format_exa_results(_safe(hn), "HackerNews")
    if fmt:
        sections.append(f"=== HACKERNEWS ===\n{fmt}")
    fmt = _format_exa_results(_safe(counter), "Critique")
    if fmt:
        sections.append(f"=== CRITIQUES & CONTRE-ARGUMENTS ===\n{fmt}")

    return "\n\n".join(sections)


# ═══════════════════════════════════════
# CORE ANALYSIS — async with parallel I/O + prompt caching
# ═══════════════════════════════════════

async def analyze_single_async(content: str, author: str, api_key: str,
                                url: str = "", exa_key: str = "") -> dict:
    import time
    t0 = time.time()
    client = _get_httpx_client()
    raw_urls = list(set(_extract_urls(content) + ([url] if url else [])))
    gh_repos = _extract_github_repos(raw_urls)
    # Identify external article URLs (not twitter/x.com/github)
    _social = ("x.com", "twitter.com", "t.co")
    article_urls = [
        u for u in raw_urls
        if not any(s in u for s in _social) and "github.com" not in u
    ]
    has_interesting_urls = any(
        u for u in raw_urls
        if "github.com" in u or not any(s in u for s in _social)
    )
    dossier = ""
    if exa_key and (gh_repos or has_interesting_urls):
        resolve_task = _resolve_shortened_urls_async(raw_urls, client)
        gh_task_list = [_fetch_github_context_async(o, r, client)
                        for o, r in gh_repos[:1]]
        exa_task = _research_topic_async(content, author, raw_urls, exa_key,
                                          client, github_repos=gh_repos)
        crawl_list = [_exa_crawl_async(u, exa_key, client, max_chars=5000)
                      for u in article_urls[:3]]

        exa_research, resolved_urls, *rest = await asyncio.gather(
            exa_task, resolve_task, *gh_task_list, *crawl_list,
            return_exceptions=True,
        )
        gh_results = rest[:len(gh_task_list)]
        crawl_results = rest[len(gh_task_list):]
        t1 = time.time()
        print(f"[PERF] I/O phase: {t1-t0:.1f}s")

        dossier_parts = []
        for i, aurl in enumerate(article_urls[:3]):
            if i < len(crawl_results):
                text = crawl_results[i] if not isinstance(crawl_results[i], Exception) else ""
                if text:
                    dossier_parts.append(
                        f"=== CONTENU ARTICLE: {aurl} ===\n{text}")

        for i, (owner, repo) in enumerate(gh_repos[:1]):
            ctx = gh_results[i] if not isinstance(gh_results[i], Exception) else ""
            if ctx:
                dossier_parts.append(
                    f"=== REPO GITHUB VÉRIFIÉ: {owner}/{repo} ===\n{ctx}")

        if isinstance(exa_research, str) and exa_research:
            dossier_parts.append(exa_research)

        dossier = "\n\n".join(dossier_parts)
    elif exa_key:
        # Light research — no interesting URLs but still search
        crawl_tasks = [_exa_crawl_async(u, exa_key, client, max_chars=5000)
                       for u in article_urls[:3]]
        exa_fut = _research_topic_async(
            content, author, [], exa_key, client, github_repos=[])
        results = await asyncio.gather(exa_fut, *crawl_tasks, return_exceptions=True)
        exa_research = results[0]
        crawl_results = results[1:]
        t1 = time.time()
        print(f"[PERF] Light I/O: {t1-t0:.1f}s")
        dossier_parts = []
        for i, aurl in enumerate(article_urls[:3]):
            text = crawl_results[i] if i < len(crawl_results) and not isinstance(crawl_results[i], Exception) else ""
            if text:
                dossier_parts.append(
                    f"=== CONTENU ARTICLE: {aurl} ===\n{text}")
        if isinstance(exa_research, str) and exa_research:
            dossier_parts.append(exa_research)
        dossier = "\n\n".join(dossier_parts)
    else:
        print(f"[PERF] No research (long_content={is_long_content}, exa={'yes' if exa_key else 'no'})")

    user_msg = f"""<content>
Contenu de @{author} :
{content}

URL source: {url}
</content>"""

    if dossier:
        user_msg += f"""

<research_dossier>
{dossier}
</research_dossier>

Base CHAQUE vérification sur ces données. Cite les sources."""
    else:
        user_msg += "\n\nAucune recherche externe n'a abouti. Analyse sur la base du contenu seul."

    t2 = time.time()
    tokens = 2500 if len(content) > 500 else 1500
    aclient = _get_anthropic_client(api_key)
    response = await aclient.messages.create(
        model="claude-sonnet-4-6-20250514",
        max_tokens=tokens,
        system=ANALYSIS_SYSTEM_CACHED,
        messages=[{"role": "user", "content": user_msg}],
    )
    t3 = time.time()
    print(f"[PERF] Claude call: {t3-t2:.1f}s | Total: {t3-t0:.1f}s")

    raw = response.content[0].text.strip()
    try:
        return _parse_claude_json(raw)
    except (json.JSONDecodeError, Exception) as e:
        print(f"[PARSE] Failed: {e} | Raw first 200: {repr(raw[:200])}")
        return {
            "tags": ["Uncategorized"], "summary": raw[:500],
            "claims_check": f"Parse error: {e}", "red_flags": "N/A",
            "actionable": "N/A", "verdict": "MIXED", "_raw": raw,
        }




# ═══════════════════════════════════════
# BATCH ANALYSIS — concurrent with semaphore
# ═══════════════════════════════════════



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


# ═══════════════════════════════════════
# BRIEF CHAT — discuss & evolve a brief with database search
# ═══════════════════════════════════════

BRIEF_CHAT_SYSTEM_PROMPT = """<role>
Tu es un architecte logiciel senior et co-pilote de projet.
Tu as accès au brief technique complet d'un projet, et tu peux :
1. Répondre aux questions sur le brief
2. Chercher dans la base de bookmarks analysés pour enrichir tes réponses
3. Modifier le brief en fonction de la conversation
</role>

<context>
L'utilisateur Bacary gère ces projets :
- AI Agent Company (agents IA, automatisation, MCP)
- Ledger (crypto, hardware wallet)
- WhoGhost (Instagram unfollower tracking)
- Groove Candy (music tech, distribution)
</context>

<rules>
- Réponds en français, de manière concise et actionnable
- Formate avec **gras** pour les points clés et `code` pour les termes techniques
- Pas de markdown headers (#), utilise **gras** pour structurer
- Si l'utilisateur demande de modifier le brief (changer la stack, ajouter une feature, modifier l'architecture, etc.), inclus un bloc <brief_update> avec les champs modifiés en JSON
- Si tu utilises des résultats de recherche de la base de bookmarks, cite les sources
- Sois proactif : si tu vois une opportunité d'amélioration basée sur les bookmarks, propose-la
</rules>

<brief_update_format>
Quand tu modifies le brief, ajoute à la FIN de ta réponse un bloc comme ceci :
<brief_update>
{"field": "value", "field2": "value2"}
</brief_update>

Champs modifiables : project_name, objective, tech_stack_suggestion, architecture_notes, cursor_prompt, claude_code_prompt, risks, next_steps, relevant_resources
Seuls les champs modifiés doivent être inclus. next_steps est un tableau de strings.
</brief_update_format>"""


async def chat_about_brief_async(message: str, brief_data: dict,
                                  history: list, api_key: str,
                                  exa_key: str = "",
                                  bookmarks_context: str = "") -> dict:
    """Chat about a brief. Returns {reply, updated_brief?}."""
    brief_context = json.dumps(brief_data, ensure_ascii=False, indent=2)

    messages = []
    messages.append({
        "role": "user",
        "content": (f"<brief_context>\n{brief_context}\n</brief_context>\n\n"
                    "Réponds 'OK, j'ai le contexte du brief.' pour confirmer.")
    })
    messages.append({"role": "assistant", "content": "OK, j'ai le contexte du brief."})

    for h in history[:-1]:
        if h.get("role") in ("user", "assistant"):
            messages.append({"role": h["role"], "content": h["content"]})

    user_msg = message

    # Search bookmarks if relevant keywords detected
    search_triggers = ("recherche", "cherche", "find", "search", "explore",
                       "bookmark", "base", "données", "ressource", "alternative",
                       "compétiteur", "concurrent", "approfondi", "exemple")
    exa_supplement = ""
    if exa_key and any(t in message.lower() for t in search_triggers):
        async with httpx.AsyncClient() as client:
            query = f"{message} {brief_data.get('objective', '')}"
            results = await _exa_search_async(query, exa_key, client,
                                              num_results=5)
        if results:
            exa_supplement = _format_exa_results(results, "Recherche web")

    if bookmarks_context:
        user_msg += f"\n\n<bookmarks_database>\n{bookmarks_context}\n</bookmarks_database>"

    if exa_supplement:
        user_msg += (f"\n\n<recherche_web>\n{exa_supplement}\n</recherche_web>\n"
                     "Utilise ces résultats pour enrichir ta réponse.")

    messages.append({"role": "user", "content": user_msg})

    aclient = _get_anthropic_client(api_key)
    response = await aclient.messages.create(
        model="claude-sonnet-4-6-20250514",
        max_tokens=3000,
        system=BRIEF_CHAT_SYSTEM_PROMPT,
        messages=messages,
    )
    raw_reply = response.content[0].text.strip()

    # Parse optional brief update block
    result = {"reply": raw_reply}
    update_match = re.search(
        r'<brief_update>\s*(\{.*?\})\s*</brief_update>',
        raw_reply, re.DOTALL)
    if update_match:
        try:
            updates = json.loads(update_match.group(1))
            result["updated_fields"] = updates
            # Clean the update block from the visible reply
            result["reply"] = raw_reply[:update_match.start()].strip()
        except json.JSONDecodeError:
            pass

    return result



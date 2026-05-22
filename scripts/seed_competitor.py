#!/usr/bin/env python3
"""
Seed a single competitor MD file from scratch using:
  - Jina Reader (https://r.jina.ai) for free, JS-aware page scraping
  - Google Gemini free tier (gemini-2.5-flash) for analysis

Usage:
    python scripts/seed_competitor.py interakt https://www.interakt.shop

Prerequisites (one-time setup):
    1. pip install -r requirements.txt
    2. cp .env.example .env  (then put your real GEMINI_API_KEY in .env)
"""
import sys
import os
import re
import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY or "YOUR_GEMINI_KEY" in GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set in .env.")
    print("       Get a free key at https://aistudio.google.com/apikey")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"

JINA_PREFIX = "https://r.jina.ai/"

# Compact Wati context for the analysis prompt.
WATI_CONTEXT = """
WATI (WhatsApp Team Inbox) — WhatsApp Business Solution Provider, HQ Hong Kong, 16K+ customers.
Pricing (USD/month, annual billing): Growth $59 (3 users, hard cap), Pro $119 (5 users + $39/extra),
Business $279 (5 users + $69/extra), Enterprise custom.
Strengths: easiest WhatsApp API onboarding, official Meta + Google partner, BYOA framework,
WhatsApp Business Calling, SOC 2, strong in India/Brazil/SEA/MENA/LATAM.
Weaknesses: ~20% Meta markup (highest), 3-user cap on Growth forces upgrade, no revenue attribution
or A/B testing, mobile app sluggish on Android, native AI agent upsells to Astra at ~$100/mo.
"""

# URL path candidates per content category. The scraper tries each in order
# and keeps the first one that returns substantial content.
PATH_CANDIDATES = {
    "pricing": ["/pricing", "/plans", "/price", "/pricing-plans"],
    "product": ["/features", "/product", "/solutions", "/platform", "/capabilities"],
    "about": ["/about", "/about-us", "/company", "/who-we-are"],
    "customers": ["/customers", "/case-studies", "/clients", "/success-stories", "/case-study"],
}

# Documentation site discovery — tried in order, first one returning content wins.
DOC_SUBDOMAINS = ["docs", "help", "support", "developer", "developers", "guide", "kb", "learn", "academy"]
DOC_PATH_CANDIDATES = ["/docs", "/help", "/support", "/guides", "/knowledge-base",
                       "/resources/docs", "/documentation", "/help-center",
                       "/resource-center", "/resources", "/learn", "/academy",
                       "/help-centre", "/resource-centre"]

# Cap on how many doc pages to crawl per competitor (each is a Jina request).
MAX_DOC_PAGES = 30

ANALYSIS_PROMPT = """You are Wati's competitive intelligence analyst. Analyze the competitor below against Wati.

The scraped content includes the competitor's marketing pages AND their full product documentation.
Use the documentation to find SPECIFIC feature details, integration capabilities, API behaviors,
and limits that don't appear on marketing pages — these are gold for objection handling.

WATI CONTEXT (use these exact numbers — do not guess Wati's prices):
{wati_context}

COMPETITOR: {competitor_name}

SCRAPED CONTENT FROM COMPETITOR (marketing + documentation):
{scraped_content}

Return ONLY valid JSON (no preamble, no markdown code fences):
{{
  "tldr": "2-3 sentence positioning summary of this competitor vs Wati",
  "wins": ["8-10 bullets — concrete places Wati beats this competitor, sales-usable, cite specific features"],
  "losses": ["8-10 bullets — honest places this competitor beats Wati, cite specific features from their docs"],
  "feature_notes": ["6-10 bullets — specific feature behaviors / limits / quirks from documentation that sales should know about"],
  "objections": [
    {{
      "objection": "buyer pushback in their own words",
      "response": "Wati's fact-based counter, 2-3 sentences",
      "proof_points": ["1-2 concrete data/feature points, ideally from competitor docs"]
    }}
  ],
  "pricing_battle": {{
    "summary": "2-sentence pricing positioning vs Wati",
    "by_segment": [
      {{
        "segment": "e.g. 5-agent SMB, 10K msg/mo",
        "wati_cost": "$X/mo (use Wati pricing from context above)",
        "competitor_cost": "$Y/mo",
        "winner": "wati or competitor",
        "why": "1 sentence reason"
      }}
    ]
  }}
}}

Produce 5-6 objections and 3 pricing segments (small SMB, mid-market, scale).
Use Wati's prices from the context — do not calculate or invent them.
Cite specific competitor docs/features. If data is missing, say "Data not available".
"""


def hash_content(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def fetch_via_jina(url: str, timeout: int = 60) -> str:
    """Fetch a URL through Jina Reader and return clean markdown."""
    full = JINA_PREFIX + url
    try:
        r = requests.get(full, timeout=timeout, headers={"Accept": "text/plain"})
        if r.status_code != 200:
            return ""
        return r.text.strip()
    except Exception as e:
        print(f"      [warn] fetch failed for {url}: {e}")
        return ""


def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "\n\n[Content truncated]"


def find_docs_root(base_url: str) -> str:
    """Discover the competitor's documentation site URL.
    Tries common subdomains (docs.X) first, then paths (X/docs)."""
    parsed = urlparse(base_url)
    bare_domain = parsed.netloc.replace("www.", "")

    for sub in DOC_SUBDOMAINS:
        candidate = f"https://{sub}.{bare_domain}"
        body = fetch_via_jina(candidate, timeout=30)
        if body and len(body) > 500:
            print(f"      [ok] docs root found: {candidate}")
            return candidate

    base = base_url.rstrip("/")
    for path in DOC_PATH_CANDIDATES:
        candidate = base + path
        body = fetch_via_jina(candidate, timeout=30)
        if body and len(body) > 500:
            print(f"      [ok] docs root found: {candidate}")
            return candidate

    print("      [warn] no documentation site found")
    return ""


def extract_internal_links(markdown: str, root_url: str) -> list:
    """Extract internal links from a markdown page. Returns deduplicated, same-domain URLs."""
    links = re.findall(r"\[[^\]]*\]\((https?://[^)\s]+)\)", markdown)
    root_host = urlparse(root_url).netloc
    seen = set()
    out = []
    for link in links:
        # strip anchors
        clean = link.split("#")[0].rstrip("/")
        if not clean or clean in seen:
            continue
        if urlparse(clean).netloc != root_host:
            continue
        if clean == root_url.rstrip("/"):
            continue
        seen.add(clean)
        out.append(clean)
    return out


def scrape_docs_site(docs_root: str, max_pages: int = MAX_DOC_PAGES) -> str:
    """Crawl the documentation site: fetch the index, then up to N linked pages in parallel."""
    if not docs_root:
        return ""

    index = fetch_via_jina(docs_root, timeout=60)
    if not index:
        return ""

    links = extract_internal_links(index, docs_root)[:max_pages]
    print(f"      [ok] crawling {len(links)} doc pages in parallel ...")

    pages = [f"=== DOCS INDEX: {docs_root} ===\n{index}"]

    # Parallel fetch (Jina free tier handles ~20 RPM, we use 5 workers)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_via_jina, link, 45): link for link in links}
        for fut in as_completed(futures):
            link = futures[fut]
            try:
                body = fut.result()
                if body and len(body) > 200:
                    pages.append(f"=== {link} ===\n{body}")
            except Exception as e:
                print(f"      [warn] doc page failed {link}: {e}")

    print(f"      [ok] got {len(pages) - 1} doc pages with content")
    return "\n\n".join(pages)


def scrape_competitor(base_url: str, docs_url_override: str = "") -> dict:
    """Scrape competitor: homepage + category pages + full documentation site.
    If docs_url_override is provided, skip auto-discovery and use it directly."""
    base = base_url.rstrip("/")
    print(f"[1/3] Scraping {base} via Jina Reader ...")

    content = {
        "homepage": "", "pricing": "", "product": "",
        "about": "", "customers": "", "documentation": "",
    }
    pages_scraped = 0

    # Homepage
    homepage = fetch_via_jina(base)
    if homepage:
        content["homepage"] = truncate(homepage, 8000)
        pages_scraped += 1
        print(f"      [ok] homepage ({len(homepage)} chars)")

    # Category pages
    for category, paths in PATH_CANDIDATES.items():
        for path in paths:
            url = base + path
            text = fetch_via_jina(url)
            if text and len(text) > 500:
                content[category] = truncate(text, 10000 if category == "pricing" else 8000)
                pages_scraped += 1
                print(f"      [ok] {category}: {path} ({len(text)} chars)")
                break

    # Documentation site — this is the big one the user explicitly wants
    if docs_url_override:
        print(f"      [..] using docs override: {docs_url_override}")
        docs_root = docs_url_override
    else:
        print("      [..] looking for docs site ...")
        docs_root = find_docs_root(base_url)
    if docs_root:
        docs_blob = scrape_docs_site(docs_root)
        # Docs can be huge — cap at 150K chars to stay safe within Gemini 1M context.
        content["documentation"] = truncate(docs_blob, 150000)
        pages_scraped += docs_blob.count("===") // 2

    return {
        "url": base_url,
        "pages_scraped": pages_scraped,
        "docs_root": docs_root,
        "content": content,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def analyze(competitor_name: str, scraped: dict, max_retries: int = 5) -> dict:
    """Send scraped content (incl. documentation) to Gemini and get structured analysis back.
    Retries on transient 503/overloaded errors with exponential backoff."""
    parts = []
    c = scraped["content"]
    for key in ["homepage", "pricing", "product", "about", "customers", "documentation"]:
        if c.get(key):
            parts.append(f"=== {key.upper()} ===\n{c[key]}")
    # Gemini 2.5 Flash window is ~1M tokens; we'll feed up to ~700K chars (~175K tokens).
    scraped_text = "\n\n".join(parts)[:700000]

    prompt = ANALYSIS_PROMPT.format(
        wati_context=WATI_CONTEXT.strip(),
        competitor_name=competitor_name,
        scraped_content=scraped_text,
    )

    print(f"[2/3] Analyzing with Gemini ({MODEL}) ...")
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            return json.loads(resp.text.strip())
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            is_transient = any(s in msg for s in ["503", "unavailable", "overloaded", "high demand", "rate limit", "429"])
            if not is_transient or attempt == max_retries:
                raise
            wait = min(60, 5 * (2 ** (attempt - 1)))  # 5, 10, 20, 40, 60s
            print(f"      [retry {attempt}/{max_retries}] transient error, waiting {wait}s ...")
            time.sleep(wait)
    raise last_err


def render_md(slug: str, website: str, scraped: dict, analysis: dict) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    c = scraped["content"]

    source_hashes = {
        section: hash_content(c.get(section, ""))
        for section in ["homepage", "pricing", "product", "about", "customers", "documentation"]
    }

    wins_md = "\n".join(f"- {b}" for b in analysis["wins"])
    losses_md = "\n".join(f"- {b}" for b in analysis["losses"])
    feature_notes_md = "\n".join(f"- {b}" for b in analysis.get("feature_notes", []))

    objections_md = ""
    for obj in analysis["objections"]:
        proof = ", ".join(obj.get("proof_points", []))
        objections_md += (
            f"\n### {obj['objection']}\n"
            f"**Response:** {obj['response']}\n\n"
            f"**Proof points:** {proof}\n"
        )

    pricing_rows = ""
    for seg in analysis["pricing_battle"]["by_segment"]:
        pricing_rows += (
            f"| {seg['segment']} | {seg['wati_cost']} | {seg['competitor_cost']} "
            f"| **{seg['winner']}** | {seg['why']} |\n"
        )

    return f"""---
name: {slug}
website: {website}
last_updated: {today}
source_hashes:
  homepage: {source_hashes['homepage']}
  pricing: {source_hashes['pricing']}
  product: {source_hashes['product']}
  about: {source_hashes['about']}
  customers: {source_hashes['customers']}
  documentation: {source_hashes['documentation']}
---

<!-- ANCHOR:tldr -->
## TL;DR vs Wati
{analysis['tldr']}
<!-- /ANCHOR:tldr -->

<!-- ANCHOR:wins -->
## Where We Win
{wins_md}
<!-- /ANCHOR:wins -->

<!-- ANCHOR:losses -->
## Where We Lose
{losses_md}
<!-- /ANCHOR:losses -->

<!-- ANCHOR:feature_notes -->
## Feature Notes (from their docs)
{feature_notes_md}
<!-- /ANCHOR:feature_notes -->

<!-- ANCHOR:objections -->
## Objection Handling
{objections_md}
<!-- /ANCHOR:objections -->

<!-- ANCHOR:pricing -->
## Pricing Battle
{analysis['pricing_battle']['summary']}

| Segment | Wati Cost | {slug} Cost | Winner | Why |
|---|---|---|---|---|
{pricing_rows}
<!-- /ANCHOR:pricing -->

<!-- ANCHOR:changelog -->
## Changelog
- {today}: Initial seed
<!-- /ANCHOR:changelog -->

<!-- ============================================================ -->
<!-- SOURCE LAYER — raw scraped content. Used by the refresher to  -->
<!-- detect what changed week-over-week. The Slack bot ignores it. -->
<!-- ============================================================ -->

<!-- ANCHOR:source_pricing -->
## Source: Pricing Page
{c.get('pricing') or '_(no pricing page found)_'}
<!-- /ANCHOR:source_pricing -->

<!-- ANCHOR:source_product -->
## Source: Product / Features
{c.get('product') or '_(no product page found)_'}
<!-- /ANCHOR:source_product -->

<!-- ANCHOR:source_about -->
## Source: About
{c.get('about') or '_(no about page found)_'}
<!-- /ANCHOR:source_about -->

<!-- ANCHOR:source_customers -->
## Source: Customers
{c.get('customers') or '_(no customers page found)_'}
<!-- /ANCHOR:source_customers -->

<!-- ANCHOR:source_documentation -->
## Source: Documentation
{c.get('documentation') or '_(no documentation site found)_'}
<!-- /ANCHOR:source_documentation -->
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", help="Competitor slug, e.g. 'interakt'")
    ap.add_argument("url", help="Competitor website URL, e.g. https://www.interakt.shop")
    ap.add_argument("--docs-url", default="",
                    help="Override docs site URL (e.g. https://www.interakt.shop/resource-center/)")
    ap.add_argument("--from-snapshot", default="",
                    help="Skip scraping and use an existing snapshot JSON file (path)")
    args = ap.parse_args()

    competitors_dir = ROOT / "competitors"
    snapshots_dir = competitors_dir / "_snapshots"

    if args.from_snapshot:
        print(f"[1/3] Loading existing snapshot: {args.from_snapshot}")
        scraped = json.loads(Path(args.from_snapshot).read_text())
    else:
        scraped = scrape_competitor(args.url, docs_url_override=args.docs_url)
    print(f"      Scraped {scraped['pages_scraped']} pages total.")

    snapshot_path = snapshots_dir / f"{args.slug}-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
    if not args.from_snapshot:
        snapshot_path.write_text(json.dumps(scraped, indent=2))

    analysis = analyze(args.slug, scraped)

    print("[3/3] Writing MD ...")
    md = render_md(args.slug, args.url, scraped, analysis)
    md_path = competitors_dir / f"{args.slug}.md"
    md_path.write_text(md)

    print()
    print(f"Done. MD written to: {md_path}")
    print(f"Snapshot saved to:   {snapshot_path}")


if __name__ == "__main__":
    main()

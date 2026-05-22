#!/usr/bin/env python3
"""
Diff-only refresher for a single competitor MD file.

Logic:
  1. Read the existing competitor MD.
  2. Extract the previous source_hashes from frontmatter and the website URL.
  3. Re-scrape the live site (same logic as seed_competitor.py).
  4. Compare new source hashes to old hashes.
  5. If NOTHING changed -> exit "no changes" (no LLM call, no commit).
  6. If anything changed -> ONE Gemini call to re-derive the analysis sections
     (TL;DR, wins, losses, feature notes, objections, pricing battle).
  7. Replace ONLY the anchored MD sections in place, update source_hashes +
     last_updated + Changelog, and write the file back.

This keeps the weekly refresh inside the Gemini free tier (1 call per changed
competitor, ~7 competitors total -> well under 250/day).

Usage:
    python scripts/refresh_competitor.py competitors/interakt.md
    python scripts/refresh_competitor.py competitors/interakt.md --force
"""
import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Re-use everything from the seed script
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from seed_competitor import (  # noqa: E402
    scrape_competitor, analyze, hash_content,
)

ANCHOR_RE = re.compile(
    r"<!-- ANCHOR:(?P<name>[a-z_]+) -->(?P<body>.*?)<!-- /ANCHOR:(?P=name) -->",
    re.DOTALL,
)
FRONTMATTER_RE = re.compile(r"^---\n(?P<fm>.*?)\n---\n", re.DOTALL)


def parse_frontmatter(md: str) -> dict:
    """Very small YAML-ish parser — only handles the flat keys we write
    plus the nested source_hashes block. Avoids a PyYAML dependency."""
    m = FRONTMATTER_RE.match(md)
    if not m:
        return {}
    out = {"source_hashes": {}}
    in_hashes = False
    for line in m.group("fm").splitlines():
        if not line.strip():
            continue
        if line.startswith("source_hashes:"):
            in_hashes = True
            continue
        if in_hashes and line.startswith("  "):
            k, _, v = line.strip().partition(":")
            out["source_hashes"][k.strip()] = v.strip()
            continue
        in_hashes = False
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def replace_anchor(md: str, name: str, new_body: str) -> str:
    """Swap an anchored section's body, preserving the anchor tags."""
    pattern = re.compile(
        rf"(<!-- ANCHOR:{name} -->)(.*?)(<!-- /ANCHOR:{name} -->)",
        re.DOTALL,
    )
    if not pattern.search(md):
        print(f"      [warn] anchor {name!r} not found in MD; skipping")
        return md
    # Use a function replacement so $ and \ in the body aren't re-interpreted.
    return pattern.sub(lambda _m: f"<!-- ANCHOR:{name} -->\n{new_body}\n<!-- /ANCHOR:{name} -->", md)


def render_analysis_anchors(slug: str, analysis: dict) -> dict:
    """Build the body string for each derived-analysis anchor.
    Mirrors the layout in seed_competitor.render_md so the file stays consistent."""
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

    pricing_body = (
        f"## Pricing Battle\n"
        f"{analysis['pricing_battle']['summary']}\n\n"
        f"| Segment | Wati Cost | {slug} Cost | Winner | Why |\n"
        f"|---|---|---|---|---|\n"
        f"{pricing_rows}"
    )

    return {
        "tldr": f"## TL;DR vs Wati\n{analysis['tldr']}",
        "wins": f"## Where We Win\n{wins_md}",
        "losses": f"## Where We Lose\n{losses_md}",
        "feature_notes": f"## Feature Notes (from their docs)\n{feature_notes_md}",
        "objections": f"## Objection Handling\n{objections_md}",
        "pricing": pricing_body,
    }


def update_frontmatter(md: str, new_hashes: dict, today: str) -> str:
    """Replace the source_hashes block and last_updated field in frontmatter."""
    def replace_block(m: re.Match) -> str:
        fm = m.group("fm")
        # Update last_updated
        fm = re.sub(r"last_updated:\s*\S+", f"last_updated: {today}", fm)
        # Rebuild the source_hashes block
        hashes_block = "source_hashes:\n" + "\n".join(
            f"  {k}: {v}" for k, v in new_hashes.items()
        )
        fm = re.sub(
            r"source_hashes:\n(?:  \w+:\s*\S+\n?)+",
            hashes_block + "\n",
            fm,
            count=1,
        )
        return f"---\n{fm}\n---\n"

    return FRONTMATTER_RE.sub(replace_block, md, count=1)


def append_changelog(md: str, today: str, changed_sections: list) -> str:
    """Add a dated line to the Changelog anchor."""
    pattern = re.compile(
        r"(<!-- ANCHOR:changelog -->)(.*?)(<!-- /ANCHOR:changelog -->)",
        re.DOTALL,
    )
    m = pattern.search(md)
    if not m:
        return md
    existing = m.group(2).rstrip()
    new_line = f"- {today}: Refreshed — changed: {', '.join(changed_sections)}"
    new_body = existing + "\n" + new_line + "\n"
    return pattern.sub(
        lambda _m: f"<!-- ANCHOR:changelog -->{new_body}<!-- /ANCHOR:changelog -->",
        md,
    )


def refresh(md_path: Path, force: bool = False) -> dict:
    """Refresh one competitor MD. Returns a status dict for reporting."""
    md = md_path.read_text()
    fm = parse_frontmatter(md)
    slug = fm.get("name", md_path.stem)
    website = fm.get("website", "")
    if not website:
        return {"slug": slug, "status": "error", "reason": "no website in frontmatter"}

    old_hashes = fm.get("source_hashes", {})
    docs_url_override = fm.get("docs_url", "")  # optional; honour if user set it

    print(f"\n=== Refreshing {slug} ({website}) ===")
    scraped = scrape_competitor(website, docs_url_override=docs_url_override)

    new_hashes = {
        section: hash_content(scraped["content"].get(section, ""))
        for section in ["homepage", "pricing", "product", "about", "customers", "documentation"]
    }

    changed = [s for s, h in new_hashes.items() if old_hashes.get(s) != h]
    if not changed and not force:
        print(f"      [skip] no source changes for {slug}")
        return {"slug": slug, "status": "unchanged", "changed_sections": []}

    print(f"      [diff] changed sections: {changed or '(forced)'}")

    # ONE Gemini call to re-derive analysis from the new scrape.
    analysis = analyze(slug, scraped)

    # Swap derived-analysis anchors
    anchors = render_analysis_anchors(slug, analysis)
    for name, body in anchors.items():
        md = replace_anchor(md, name, body)

    # Swap source anchors (keeps the file's source layer accurate)
    md = replace_anchor(md, "source_pricing",
                        f"## Source: Pricing Page\n{scraped['content'].get('pricing') or '_(no pricing page found)_'}")
    md = replace_anchor(md, "source_product",
                        f"## Source: Product / Features\n{scraped['content'].get('product') or '_(no product page found)_'}")
    md = replace_anchor(md, "source_about",
                        f"## Source: About\n{scraped['content'].get('about') or '_(no about page found)_'}")
    md = replace_anchor(md, "source_customers",
                        f"## Source: Customers\n{scraped['content'].get('customers') or '_(no customers page found)_'}")
    md = replace_anchor(md, "source_documentation",
                        f"## Source: Documentation\n{scraped['content'].get('documentation') or '_(no documentation site found)_'}")

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    md = update_frontmatter(md, new_hashes, today)
    md = append_changelog(md, today, changed or ["forced"])

    md_path.write_text(md)

    # Drop a snapshot too, for forensic diffing.
    snap_dir = md_path.parent / "_snapshots"
    snap_dir.mkdir(exist_ok=True)
    snap_path = snap_dir / f"{slug}-{today}.json"
    snap_path.write_text(json.dumps(scraped, indent=2))

    print(f"      [ok] refreshed {slug}; snapshot at {snap_path}")
    return {"slug": slug, "status": "refreshed", "changed_sections": changed or ["forced"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md_path", help="Path to competitor MD, e.g. competitors/interakt.md")
    ap.add_argument("--force", action="store_true",
                    help="Re-analyze even if no source hashes changed")
    args = ap.parse_args()

    result = refresh(Path(args.md_path), force=args.force)
    print(f"\nResult: {json.dumps(result)}")
    if result["status"] == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()

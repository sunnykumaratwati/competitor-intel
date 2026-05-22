#!/usr/bin/env python3
"""
Refresh every competitor MD under competitors/.
Prints a JSON summary at the end so the GitHub Actions workflow can use it.
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from refresh_competitor import refresh  # noqa: E402


def main():
    competitors_dir = ROOT / "competitors"
    md_files = sorted(p for p in competitors_dir.glob("*.md"))

    if not md_files:
        print("No competitor MD files found.")
        return

    results = []
    for md in md_files:
        try:
            results.append(refresh(md))
        except Exception as e:
            print(f"      [error] {md.name}: {e}")
            results.append({"slug": md.stem, "status": "error", "reason": str(e)})

    refreshed = [r for r in results if r["status"] == "refreshed"]
    unchanged = [r for r in results if r["status"] == "unchanged"]
    errors = [r for r in results if r["status"] == "error"]

    print("\n=== Refresh Summary ===")
    print(f"  refreshed: {len(refreshed)}  unchanged: {len(unchanged)}  errors: {len(errors)}")
    for r in refreshed:
        print(f"    [refreshed] {r['slug']} — sections: {', '.join(r['changed_sections'])}")
    for r in errors:
        print(f"    [error] {r['slug']} — {r['reason']}")

    # Machine-readable line for the workflow to grep.
    print("\nSUMMARY_JSON=" + json.dumps({
        "refreshed": [r["slug"] for r in refreshed],
        "unchanged": [r["slug"] for r in unchanged],
        "errors": [r["slug"] for r in errors],
        "details": results,
    }))

    if errors and not refreshed:
        sys.exit(1)


if __name__ == "__main__":
    main()

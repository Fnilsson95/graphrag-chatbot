"""Compare two GraphRAG output directories side-by-side.

Usage:
    uv run python scripts/compare_indexes.py output_before output

Pipe to a file to save:
    uv run python scripts/compare_indexes.py output_before
    output > eval/index-diff.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load(directory: Path, name: str) -> pd.DataFrame | None:
    # Load a parquet table from a GraphRAG output directory,
    # or None if missing.
    p = directory / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def types_table(entities: pd.DataFrame, top: int = 20) -> str:
    counts = entities["type"].value_counts().head(top)
    total = len(entities)
    lines = ["| Type | Count | % |", "|---|---:|---:|"]
    for t, c in counts.items():
        lines.append(f"| `{t}` | {c} | {100 * c / total:.1f}% |")
    return "\n".join(lines)


def top_entities(
    entities: pd.DataFrame, rels: pd.DataFrame, top: int = 20
) -> str:
    deg = (
        rels["source"]
        .value_counts()
        .add(rels["target"].value_counts(), fill_value=0)
        .sort_values(ascending=False)
        .head(top)
    )
    name_col = "title" if "title" in entities.columns else "name"
    type_map = entities.set_index(name_col)["type"].to_dict()
    lines = ["| Entity | Type | Degree |", "|---|---|---:|"]
    for name, d in deg.items():
        lines.append(f"| `{name}` | `{type_map.get(name, '?')}` | {int(d)} |")
    return "\n".join(lines)


def community_sample(reports: pd.DataFrame, top: int = 10) -> str:
    rating_col = next(
        (
            c
            for c in (
                "rating",
                "importance",
                "severity_rating",
                "community_rank",
                "rank",
            )
            if c in reports.columns
        ),
        None,
    )
    cols = [
        c for c in ("title", rating_col, "level") if c and c in reports.columns
    ]
    sample = (
        reports.sort_values(rating_col, ascending=False)
        if rating_col
        else reports
    ).head(top)[cols]

    header = (
        "| " + " | ".join(c.replace("_", " ").title() for c in cols) + " |"
    )
    sep = "|" + "|".join("---" if c == "title" else "---:" for c in cols) + "|"
    lines = [header, sep]
    for _, row in sample.iterrows():
        lines.append(
            "| " + " | ".join(str(row.get(c, "?")) for c in cols) + " |"
        )
    return "\n".join(lines)


def stats_summary(directory: Path) -> str:
    p = directory / "stats.json"
    if not p.exists():
        return "_(no stats.json)_"
    s = json.loads(p.read_text())
    return (
        f"- total_runtime: {s.get('total_runtime', 0):.1f}s\n"
        f"- num_documents: {s.get('num_documents', 0)}"
    )


def render(
    label: str, before: str, after: str, before_dir: Path, after_dir: Path
) -> str:
    return (
        f"\n## {label}\n\n"
        f"### Before (`{before_dir}`)\n\n{before}\n\n"
        f"### After (`{after_dir}`)\n\n{after}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    args = ap.parse_args()

    print("# GraphRAG output comparison\n")
    print(f"- Before: `{args.before}`")
    print(f"- After:  `{args.after}`")

    be = load(args.before, "entities")
    ae = load(args.after, "entities")
    br = load(args.before, "relationships")
    ar = load(args.after, "relationships")
    bc = load(args.before, "community_reports")
    ac = load(args.after, "community_reports")

    print(
        render(
            "Counts",
            f"- Entities: {len(be) if be is not None else 'n/a'}\n"
            f"- Relationships: {len(br) if br is not None else 'n/a'}\n"
            f"- Community reports: {len(bc) if bc is not None else 'n/a'}\n\n"
            f"**stats.json:**\n{stats_summary(args.before)}",
            f"- Entities: {len(ae) if ae is not None else 'n/a'}\n"
            f"- Relationships: {len(ar) if ar is not None else 'n/a'}\n"
            f"- Community reports: {len(ac) if ac is not None else 'n/a'}\n\n"
            f"**stats.json:**\n{stats_summary(args.after)}",
            args.before,
            args.after,
        )
    )

    if be is not None and ae is not None:
        print(
            render(
                "Entity types (top 20)",
                types_table(be),
                types_table(ae),
                args.before,
                args.after,
            )
        )

    if be is not None and br is not None and ae is not None and ar is not None:
        print(
            render(
                "Top 20 entities by degree",
                top_entities(be, br),
                top_entities(ae, ar),
                args.before,
                args.after,
            )
        )

    if bc is not None and ac is not None:
        print(
            render(
                "Top 10 community reports by rating",
                community_sample(bc),
                community_sample(ac),
                args.before,
                args.after,
            )
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate data/corpus_stats.json from cms_corpus.json and templates.json.

Run after any corpus rebuild (parse_cms_data.py) to refresh stats:
    python scripts/generate_corpus_stats.py
"""

from __future__ import annotations

import json
import os
from collections import Counter

ROOT = os.path.join(os.path.dirname(__file__), "..")
CORPUS_PATH = os.path.join(ROOT, "data", "cms_corpus.json")
TEMPLATES_PATH = os.path.join(ROOT, "data", "templates.json")
OUTPUT_PATH = os.path.join(ROOT, "data", "corpus_stats.json")


def difficulty_label(min_coverage_chunks: int) -> str:
    if min_coverage_chunks >= 3:
        return "hard"
    if min_coverage_chunks >= 2:
        return "medium"
    return "easy"


def generate(corpus_path: str = CORPUS_PATH,
             templates_path: str = TEMPLATES_PATH,
             output_path: str = OUTPUT_PATH) -> dict:
    with open(corpus_path) as f:
        corpus = json.load(f)
    with open(templates_path) as f:
        templates = json.load(f)

    total = len(corpus)
    proc_counts: Counter = Counter()
    proc_section: Counter = Counter()

    for chunk in corpus:
        pc = chunk["procedure_code"]
        st = chunk["section_type"]
        proc_counts[pc] += 1
        proc_section[(pc, st)] += 1

    section_type_counts = {
        st: sum(1 for c in corpus if c["section_type"] == st)
        for st in ("coverage_criteria", "exclusions", "billing")
    }

    procedures: dict = {}
    for pc in sorted(templates.keys()):
        n = proc_counts.get(pc, 0)
        cov = proc_section.get((pc, "coverage_criteria"), 0)
        exc = proc_section.get((pc, "exclusions"), 0)
        bill = proc_section.get((pc, "billing"), 0)
        mc = templates[pc].get("min_coverage_chunks", 1)
        procedures[pc] = {
            "name": templates[pc]["name"],
            "total_chunks": n,
            "corpus_pct": round(n / total * 100, 1) if total else 0.0,
            "coverage_criteria": cov,
            "exclusions": exc,
            "billing": bill,
            "min_coverage_chunks": mc,
            "difficulty": difficulty_label(mc),
        }

    stats = {
        "total_chunks": total,
        "total_procedures": len(templates),
        "section_type_counts": section_type_counts,
        "procedures": procedures,
    }

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def _print_table(stats: dict) -> None:
    procs = stats["procedures"]
    header = f"{'Proc':>6}  {'Name':<35}  {'Total':>5}  {'Pct':>5}  {'Cov':>4}  {'Exc':>4}  {'Bill':>4}  {'min_cov':>7}  {'Diff':<6}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for pc, p in sorted(procs.items(), key=lambda x: -x[1]["total_chunks"]):
        print(
            f"{pc:>6}  {p['name']:<35}  {p['total_chunks']:>5}  "
            f"{p['corpus_pct']:>4.1f}%  {p['coverage_criteria']:>4}  "
            f"{p['exclusions']:>4}  {p['billing']:>4}  "
            f"{p['min_coverage_chunks']:>7}  {p['difficulty']:<6}"
        )
    print(sep)
    sc = stats["section_type_counts"]
    print(
        f"{'TOTAL':>6}  {'':<35}  {stats['total_chunks']:>5}  "
        f"{'100%':>5}  {sc['coverage_criteria']:>4}  "
        f"{sc['exclusions']:>4}  {sc['billing']:>4}"
    )


if __name__ == "__main__":
    stats = generate()
    _print_table(stats)
    print(f"\nWrote {OUTPUT_PATH}")

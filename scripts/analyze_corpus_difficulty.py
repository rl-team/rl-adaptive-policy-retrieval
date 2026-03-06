#!/usr/bin/env python3
"""Corpus solvability and retrieval difficulty analysis.

Measures how well the semantic retriever surfaces procedure-specific chunks
at various k values, and reports per-procedure accuracy, coverage chunk
retrieval rates, and the overall FixedK accuracy gradient.

This script is reusable when:
 - The corpus (cms_corpus.json) or templates (templates.json) change.
 - You need to verify solvability after adjusting min_coverage_chunks.
 - You want to generate difficulty statistics for the Observations tab.

Usage:
    python scripts/analyze_corpus_difficulty.py
    python scripts/analyze_corpus_difficulty.py --episodes 100 --seed 42
"""

from __future__ import annotations

import argparse
import sys
import os
import json

import numpy as np

# Allow running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.pa_simulator import PASimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corpus difficulty analysis.")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per procedure per k value (default: 50)")
    parser.add_argument("--seed", type=int, default=99, help="Random seed")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON instead of human-readable table")
    return parser.parse_args()


def analyze(episodes: int = 50, seed: int = 99) -> dict:
    """Run solvability analysis and return structured results."""
    sim = PASimulator(seed=seed)
    corpus = sim.get_corpus()
    templates = sim._oracle.templates
    procedures = sorted(templates.keys())

    shared = sum(1 for c in corpus if len(c.procedure_codes) > 1)

    results = {
        "corpus": {
            "total_chunks": len(corpus),
            "shared_chunks": shared,
            "procedures": len(procedures),
            "coverage": sum(1 for c in corpus if c.section_type == "coverage_criteria"),
            "exclusions": sum(1 for c in corpus if c.section_type == "exclusions"),
            "billing": sum(1 for c in corpus if c.section_type == "billing"),
        },
        "per_procedure": {},
        "overall": {},
    }

    # Per-procedure solvability
    for proc_code in procedures:
        template = templates[proc_code]
        mc = template.get("min_coverage_chunks", 1)
        proc_cov_total = len([
            c for c in corpus
            if c.section_type == "coverage_criteria" and proc_code in c.procedure_codes
        ])

        acc_at_k = {}
        cov_at_k = {}
        for k in [3, 5, 7, 10]:
            correct = 0
            cov_retrieved = []
            for _ in range(episodes):
                request = sim.generate_request(procedure_code=proc_code)
                gt = sim.oracle_decision(request, corpus)
                query_emb = sim.encode(request.to_text())
                candidates = sim.get_top_k_candidates(query_emb, k=k)
                retrieved = [sim.get_chunk(idx) for idx in candidates]
                decision = sim.oracle_decision(request, retrieved)
                correct += int(decision == gt)
                n_cov = len([
                    c for c in retrieved
                    if c.section_type == "coverage_criteria" and proc_code in c.procedure_codes
                ])
                cov_retrieved.append(n_cov)
            acc_at_k[k] = correct / episodes
            cov_at_k[k] = float(np.mean(cov_retrieved))

        results["per_procedure"][proc_code] = {
            "name": template["name"],
            "min_coverage_chunks": mc,
            "coverage_chunks_in_corpus": proc_cov_total,
            "accuracy": acc_at_k,
            "avg_coverage_retrieved": cov_at_k,
        }

    # Overall FixedK accuracy
    for k in [3, 5, 7, 10]:
        total_correct = 0
        total_n = 0
        for proc_code in procedures:
            for _ in range(episodes):
                request = sim.generate_request(procedure_code=proc_code)
                gt = sim.oracle_decision(request, corpus)
                query_emb = sim.encode(request.to_text())
                candidates = sim.get_top_k_candidates(query_emb, k=k)
                retrieved = [sim.get_chunk(idx) for idx in candidates]
                decision = sim.oracle_decision(request, retrieved)
                total_correct += int(decision == gt)
                total_n += 1
        results["overall"][str(k)] = {
            "correct": total_correct,
            "total": total_n,
            "accuracy": total_correct / total_n,
        }

    return results


def print_table(results: dict) -> None:
    """Pretty-print the results as human-readable tables."""
    c = results["corpus"]
    print("=" * 100)
    print("CORPUS DIFFICULTY ANALYSIS")
    print("=" * 100)
    print(f"\n  Corpus: {c['total_chunks']} chunks ({c['shared_chunks']} shared)")
    print(f"  Types:  coverage={c['coverage']}, exclusions={c['exclusions']}, billing={c['billing']}")
    print(f"  Procedures: {c['procedures']}")

    print(f"\n{'Proc':>6s}  {'Name':35s}  {'mc':>2s}  {'cov':>3s}  "
          f"{'k=3':>5s}  {'k=5':>5s}  {'k=7':>5s}  {'k=10':>5s}  "
          f"{'cov@7':>5s}  {'Status':>8s}")
    print("-" * 100)

    for pc, p in sorted(results["per_procedure"].items(), key=lambda x: x[1]["accuracy"][7], reverse=True):
        mc = p["min_coverage_chunks"]
        cov = p["coverage_chunks_in_corpus"]
        acc = p["accuracy"]
        cov7 = p["avg_coverage_retrieved"][7]
        status = "PASS" if acc[7] >= 0.10 else "FAIL"
        print(f"{pc:>6s}  {p['name'][:35]:35s}  {mc:>2d}  {cov:>3d}  "
              f"{acc[3]:>5.0%}  {acc[5]:>5.0%}  {acc[7]:>5.0%}  {acc[10]:>5.0%}  "
              f"{cov7:>5.1f}  {status:>8s}")

    print("-" * 100)
    print("\nOverall FixedK Accuracy:")
    for k_str, ov in sorted(results["overall"].items(), key=lambda x: int(x[0])):
        print(f"  k={k_str}: {ov['correct']}/{ov['total']} = {ov['accuracy']:.0%}")
    print("=" * 100)


if __name__ == "__main__":
    args = parse_args()
    results = analyze(episodes=args.episodes, seed=args.seed)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_table(results)

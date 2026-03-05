#!/usr/bin/env python3
"""Raw CMS CSVs -> data/cms_parsed.json -> data/cms_corpus.json."""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.parser import parse_cms_data
from simulator.chunker import build_corpus

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    parser = argparse.ArgumentParser(
        description="Parse raw CMS CSVs into cms_parsed.json and cms_corpus.json."
    )
    parser.add_argument(
        "--save-sources",
        action="store_true",
        help=(
            "Write auto-discovered sources back into data/templates.json for any "
            "procedure that had no sources defined. Safe to run repeatedly."
        ),
    )
    args = parser.parse_args()

    discovered: dict | None = {} if args.save_sources else None

    # Step 1: parse raw CMS data
    parsed = parse_cms_data(discovered=discovered)
    parsed_path = os.path.join(DATA_DIR, "cms_parsed.json")
    with open(parsed_path, "w") as f:
        json.dump(parsed, f, indent=2)
    for proc, sections in parsed.items():
        print(f"[parse] {proc}: {len(sections)} sections")

    # Step 2: chunk into corpus
    corpus = build_corpus(parsed)
    corpus_path = os.path.join(DATA_DIR, "cms_corpus.json")
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2)
    print(f"[chunk] {len(corpus)} chunks -> {corpus_path}")

    # Step 3 (optional): persist auto-discovered sources to templates.json
    if args.save_sources and discovered:
        templates_path = os.path.join(DATA_DIR, "templates.json")
        with open(templates_path) as f:
            templates = json.load(f)
        for proc_code, sources in discovered.items():
            templates[proc_code]["sources"] = sources
        with open(templates_path, "w") as f:
            json.dump(templates, f, indent=2)
        print(
            f"[save-sources] Wrote sources for {len(discovered)} procedure(s) "
            f"-> {templates_path}"
        )


if __name__ == "__main__":
    main()

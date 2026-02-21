#!/usr/bin/env python3
"""Raw CMS CSVs -> data/cms_parsed.json -> data/cms_corpus.json."""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.parser import parse_cms_data
from simulator.chunker import build_corpus

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    # Step 1: parse raw CMS data
    parsed = parse_cms_data()
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


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import os
import sys

from simulator.parser import get_sources_for_code

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_sources_for_code.py <HCPCS_CODE>")
        sys.exit(1)

    code = sys.argv[1]
    res = get_sources_for_code(code)
    print(json.dumps({code: res}, indent=2))

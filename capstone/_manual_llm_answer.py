#!/usr/bin/env python3
"""Write my answer into the provider's cache at the exact path _manual_llm_demo.py
printed, so a re-run of that driver sees a cache hit for that decision step.

Usage: python3 _manual_llm_answer.py <cache_path> <response_json_file>
"""
import json
import sys

cache_path, response_file = sys.argv[1], sys.argv[2]
response = open(response_file, encoding="utf-8").read()
json.dump({"prompt": "(answered manually, see data/manual_llm_trace.jsonl for the real prompt)",
           "response": response, "model": "claude-sonnet-5(manual)", "usage": {}},
          open(cache_path, "w", encoding="utf-8"))
print("wrote", cache_path)

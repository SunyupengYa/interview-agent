#!/usr/bin/env python3
"""检索质量评测：hit@k —— 期望来源是否出现在 topK 检索结果里。

用法: python3 eval/eval.py [base_url]
依赖: 服务已启动（检索不需要 LLM key）
"""
import json
import sys
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:18080"
TOP_K = 3

cases = [json.loads(line) for line in open(f"{__file__.rsplit('/', 1)[0]}/eval.jsonl", encoding="utf-8") if line.strip()]
hit = 0
for c in cases:
    url = f"{BASE}/api/search?q={urllib.parse.quote(c['q'])}&topK={TOP_K}"
    with urllib.request.urlopen(url, timeout=30) as r:
        d = json.load(r)
    got = [h["source"] for h in d["hits"]]
    ok = c["expect_source"] in got
    hit += ok
    mark = "PASS" if ok else "MISS"
    top = f"{got[0]}({d['hits'][0]['score']:.3f})" if got else "-"
    print(f"[{mark}] {c['q']}  期望={c['expect_source']}  top1={top}")

print(f"\nhit@{TOP_K} = {hit}/{len(cases)} = {hit / len(cases):.0%}")

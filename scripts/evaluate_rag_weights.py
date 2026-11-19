"""Run an explicit read-only RAG weight grid against the fixed evaluation set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.retrieval.runner import write_weight_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 RAG 规则融合权重网格（只读）")
    parser.add_argument("--weights", required=True, help="包含权重对象数组的 JSON 文件")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strategy", default="rule", choices=("rule", "cross_encoder", "rrf"))
    parser.add_argument("--backend", default="local_hybrid")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    candidates = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or not all(isinstance(item, dict) for item in candidates):
        parser.error("--weights must be a JSON array of objects")
    target, payload = write_weight_grid(
        Path(args.output), candidates, strategy=args.strategy,
        top_k=args.top_k, backend_override=args.backend,
    )
    print(json.dumps({"path": str(target), "candidates": len(payload["results"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()

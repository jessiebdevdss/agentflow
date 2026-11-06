"""Command line interface.

    agentflow run "your question" [--mock] [--json]
    agentflow eval [--dataset evals/dataset.json] [--mock] [--no-judge] [--out report.json]
    agentflow tools            # print the tool JSON schemas sent to the model
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from agentflow.config import Settings
from agentflow.graph import AgentFlow
from agentflow.tools import default_registry

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "evals" / "dataset.json"


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="agentflow", description="Multi-agent research assistant (LangGraph + DeepSeek).")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Answer a research question")
    r.add_argument("question")
    r.add_argument("--mock", action="store_true", help="Use the offline mock LLM and offline tools")
    r.add_argument("--json", action="store_true", help="Print the full result as JSON")

    e = sub.add_parser("eval", help="Run trajectory evaluation over a dataset")
    e.add_argument("--dataset", default=str(DEFAULT_DATASET))
    e.add_argument("--mock", action="store_true")
    e.add_argument("--no-judge", action="store_true", help="Skip the LLM-as-judge score")
    e.add_argument("--out", help="Write the JSON report to this file")

    sub.add_parser("tools", help="Print tool schemas")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.cmd == "tools":
        print(json.dumps(default_registry(Settings.from_env(mock=True)).schemas(), indent=2))
        return 0

    settings = Settings.from_env(mock=getattr(args, "mock", False))
    if settings.use_mock_llm:
        print("[agentflow] mock mode: no DEEPSEEK_API_KEY (or --mock); using offline LLM + tools.", file=sys.stderr)
    flow = AgentFlow(settings)

    if args.cmd == "run":
        res = flow.run(args.question)
        if args.json:
            d = asdict(res)
            d.pop("trajectory")
            print(json.dumps(d, indent=2, default=str))
        else:
            print(res.answer)
            print("\n---")
            print(f"run_id={res.run_id}  revisions={res.revisions}  critic={res.critique.get('score')}  "
                  f"tool_calls={len(res.tool_log)}  errors={len(res.errors)}  {res.duration_s}s")
            if res.trajectory_path:
                print(f"trajectory: {res.trajectory_path}")
        return 0

    if args.cmd == "eval":
        from agentflow.evaluation import load_dataset, run_eval

        report = run_eval(flow, load_dataset(args.dataset), judge=not args.no_judge)
        text = json.dumps(report, indent=2)
        if args.out:
            Path(args.out).write_text(text)
        print(json.dumps(report["summary"], indent=2))
        for row in report["results"]:
            print(f"  {row['case_id']:<24} overall={row['overall']:.2f} recall={row['tool_recall']:.2f} "
                  f"success={row['tool_success_rate']:.2f} citations={row['citations']}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

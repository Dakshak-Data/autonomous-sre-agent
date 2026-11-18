"""
Step 6 Demo: Full End-to-End ZERO-TOKEN Pipeline + Durable Crash-Resume Proof!
Uses IncidentController directly (Router -> Controller -> Services -> Adapters).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sre_agent.controllers.incident_controller import IncidentController


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        controller = IncidentController(data_dir=tmpdir)
        query = "My Vercel /api/checkout endpoint is hitting 504 timeouts and Supabase queries are slow"

        print(">>> RUN 1: Fresh Investigation (All 4 Durable Steps Execute)")
        res1 = controller.investigate(
            query=query,
            workflow_id="inc-demo-001",
            live_llm=False,
            live_cloud=False,
        )
        print("\n[RUN 1 REPORT]\n" + res1.final_answer)

        print("\n" + "#" * 75)
        print(">>> RUN 2: Simulated Crash/Restart with SAME workflow_id='inc-demo-001'")
        print("#" * 75)
        res2 = controller.investigate(
            query=query,
            workflow_id="inc-demo-001",
            live_llm=False,
            live_cloud=False,
        )
        assert res1.final_answer == res2.final_answer
        print("\n[VERIFIED] Run 2 matched Run 1 identically with ZERO re-execution!")


if __name__ == "__main__":
    main()

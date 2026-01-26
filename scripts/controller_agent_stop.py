"""
Controller script: run for up to 1 hour, announce STOP to agents, steer toward
best implementation, and early-stop when 90%+ of recent implementations are worse.

- Reads output/agent_coordination.txt (PLAN / RESULTS blocks).
- Writes output/controller_announcements.txt (STOP + steer, or KEEP GOING + best).
- Run in background: python scripts/controller_agent_stop.py

Agents must read controller_announcements.txt and stop when they see STOP.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


COORDINATION_FILE = "output/agent_coordination.txt"
ANNOUNCEMENTS_FILE = "output/controller_announcements.txt"
DEFAULT_DURATION_SEC = 3600   # 1 hour
DEFAULT_CHECK_INTERVAL_SEC = 300   # 5 min
DEFAULT_NEW_WINDOW = 5       # last N completed implementations
DEFAULT_WORST_FRACTION = 0.9  # 90%+ worse => early stop


@dataclass
class Result:
    method_id: str
    method: str
    script: str
    output_csv: str
    mae_vs_baseline: float | None
    mae_u_v: float | None
    runtime_sec: float | None
    outliers: str   # PASS | FAIL
    variance: str
    spatial_global: str
    raw: str = ""

    @property
    def sanity_pass_count(self) -> int:
        n = 0
        if self.outliers.upper() == "PASS":
            n += 1
        if self.variance.upper() == "PASS":
            n += 1
        if self.spatial_global.upper() == "PASS":
            n += 1
        return n

    def is_worse_than(self, other: Result) -> bool:
        """True if self is worse than other (fewer sanity passes or higher MAE)."""
        if self.sanity_pass_count != other.sanity_pass_count:
            return self.sanity_pass_count < other.sanity_pass_count
        # Tie-break: lower MAE vs baseline is better
        a = self.mae_vs_baseline if self.mae_vs_baseline is not None else float("inf")
        b = other.mae_vs_baseline if other.mae_vs_baseline is not None else float("inf")
        return a > b


def _parse_float(s: str) -> float | None:
    if s is None or not s.strip():
        return None
    try:
        return float(s.strip())
    except ValueError:
        return None


def _parse_results_blocks(content: str) -> list[Result]:
    """Parse RESULTS blocks from coordination file. Returns list in order of appearance."""
    results: list[Result] = []
    # Split by --- but keep block structure
    blocks = re.split(r'\n---\n', content)
    for raw_block in blocks:
        block = raw_block.strip()
        if "RESULTS:" not in block or "Status: DONE" not in block:
            continue
        lines = block.splitlines()
        method_id = ""
        method = ""
        script = ""
        output_csv = ""
        mae_vs_baseline: float | None = None
        mae_u_v: float | None = None
        runtime_sec: float | None = None
        outliers = "FAIL"
        variance = "FAIL"
        spatial_global = "FAIL"

        for line in lines:
            if "| Agent / Method ID:" in line:
                m = re.search(r"Agent / Method ID:\s*(\S+)", line)
                if m:
                    method_id = m.group(1).strip()
            elif line.strip().startswith("Method:"):
                method = line.split("Method:", 1)[1].strip()
            elif line.strip().startswith("Script:"):
                script = line.split("Script:", 1)[1].strip()
            elif line.strip().startswith("Output CSV:"):
                output_csv = line.split("Output CSV:", 1)[1].strip()
            elif "Baseline comparison:" in line:
                rest = line.split("Baseline comparison:", 1)[-1].strip()
                for part in rest.replace(",", " ").split():
                    if "MAE_vs_baseline=" in part:
                        m = re.search(r"MAE_vs_baseline=([\d.eE+-]+)", part)
                        if m:
                            mae_vs_baseline = _parse_float(m.group(1))
                    elif "MAE_U_V=" in part or "mae_u_v=" in part.lower():
                        m = re.search(r"MAE_U_V=([\d.eE+-]+)", part, re.I)
                        if m:
                            mae_u_v = _parse_float(m.group(1))
                    elif "runtime_sec=" in part or "runtime=" in part.lower():
                        m = re.search(r"runtime[_]?sec?=([\d.eE+-]+)", part, re.I)
                        if m:
                            runtime_sec = _parse_float(m.group(1))
            elif "Sanity:" in line:
                rest = line.split("Sanity:", 1)[-1].strip()
                m = re.search(r"OUTLIERS=(PASS|FAIL)", rest, re.I)
                if m:
                    outliers = m.group(1).upper()
                m = re.search(r"VARIANCE=(PASS|FAIL)", rest, re.I)
                if m:
                    variance = m.group(1).upper()
                m = re.search(r"SPATIAL_GLOBAL=(PASS|FAIL)", rest, re.I)
                if m:
                    spatial_global = m.group(1).upper()

        if method_id or method or script:
            results.append(
                Result(
                    method_id=method_id or "unknown",
                    method=method,
                    script=script,
                    output_csv=output_csv,
                    mae_vs_baseline=mae_vs_baseline,
                    mae_u_v=mae_u_v,
                    runtime_sec=runtime_sec,
                    outliers=outliers,
                    variance=variance,
                    spatial_global=spatial_global,
                    raw=block,
                )
            )
    return results


def _best_result(results: list[Result]) -> Result | None:
    if not results:
        return None
    best = results[0]
    for r in results[1:]:
        if r.sanity_pass_count > best.sanity_pass_count:
            best = r
        elif r.sanity_pass_count == best.sanity_pass_count:
            a = r.mae_vs_baseline if r.mae_vs_baseline is not None else float("inf")
            b = best.mae_vs_baseline if best.mae_vs_baseline is not None else float("inf")
            if a < b:
                best = r
    return best


def _early_stop(
    results: list[Result],
    new_window: int,
    worst_fraction: float,
) -> bool:
    """True if 90%+ of last `new_window` completed impls are worse than best overall."""
    if len(results) < new_window:
        return False
    best = _best_result(results)
    if not best:
        return False
    window = results[-new_window:]
    worse = sum(1 for r in window if r.is_worse_than(best))
    return (worse / len(window)) >= worst_fraction


def _write_announcement(
    path: Path,
    stop: bool,
    best: Result | None,
    reason: str,
    elapsed_sec: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "=== CONTROLLER ANNOUNCEMENT ===",
        f"Updated: {now}",
        f"Elapsed: {elapsed_sec:.0f}s",
        "",
        "STOP" if stop else "KEEP GOING",
        "",
        f"Reason: {reason}",
        "",
    ]
    if stop:
        lines.append("All agents: STOP working. Do not start new methods.")
        lines.append("")
    if best:
        lines.append("--- Steer toward best implementation ---")
        lines.append(f"  Method ID: {best.method_id}")
        lines.append(f"  Method: {best.method}")
        lines.append(f"  Script: {best.script}")
        lines.append(f"  Sanity: OUTLIERS={best.outliers}, VARIANCE={best.variance}, SPATIAL_GLOBAL={best.spatial_global}")
        if best.mae_vs_baseline is not None:
            lines.append(f"  MAE_vs_baseline: {best.mae_vs_baseline}")
        if best.mae_u_v is not None:
            lines.append(f"  MAE_U_V: {best.mae_u_v}")
        lines.append("")
        lines.append("Prefer similar hyperparameters or extensions of this approach.")
        lines.append("")
    lines.append("=== END ===")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Controller: announce STOP after 1h, steer to best, early-stop when 90%+ new impls worse.",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SEC,
        help=f"Max run duration in seconds (default: {DEFAULT_DURATION_SEC}).",
    )
    ap.add_argument(
        "--check-interval",
        type=float,
        default=DEFAULT_CHECK_INTERVAL_SEC,
        help=f"Seconds between checks (default: {DEFAULT_CHECK_INTERVAL_SEC}).",
    )
    ap.add_argument(
        "--new-window",
        type=int,
        default=DEFAULT_NEW_WINDOW,
        help=f"Last N completed impls for early-stop (default: {DEFAULT_NEW_WINDOW}).",
    )
    ap.add_argument(
        "--worst-fraction",
        type=float,
        default=DEFAULT_WORST_FRACTION,
        help=f"Early-stop when this fraction of new impls are worse (default: {DEFAULT_WORST_FRACTION}).",
    )
    ap.add_argument(
        "--coordination",
        type=str,
        default=COORDINATION_FILE,
        help=f"Path to coordination file (default: {COORDINATION_FILE}).",
    )
    ap.add_argument(
        "--announcements",
        type=str,
        default=ANNOUNCEMENTS_FILE,
        help=f"Path to announcements file (default: {ANNOUNCEMENTS_FILE}).",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    coord_path = root / args.coordination
    announce_path = root / args.announcements

    start = time.perf_counter()
    print(f"Controller started. Duration={args.duration}s, check_interval={args.check_interval}s.")
    print(f"Coordination: {coord_path}")
    print(f"Announcements: {announce_path}")
    print("")

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= args.duration:
            reason = "Time limit reached (1 hour)."
            best = None
            if coord_path.exists():
                content = coord_path.read_text(encoding="utf-8")
                results = _parse_results_blocks(content)
                best = _best_result(results)
            _write_announcement(announce_path, stop=True, best=best, reason=reason, elapsed_sec=elapsed)
            print(f"[{elapsed:.0f}s] STOP. {reason}")
            if best:
                print(f"  Best: {best.method_id} | {best.script}")
            break

        if coord_path.exists():
            content = coord_path.read_text(encoding="utf-8")
            results = _parse_results_blocks(content)
            best = _best_result(results)

            if _early_stop(results, args.new_window, args.worst_fraction):
                reason = (
                    f"Early stop: {int(args.worst_fraction * 100)}%+ of last {args.new_window} "
                    "implementations are worse than best."
                )
                _write_announcement(announce_path, stop=True, best=best, reason=reason, elapsed_sec=elapsed)
                print(f"[{elapsed:.0f}s] STOP (early). {reason}")
                if best:
                    print(f"  Best: {best.method_id} | {best.script}")
                break

            reason = f"Time remaining: {max(0, args.duration - elapsed):.0f}s."
            _write_announcement(announce_path, stop=False, best=best, reason=reason, elapsed_sec=elapsed)
            if best:
                print(f"[{elapsed:.0f}s] KEEP GOING. Best: {best.method_id} | sanity={best.sanity_pass_count}/3")
            else:
                print(f"[{elapsed:.0f}s] KEEP GOING. No RESULTS yet.")
        else:
            reason = f"Coordination file not found. Time remaining: {max(0, args.duration - elapsed):.0f}s."
            _write_announcement(announce_path, stop=False, best=None, reason=reason, elapsed_sec=elapsed)
            print(f"[{elapsed:.0f}s] KEEP GOING. {reason}")

        time.sleep(args.check_interval)

    print("Controller finished.")


if __name__ == "__main__":
    main()

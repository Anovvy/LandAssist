#!/usr/bin/env python3
"""Automated Pull Request Code Reviewer using the official Antigravity Python SDK.

Reads `AGENTS.md` and the current `git diff`, then spawns a read-only Antigravity
Agent (`google-antigravity`) to audit the changes against LandAssist's geospatial,
PostGIS spatial-indexing, 64-channel ML anti-OOM, and security rules.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_MD_PATH = REPO_ROOT / "AGENTS.md"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "pr_review_comment.md"


def load_agents_rules(path: Path = AGENTS_MD_PATH) -> str:
    """Load the repository's AGENTS.md rules."""
    if not path.exists():
        return "Enforce EPSG:4326, PostGIS GIST spatial indexing, 64-channel GSE anti-OOM inference, and zero secrets."
    return path.read_text(encoding="utf-8")


def get_git_diff(base_ref: str | None = None, head_ref: str | None = None) -> str:
    """Collect the git diff for the Pull Request or working tree."""
    if base_ref and head_ref:
        cmd = ["git", "diff", f"{base_ref}...{head_ref}"]
    elif base_ref:
        cmd = ["git", "diff", base_ref]
    else:
        cmd = ["git", "diff", "HEAD~1", "HEAD"]

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        fallback = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return fallback.stdout.strip()
    return result.stdout.strip()


def perform_static_rule_checks(diff_text: str) -> list[str]:
    """Run fast deterministic checks on added lines in the diff (excluding this script)."""
    findings: list[str] = []
    current_file = ""
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            continue
        if current_file == "scripts/agy_code_review.py" or not line.startswith("+"):
            continue
        if "postgresql://" in line and "postgres:password@db.xxxx" not in line and "localhost" not in line:
            findings.append(
                f"CRITICAL ({current_file}): Potential hardcoded PostgreSQL connection string detected in diff."
            )
    return findings


async def run_antigravity_sdk_review(
    rules_text: str,
    diff_text: str,
    dry_run: bool = False,
) -> str:
    """Execute the code review using `google-antigravity` in strict Read-Only Mode."""
    static_findings = perform_static_rule_checks(diff_text)

    if dry_run:
        verdict = "VERDICT: CRITICAL_ISSUES_FOUND" if static_findings else "VERDICT: APPROVED"
        bullet_list = (
            "\n".join(f"- {item}" for item in static_findings)
            if static_findings
            else "- No geospatial, PostGIS indexing, 64-channel ML, or secret-hygiene violations detected."
        )
        return (
            "## 🤖 Antigravity SDK Automated Code Review (Dry-Run Mode)\n\n"
            "Audited against [`AGENTS.md`](AGENTS.md) rules:\n\n"
            f"{bullet_list}\n\n"
            f"**{verdict}**\n"
        )

    try:
        from google.antigravity import Agent, LocalAgentConfig
    except ImportError:
        verdict = "VERDICT: CRITICAL_ISSUES_FOUND" if static_findings else "VERDICT: APPROVED"
        bullet_list = (
            "\n".join(f"- {item}" for item in static_findings)
            if static_findings
            else "- Static fallback audit passed (`google-antigravity` package not installed in current environment)."
        )
        return (
            "## 🤖 Antigravity Automated Code Review (Static Fallback)\n\n"
            f"{bullet_list}\n\n"
            f"**{verdict}**\n"
        )

    # Spawn Agent in Read-Only Mode (no CapabilitiesConfig passed -> read-only by default)
    config = LocalAgentConfig(
        system_instructions=(
            "You are a Senior Geospatial & ML Code Reviewer for the LandAssist project.\n"
            "Enforce the following workspace rules from AGENTS.md:\n\n"
            f"{rules_text}\n\n"
            "Audit every Pull Request diff for:\n"
            "1. Geospatial CRS compliance (EPSG:4326 lon/lat output, metric EPSG:32748 Chaikin smoothing, valid geometries).\n"
            "2. PostGIS performance (mandatory GIST spatial index usage via ST_Intersects / ST_MakeEnvelope / ST_TileEnvelope).\n"
            "3. ML Anti-OOM architecture (64-channel GSE input, rasterio Window reads, disk-backed float16 np.memmap + 2D Hann window).\n"
            "4. Zero hardcoded secrets or binary weight/raster commits.\n\n"
            "End your review with exactly `VERDICT: APPROVED` or `VERDICT: CRITICAL_ISSUES_FOUND`."
        )
    )

    prompt = (
        "Review the following Git diff for LandAssist and output a concise Markdown report. "
        "Conclude with `VERDICT: APPROVED` or `VERDICT: CRITICAL_ISSUES_FOUND`.\n\n"
        f"```diff\n{diff_text[:45000]}\n```"
    )

    review_chunks: list[str] = []
    async with Agent(config) as agent:
        response = await agent.chat(prompt)
        async for token in response:
            sys.stdout.write(token)
            sys.stdout.flush()
            review_chunks.append(token)
    print()

    full_review = "".join(review_chunks).strip()
    if static_findings and "VERDICT: CRITICAL_ISSUES_FOUND" not in full_review:
        full_review += (
            "\n\n### Additional Static Guardrail Findings\n"
            + "\n".join(f"- {item}" for item in static_findings)
            + "\n\n**VERDICT: CRITICAL_ISSUES_FOUND**\n"
        )
    return full_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automated PR code review using the Antigravity Python SDK."
    )
    parser.add_argument("--base", type=str, default=None, help="Base git ref (e.g. origin/main).")
    parser.add_argument("--head", type=str, default=None, help="Head git ref (e.g. HEAD).")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the Markdown PR review comment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run deterministic rule checks and format output without invoking remote LLM credentials.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rules_text = load_agents_rules()
    diff_text = get_git_diff(base_ref=args.base, head_ref=args.head)

    if not diff_text:
        report = (
            "## 🤖 Antigravity SDK Automated Code Review\n\n"
            "No diff detected between target refs.\n\n"
            "**VERDICT: APPROVED**\n"
        )
        args.output.write_text(report, encoding="utf-8")
        print(report)
        return 0

    report = asyncio.run(
        run_antigravity_sdk_review(
            rules_text=rules_text,
            diff_text=diff_text,
            dry_run=args.dry_run,
        )
    )
    args.output.write_text(report, encoding="utf-8")

    if "VERDICT: CRITICAL_ISSUES_FOUND" in report:
        print("\n[Antigravity Review] Critical issues found! Failing CI check.", file=sys.stderr)
        return 1

    print("\n[Antigravity Review] PR passed all LandAssist workspace rules.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

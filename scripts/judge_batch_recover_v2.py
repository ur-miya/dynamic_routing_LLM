#!/usr/bin/env python3
"""Continue an existing recovery-v1 run with separate failure circuits."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import textwrap

RECOVERY_SOURCE = Path("scripts/judge_batch_recover_v1.py")
RECOVERY_SHA = (
    "b435655d0133f9656031aa67bfdf2ae5570b8d1d44735455039a4f6c228e89ac"
)


class CompatibilityError(Exception):
    pass


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_block(source, before, after):
    before = textwrap.dedent(before).strip("\n")
    after = textwrap.dedent(after).strip("\n")
    lines = source.splitlines(keepends=True)
    size = len(before.splitlines())
    matches = []

    for i in range(len(lines) - size + 1):
        block = "".join(lines[i:i + size])
        if textwrap.dedent(block).strip("\n") == before:
            matches.append(i)

    if len(matches) != 1:
        raise CompatibilityError(
            "Patch does not match the supplied recovery source exactly"
        )

    i = matches[0]
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    replacement = textwrap.indent(after + "\n", indent)
    return "".join(lines[:i]) + replacement + "".join(lines[i + size:])


def patched_run(module):
    source = inspect.getsource(module.run)

    source = replace_block(
        source,
        "attempted = handled = errors = hits = streak = 0",
        """
        attempted = handled = errors = hits = streak = 0
        service_streak = 0
        validation_task_streak = 0
        """,
    )

    source = replace_block(
        source,
        "hits += 1",
        """
        service_streak = 0
        validation_task_streak = 0
        hits += 1
        """,
    )

    source = replace_block(
        source,
        "streak = 0",
        """
        streak = 0
        service_streak = 0
        validation_task_streak = 0
        """,
    )

    source = replace_block(
        source,
        """
        errors += 1
        streak += 1
        last_error = code
        if fatal or streak >= CIRCUIT_BREAKER:
            fatal = True
            stop_reason = 'non_retryable_http_or_runtime_error' if streak < CIRCUIT_BREAKER else 'consecutive_error_circuit'
            break
        if index + 1 >= per_task:
            terminal(db, tid, index + 1, code)
            break
        """,
        """
        errors += 1
        last_error = code

        validation_failure = (
            stage == 'pilot_parser'
            and code != 'unexpected_client_or_runtime_error'
        )

        if validation_failure:
            service_streak = 0
        else:
            service_streak += 1

        if fatal:
            stop_reason = 'non_retryable_http_or_runtime_error'
            break

        exhausted = index + 1 >= per_task
        if exhausted:
            terminal(db, tid, index + 1, code)
            if validation_failure:
                validation_task_streak += 1
            else:
                validation_task_streak = 0

        if service_streak >= CIRCUIT_BREAKER:
            fatal = True
            stop_reason = 'consecutive_service_errors'
            break

        if validation_task_streak >= CIRCUIT_BREAKER:
            fatal = True
            stop_reason = 'consecutive_terminal_validation_tasks'
            break

        if exhausted:
            break
        """,
    )

    namespace = dict(vars(module))
    exec(
        compile(source, "<recovery-v2-run>", "exec"),
        namespace,
    )
    return namespace["run"]


def register_policy(module, args):
    b, cfg, pilot, pc, root, plan = module.open_context(
        args.config, args.batch_source
    )

    with b.locked(root):
        previous = module.manifest_check(b, root, plan)
        if previous is None:
            raise CompatibilityError(
                "Existing recovery-v1 manifest is required"
            )

        policy = {
            "version": "recovery-circuit-policy-2",
            "wrapper_sha256": file_sha(__file__),
            "recovery_v1_source_sha256": RECOVERY_SHA,
            "original_plan_sha256": b.sha(root / "plan.json"),
            "recovery_v1_manifest_sha256": b.sha(
                root / "recovery_v1" / "manifest.json"
            ),
            "inputs_sha256": plan["inputs_sha256"],
            "validation_circuit": "consecutive_new_terminal_tasks",
            "validation_task_limit": module.CIRCUIT_BREAKER,
            "service_circuit": "consecutive_failed_attempts",
            "service_attempt_limit": module.CIRCUIT_BREAKER,
            "reset_on_success_or_cache_hit": True,
            "attempt_allowances_unchanged": True,
            "payload_parser_and_labeling_unchanged": True,
        }

        path = root / "recovery_v1" / "circuit_policy_v2.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved != policy:
                raise CompatibilityError(
                    "Existing v2 policy differs; do not edit manifests"
                )
        else:
            b.write_json(path, policy)


def main():
    if not RECOVERY_SOURCE.is_file():
        raise CompatibilityError("Recovery-v1 source was not found")
    if file_sha(RECOVERY_SOURCE) != RECOVERY_SHA:
        raise CompatibilityError(
            "Recovery-v1 source differs from the supplied file"
        )

    spec = importlib.util.spec_from_file_location(
        "verified_recovery_v1_for_v2", RECOVERY_SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Build the replacement before changing any batch state or calling API.
    replacement = patched_run(module)

    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        return module.main()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/judge_batch_v1.yaml")
    parser.add_argument(
        "--batch-source", default="scripts/judge_batch_v1.py"
    )
    args, _ = parser.parse_known_args()

    register_policy(module, args)
    module.run = replacement
    return module.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CompatibilityError as exc:
        print(f"CompatibilityError: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"{type(exc).__name__}: recovery-v2 setup stopped; "
            "no request/response details printed. "
            "Do not modify frozen files.",
            file=sys.stderr,
        )
        raise SystemExit(2)
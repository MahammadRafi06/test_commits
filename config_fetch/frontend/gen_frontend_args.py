"""Generate Dynamo frontend args/envs JSON from a runtime image.

Run this inside an NVIDIA Dynamo runtime image. The script captures the real
``python -m dynamo.frontend --help`` output and feeds it through the shared
frontend parser in ``Frontend/frontend_config.py``.

Usage:
    python frontend/gen_frontend_args.py \
        --out /out/frontend_args.json \
        --env-out /out/frontend_envs.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "Frontend"
sys.path.insert(0, str(FRONTEND_DIR))

try:
    from frontend_config import (
        FLAG_RE,
        generate_config_from_help,
        parse_help_blocks,
        validate_config,
    )
except ImportError as exc:
    sys.exit(f"Failed to import Frontend/frontend_config.py: {exc}")


META_KEYS = {"engine", "version", "date", "source"}
CONFIG_CLASS = {
    "frontend": "Dynamo Frontend Options",
    "kv_router": "KV Router Options",
    "aic": "AIC Perf Model Options",
}
UI_PRIMARY = {
    "router_mode",
    "router_min_initial_workers",
    "serve_indexer",
    "enforce_disagg",
    "router_kv_events",
    "router_prefill_load_model",
    "dyn_chat_processor",
}
UI_ADVANCED = {
    "request_plane",
    "event_plane",
    "enable_anthropic_api",
    "strip_anthropic_preamble",
    "enable_streaming_tool_dispatch",
    "enable_streaming_reasoning_dispatch",
    "exclude_tools_when_tool_choice_none",
    "dyn_debug_perf",
    "dyn_preprocess_workers",
    "kv_cache_block_size",
    "migration_limit",
    "migration_max_seq_len",
    "active_decode_blocks_threshold",
    "active_prefill_tokens_threshold",
    "active_prefill_tokens_threshold_frac",
    "router_replica_sync",
    "router_track_active_blocks",
    "router_track_output_blocks",
    "router_assume_kv_reuse",
    "router_track_prefill_tokens",
    "router_queue_policy",
    "router_event_threads",
    "router_kv_overlap_score_weight",
    "router_temperature",
}


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _capture_help(module: str) -> str:
    proc = _run_module(module, "--help")
    if proc.returncode != 0:
        raise SystemExit(
            f"Failed to capture {module} --help (exit {proc.returncode}):\n{proc.stdout}"
        )
    return proc.stdout


def _version_from_command(module: str) -> tuple[str | None, str | None]:
    proc = _run_module(module, "--version")
    output = proc.stdout.strip()
    if proc.returncode != 0 or not output:
        return None, output or None

    match = re.search(r"\b\d+(?:\.\d+)+(?:[A-Za-z0-9_.+-]*)?\b", output)
    if match:
        return match.group(0), output
    return None, output


def _version_from_imports() -> str | None:
    try:
        import dynamo  # type: ignore[import-not-found]

        version = getattr(dynamo, "__version__", None)
        if version:
            return str(version)
    except Exception:
        pass

    for dist_name in (
        "ai-dynamo",
        "nvidia-dynamo",
        "dynamo",
    ):
        try:
            return metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _count_nested_settings(config: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    total = 0
    for container, buckets in config.items():
        count = 0
        for settings in buckets.values():
            count += len(settings)
        counts[container] = count
        total += count
    counts["total"] = total
    return counts


def _description_from_block(block: list[str]) -> str:
    lines: list[str] = []
    for line in block[1:]:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("env var:"):
            continue
        if lower.startswith("deprecating flag:"):
            continue
        lines.append(stripped.replace("[EXPERIMENTAL]", "").strip())
    return "\n".join(line for line in lines if line).strip()


def _descriptions_by_flag(help_text: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for _, block in parse_help_blocks(help_text):
        description = _description_from_block(block)
        for flag in FLAG_RE.findall(block[0]):
            descriptions[flag] = description
    return descriptions


def _candidate_flags(setting: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for key in ("arg", "true_arg", "false_arg"):
        value = setting.get(key)
        if value:
            flags.append(value)
    for choice in setting.get("choices", []):
        if isinstance(choice, dict) and choice.get("arg"):
            flags.append(choice["arg"])
    return flags


def _description_for_setting(
    setting: dict[str, Any],
    descriptions_by_flag: dict[str, str],
) -> str:
    for flag in _candidate_flags(setting):
        if flag in descriptions_by_flag:
            return descriptions_by_flag[flag]
    return ""


def _parse_number(value: Any) -> int | float | None:
    if not isinstance(value, str) or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return None


def _infer_type(name: str, setting: dict[str, Any], description: str) -> str:
    if setting.get("type") in {"boolean_flag", "flag"}:
        return "bool"
    if setting.get("type") in {"choice", "exclusive_flag"}:
        choices = setting.get("choices") or []
        raw_choices = [
            choice.get("value") if isinstance(choice, dict) else choice
            for choice in choices
        ]
        if raw_choices and all(str(choice).lstrip("-").isdigit() for choice in raw_choices):
            return "int"
        return "str"

    default_value = setting.get("default_value")
    if isinstance(default_value, bool):
        return "bool"
    if isinstance(default_value, int):
        return "int"
    if isinstance(default_value, float):
        return "float"

    parsed_default = _parse_number(default_value)
    if isinstance(parsed_default, int):
        return "int"
    if isinstance(parsed_default, float):
        return "float"

    text = f"{name} {description}".lower()
    if any(token in text for token in ("u16", "u32", "usize", "integer", "token count")):
        return "int"
    if any(token in text for token in ("fraction", "ratio", "float", "0.0-1.0")):
        return "float"
    return "str"


def _normalise_default(value: Any, type_label: str) -> Any:
    if value == "":
        if type_label == "bool":
            return False
        return None
    if type_label == "int":
        parsed = _parse_number(value)
        return int(parsed) if parsed is not None else value
    if type_label == "float":
        parsed = _parse_number(value)
        return float(parsed) if parsed is not None else value
    return value


def _ui_tier(name: str, bucket: str) -> str:
    if name in UI_PRIMARY:
        return "primary"
    if bucket == "exp" or name in UI_ADVANCED:
        return "advanced"
    return "less_frequent"


def _arg_record(
    name: str,
    container: str,
    bucket: str,
    setting: dict[str, Any],
    descriptions_by_flag: dict[str, str],
) -> OrderedDict[str, Any]:
    description = _description_for_setting(setting, descriptions_by_flag)
    type_label = _infer_type(name, setting, description)
    record: OrderedDict[str, Any] = OrderedDict()
    record["type"] = type_label
    record["flag"] = bool(setting.get("flag"))

    if setting.get("flag"):
        if setting.get("true_arg"):
            record["true_arg"] = setting["true_arg"]
        else:
            record["true_arg"] = None
        if setting.get("false_arg"):
            record["false_arg"] = setting["false_arg"]
        else:
            record["false_arg"] = None
        if setting.get("choices"):
            record["choices"] = setting["choices"]
    else:
        record["arg"] = setting.get("arg")

    record["default_value"] = _normalise_default(setting.get("default_value"), type_label)
    record["value"] = None
    record["description"] = description
    record["env_var"] = setting.get("env_var")
    record["module"] = container
    record["config_class"] = CONFIG_CLASS.get(container, container)
    record["ui"] = _ui_tier(name, bucket)
    record["aic"] = container == "aic"

    if setting.get("type") == "choice":
        record["choices"] = setting.get("choices", [])
    if setting.get("type") == "exclusive_flag" and setting.get("choices"):
        record["choices"] = setting["choices"]
    if bucket == "exp":
        record["status"] = "experimental"

    return record


def build_args_json(
    nested_config: dict[str, Any],
    help_text: str,
    version: str,
    source: str,
) -> OrderedDict[str, Any]:
    descriptions_by_flag = _descriptions_by_flag(help_text)
    out: OrderedDict[str, Any] = OrderedDict(
        [
            ("engine", "frontend"),
            ("version", version),
            ("date", date.today().isoformat()),
            ("source", source),
        ]
    )

    for container, buckets in nested_config.items():
        for bucket, settings in buckets.items():
            for name, setting in settings.items():
                out[name] = _arg_record(
                    name,
                    container,
                    bucket,
                    setting,
                    descriptions_by_flag,
                )
    return out


def build_envs_json(args_json: OrderedDict[str, Any]) -> OrderedDict[str, Any]:
    out: OrderedDict[str, Any] = OrderedDict(
        [
            ("engine", "frontend"),
            ("version", args_json["version"]),
            ("date", args_json["date"]),
            ("source", f"{args_json['source']} env vars"),
        ]
    )

    for key, rec in args_json.items():
        if key in META_KEYS:
            continue
        env_var = rec.get("env_var")
        if not env_var:
            continue

        env_rec: OrderedDict[str, Any] = OrderedDict()
        env_rec["category"] = rec.get("module", "frontend")
        env_rec["type"] = rec.get("type", "str")
        env_rec["default_value"] = rec.get("default_value")
        env_rec["value"] = None
        env_rec["description"] = rec.get("description", "")
        if rec.get("flag"):
            env_rec["true_arg"] = rec.get("true_arg")
            env_rec["false_arg"] = rec.get("false_arg")
        else:
            env_rec["arg"] = rec.get("arg")
        env_rec["arg_key"] = key
        env_rec["ui"] = rec.get("ui", "less_frequent")
        env_rec["aic"] = rec.get("aic", False)

        for optional_key in ("choices", "status"):
            if optional_key in rec:
                env_rec[optional_key] = rec[optional_key]

        out[env_var] = env_rec

    return out


def _check_no_emit(path: str, data: OrderedDict[str, Any]) -> None:
    for key, rec in data.items():
        if key in META_KEYS:
            continue
        if "emit" in rec:
            raise SystemExit(f"{path}: record {key!r} contains unsupported emit field")


def validate_engine_style_args(path: str, data: OrderedDict[str, Any]) -> None:
    required = {
        "type",
        "flag",
        "default_value",
        "value",
        "description",
        "module",
        "config_class",
        "ui",
        "aic",
    }
    for meta_key in META_KEYS:
        if meta_key not in data:
            raise SystemExit(f"{path}: missing metadata field {meta_key}")

    _check_no_emit(path, data)
    bad: list[str] = []
    for key, rec in data.items():
        if key in META_KEYS:
            continue
        missing = required - set(rec)
        if missing:
            bad.append(f"{key}: missing {sorted(missing)}")
            continue
        if not isinstance(rec["flag"], bool):
            bad.append(f"{key}: flag must be boolean")
            continue
        if rec["flag"]:
            if not (rec.get("true_arg") or rec.get("false_arg") or rec.get("choices")):
                bad.append(f"{key}: flag records need true_arg, false_arg, or choices")
        elif not rec.get("arg"):
            bad.append(f"{key}: value records need arg")

    if bad:
        raise SystemExit(f"{path}: malformed args records: {bad[:10]}")


def validate_engine_style_envs(path: str, data: OrderedDict[str, Any]) -> None:
    required = {
        "category",
        "type",
        "default_value",
        "value",
        "description",
        "ui",
        "aic",
    }
    for meta_key in META_KEYS:
        if meta_key not in data:
            raise SystemExit(f"{path}: missing metadata field {meta_key}")

    _check_no_emit(path, data)
    bad: list[str] = []
    for key, rec in data.items():
        if key in META_KEYS:
            continue
        missing = required - set(rec)
        if missing:
            bad.append(f"{key}: missing {sorted(missing)}")

    if bad:
        raise SystemExit(f"{path}: malformed env records: {bad[:10]}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def build_config(help_text: str) -> dict[str, Any]:
    config = generate_config_from_help(help_text)
    validate_config(config)
    return config


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--out",
        type=Path,
        default=Path("frontend_args.json"),
        help="Output engine-style args JSON path.",
    )
    cli.add_argument(
        "--env-out",
        type=Path,
        help="Output engine-style envs JSON path. Defaults next to --out.",
    )
    cli.add_argument(
        "--help-out",
        type=Path,
        help="Optional path to write the raw argparse help text.",
    )
    cli.add_argument(
        "--meta-out",
        type=Path,
        help="Optional path to write generator metadata.",
    )
    cli.add_argument(
        "--help-file",
        type=Path,
        help="Use an existing help-text file instead of running dynamo.frontend.",
    )
    cli.add_argument(
        "--module",
        default="dynamo.frontend",
        help="Python module to invoke for help/version capture.",
    )
    ns = cli.parse_args()
    env_out = ns.env_out or ns.out.with_name("frontend_envs.json")

    if ns.help_file:
        help_text = ns.help_file.read_text()
        version, raw_version = None, None
        source = f"parsed from {ns.help_file}"
    else:
        help_text = _capture_help(ns.module)
        version, raw_version = _version_from_command(ns.module)
        source = f"captured from python -m {ns.module} --help"

    version = version or _version_from_imports() or "unknown"
    nested_config = build_config(help_text)
    args_json = build_args_json(nested_config, help_text, version, source)
    envs_json = build_envs_json(args_json)
    validate_engine_style_args(str(ns.out), args_json)
    validate_engine_style_envs(str(env_out), envs_json)

    _write_json(ns.out, args_json)
    _write_json(env_out, envs_json)
    if ns.help_out:
        ns.help_out.parent.mkdir(parents=True, exist_ok=True)
        ns.help_out.write_text(help_text)

    counts = _count_nested_settings(nested_config)
    meta = {
        "component": "frontend",
        "version": version,
        "date": date.today().isoformat(),
        "source": source,
        "raw_version_output": raw_version,
        "settings": counts,
        "args": len([key for key in args_json if key not in META_KEYS]),
        "envs": len([key for key in envs_json if key not in META_KEYS]),
    }
    if ns.meta_out:
        _write_json(ns.meta_out, meta)

    print(
        f"Written {ns.out!s} - Dynamo frontend {version} - "
        f"{counts['total']} total settings"
    )
    for container in ("frontend", "kv_router", "aic"):
        print(f"  {container:10s} {counts.get(container, 0)} settings")


if __name__ == "__main__":
    main()

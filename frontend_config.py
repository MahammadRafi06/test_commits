"""Generate and consume the Dynamo frontend server config.

The input file named ``frontend.json`` is argparse help text, not JSON. This
module turns that help text into the unified ``frontend_args.json`` schema and
also reads that schema back into runtime env/arg arrays.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


CONTAINERS = ("frontend", "kv_router", "aic")
BUCKETS = ("ga", "exp")
SECTION_TO_CONTAINER = {
    "Dynamo Frontend Options": "frontend",
    "KV Router Options": "kv_router",
    "AIC Perf Model Options": "aic",
}
FLAG_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")
AUTO_MANAGED_ENV_VARS = {
    "DYN_NAMESPACE": "Injected by the Dynamo operator from the DGD namespace.",
    "DYN_DISCOVERY_BACKEND": "Injected by the Dynamo operator from discovery configuration.",
    "DYN_HTTP_PORT": "Injected by frontend defaults to match the generated service port.",
    "DYN_NAMESPACE_PREFIX": "Injected by frontend defaults from the Dynamo namespace.",
}


class ConfigError(ValueError):
    """Raised when the frontend config is malformed or has invalid values."""


def setting_name_from_flag(flag: str) -> str:
    """Return the logical setting name for a CLI flag."""

    name = flag.removeprefix("--")
    if name.startswith("no-"):
        name = name.removeprefix("no-")
    return name.replace("-", "_")


def normalize_default(raw: str) -> str | bool:
    """Convert argparse default text into the config representation."""

    raw = raw.strip()
    if raw == "None":
        return ""
    if raw == "True":
        return True
    if raw == "False":
        return False
    return raw


def parse_scalar(raw: str) -> str | int | float | bool | None:
    """Parse a CLI override value used by ``emit --set``."""

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def parse_help_blocks(help_text: str) -> list[tuple[str, list[str]]]:
    """Split argparse help text into one block per documented option."""

    blocks: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    current_container = "frontend"
    option_line = re.compile(r"^\s{2}(?:-[A-Za-z0-9],\s*)?--[A-Za-z0-9]")

    for raw_line in help_text.splitlines():
        line = raw_line.rstrip()
        section = SECTION_TO_CONTAINER.get(line.removesuffix(":"))
        if section:
            if current:
                blocks.append((current_container, current))
                current = None
            current_container = section
            continue

        if option_line.match(line):
            if current:
                blocks.append((current_container, current))
            current = [line]
        elif current is not None and line.strip():
            current.append(line)

    if current:
        blocks.append((current_container, current))

    return blocks


def parse_default(block: list[str]) -> str | bool:
    """Extract the default value from an argparse help block."""

    match = re.search(
        r"env var:\s*[A-Z0-9_]+(?:\s*\|\s*default:\s*(.*))?$",
        "\n".join(block),
        flags=re.M,
    )
    if not match or match.group(1) is None:
        return ""
    return normalize_default(match.group(1))


def parse_env_var(block: list[str]) -> str | None:
    """Extract the env var name from an argparse help block when present."""

    match = re.search(r"env var:\s*([A-Z0-9_]+)", "\n".join(block))
    if not match:
        return None
    return match.group(1)


def parse_choices(header: str) -> list[str] | None:
    """Extract ``{a,b,c}`` argparse choices from an option header."""

    match = re.search(r"\{([^}]+)\}", header)
    if not match:
        return None
    return [part.strip() for part in match.group(1).split(",") if part.strip()]


def apply_management_metadata(item: OrderedDict[str, Any]) -> None:
    """Mark settings whose runtime values are provided by Dynamo itself."""

    reason = AUTO_MANAGED_ENV_VARS.get(item.get("env_var"))
    if not reason:
        return

    item["emit"] = False
    item["managed_by"] = "dynamo"
    item["reason"] = reason


def is_deprecated_block(block: list[str]) -> bool:
    """Return true when the whole option block is deprecated."""

    return any("[Deprecated]" in line for line in block)


def parse_exclusive_groups(help_text: str) -> dict[str, OrderedDict[str, Any]]:
    """Read argparse mutually-exclusive groups from the usage section."""

    groups: dict[str, OrderedDict[str, Any]] = OrderedDict()

    for match in re.finditer(r"\[([^\]]*\|[^\]]*)\]", help_text, flags=re.S):
        flags = FLAG_RE.findall(match.group(1))
        if len(flags) < 2:
            continue

        positives = [flag for flag in flags if not flag.startswith("--no-")]
        negatives = [flag for flag in flags if flag.startswith("--no-")]

        if positives and negatives:
            true_arg = positives[0]
            true_stem = true_arg.removeprefix("--")
            false_arg = next(
                (flag for flag in negatives if flag.removeprefix("--no-") == true_stem),
                negatives[0],
            )
            group_name = setting_name_from_flag(true_arg)
            groups[group_name] = OrderedDict(
                [
                    ("name", group_name),
                    ("type", "boolean_flag"),
                    ("true_arg", true_arg),
                    ("false_arg", false_arg),
                    ("true_aliases", [flag for flag in positives if flag != true_arg]),
                    ("false_aliases", [flag for flag in negatives if flag != false_arg]),
                    ("members", flags),
                ]
            )
        else:
            group_name = setting_name_from_flag(flags[0])
            groups[group_name] = OrderedDict(
                [
                    ("name", group_name),
                    ("type", "exclusive_flag"),
                    (
                        "choices",
                        [
                            OrderedDict(
                                [
                                    ("value", setting_name_from_flag(flag)),
                                    ("arg", flag),
                                ]
                            )
                            for flag in flags
                        ],
                    ),
                    ("members", flags),
                ]
            )

    return groups


def generate_config_from_help(
    help_text: str,
) -> OrderedDict[str, OrderedDict[str, OrderedDict[str, Any]]]:
    """Generate the unified frontend config from argparse help text."""

    exclusive_groups = parse_exclusive_groups(help_text)
    flag_to_group = {
        flag: group_name
        for group_name, group in exclusive_groups.items()
        for flag in group["members"]
    }

    config: OrderedDict[str, OrderedDict[str, OrderedDict[str, Any]]] = OrderedDict(
        (container, OrderedDict((bucket, OrderedDict()) for bucket in BUCKETS))
        for container in CONTAINERS
    )
    created_groups: set[tuple[str, str]] = set()

    for container, block in parse_help_blocks(help_text):
        header = block[0]
        flags = FLAG_RE.findall(header)
        if not flags:
            continue

        bucket = "exp" if any("[EXPERIMENTAL]" in line for line in block) else "ga"
        env_var = parse_env_var(block)
        if env_var is None:
            continue
        default_value = parse_default(block)

        group_names = {flag_to_group[flag] for flag in flags if flag in flag_to_group}
        if group_names:
            if is_deprecated_block(block):
                continue

            group_name = next(iter(group_names))
            group_key = (container, group_name)
            if group_key in created_groups:
                continue
            created_groups.add(group_key)

            group = exclusive_groups[group_name]
            if group["type"] == "boolean_flag":
                item: OrderedDict[str, Any] = OrderedDict(
                    [
                        ("type", "boolean_flag"),
                        ("flag", True),
                        ("env_var", env_var),
                        ("default_value", default_value if isinstance(default_value, bool) else ""),
                        ("value", None),
                        ("true_arg", group["true_arg"]),
                        ("false_arg", group["false_arg"]),
                    ]
                )
            else:
                item = OrderedDict(
                    [
                        ("type", "exclusive_flag"),
                        ("flag", True),
                        ("env_var", env_var),
                        ("default_value", default_value),
                        ("value", None),
                        ("choices", group["choices"]),
                    ]
                )

            apply_management_metadata(item)
            config[container][bucket][group_name] = item
            continue

        canonical_arg = flags[0]
        name = setting_name_from_flag(canonical_arg)
        choices = parse_choices(header)

        if choices:
            item = OrderedDict(
                [
                    ("type", "choice"),
                    ("flag", False),
                    ("arg", canonical_arg),
                    ("env_var", env_var),
                    ("default_value", default_value),
                    ("value", None),
                    ("choices", choices),
                ]
            )
        else:
            item = OrderedDict(
                [
                    ("type", "value"),
                    ("flag", False),
                    ("arg", canonical_arg),
                    ("env_var", env_var),
                    ("default_value", default_value),
                    ("value", None),
                ]
            )

        aliases = flags[1:]
        apply_management_metadata(item)
        if is_deprecated_block(block):
            continue
        config[container][bucket][name] = item

    return config


def load_config(path: str | Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Load and validate a unified frontend config file."""

    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate the config shape and any user-supplied values."""

    if list(config.keys()) != list(CONTAINERS):
        raise ConfigError(
            f"Config must have exactly these top-level keys: {', '.join(CONTAINERS)}"
        )

    for container in CONTAINERS:
        container_config = config[container]
        if not isinstance(container_config, dict):
            raise ConfigError(f"{container} must be an object")
        if list(container_config.keys()) != list(BUCKETS):
            raise ConfigError(f"{container} must have exactly these keys: {', '.join(BUCKETS)}")

        for bucket in BUCKETS:
            settings = container_config[bucket]
            if not isinstance(settings, dict):
                raise ConfigError(f"{container}.{bucket} must be an object of named settings")

            for name, setting in settings.items():
                path = f"{container}.{bucket}.{name}"
                if not isinstance(setting, dict):
                    raise ConfigError(f"{path} must be an object")
                if "env_var" not in setting:
                    raise ConfigError(f"{path} is missing env_var")
                if not isinstance(setting["env_var"], str):
                    raise ConfigError(f"{path}: env_var must be a string")
                if "emit" in setting and not isinstance(setting["emit"], bool):
                    raise ConfigError(f"{path}: emit must be true or false")

                kind = setting.get("type")
                emits = setting.get("emit", True)
                value = setting.get("value") if emits else None

                if kind == "value":
                    if setting.get("flag") is not False:
                        raise ConfigError(f"{path}: value settings must have flag=false")
                    if not setting.get("arg"):
                        raise ConfigError(f"{path}: value settings need arg")
                    if isinstance(value, (bool, dict, list)):
                        raise ConfigError(f"{path}: value must be string, number, or null")
                elif kind == "choice":
                    choices = setting.get("choices")
                    if setting.get("flag") is not False:
                        raise ConfigError(f"{path}: choice settings must have flag=false")
                    if not setting.get("arg"):
                        raise ConfigError(f"{path}: choice settings need arg")
                    if not isinstance(choices, list) or not choices:
                        raise ConfigError(f"{path}: choice settings need a non-empty choices list")
                    if value is not None and value not in choices:
                        raise ConfigError(f"{path}: {value!r} is not one of {choices}")
                elif kind == "boolean_flag":
                    if setting.get("flag") is not True:
                        raise ConfigError(f"{path}: boolean_flag settings must have flag=true")
                    if not setting.get("true_arg") or not setting.get("false_arg"):
                        raise ConfigError(f"{path}: boolean_flag needs true_arg and false_arg")
                    if value is not None and not isinstance(value, bool):
                        raise ConfigError(f"{path}: value must be true, false, or null")
                elif kind == "flag":
                    if setting.get("flag") is not True:
                        raise ConfigError(f"{path}: flag settings must have flag=true")
                    if not setting.get("arg"):
                        raise ConfigError(f"{path}: flag settings need arg")
                    if value is not None and not isinstance(value, bool):
                        raise ConfigError(f"{path}: value must be true, false, or null")
                elif kind == "exclusive_flag":
                    choices = setting.get("choices")
                    if setting.get("flag") is not True:
                        raise ConfigError(f"{path}: exclusive_flag settings must have flag=true")
                    if not isinstance(choices, list) or not choices:
                        raise ConfigError(f"{path}: exclusive_flag needs choices")
                    allowed = [choice["value"] for choice in choices]
                    if value is not None and value not in allowed:
                        raise ConfigError(f"{path}: {value!r} is not one of {allowed}")
                else:
                    raise ConfigError(f"{path}: unsupported type {kind!r}")


def stringify_env_value(value: Any) -> str:
    """Convert a config value to the string expected by Kubernetes env arrays."""

    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_env_and_args(config: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    """Build Kubernetes-style env entries and CLI args from the config."""

    validate_config(config)
    env_array: list[dict[str, str]] = []
    args_array: list[str] = []

    for container in CONTAINERS:
        for bucket in BUCKETS:
            for setting in config[container][bucket].values():
                value = setting["value"]
                if value is None or setting.get("emit", True) is False:
                    continue

                kind = setting["type"]
                if setting["env_var"]:
                    env_array.append({"name": setting["env_var"], "value": stringify_env_value(value)})

                if kind in {"value", "choice"}:
                    args_array.extend([setting["arg"], str(value)])
                elif kind == "boolean_flag":
                    args_array.append(setting["true_arg"] if value else setting["false_arg"])
                elif kind == "flag":
                    if value:
                        args_array.append(setting["arg"])
                elif kind == "exclusive_flag":
                    choice = next(choice for choice in setting["choices"] if choice["value"] == value)
                    args_array.append(choice["arg"])

    return env_array, args_array


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    """Apply ``container.bucket.setting=value`` overrides to a loaded config in place."""

    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"Override must be container.bucket.setting=value: {override}")

        path, raw_value = override.split("=", 1)
        parts = path.split(".")
        if len(parts) != 3:
            raise ConfigError(f"Override path must be container.bucket.setting: {path}")

        container, bucket, setting_name = parts
        if container not in CONTAINERS:
            raise ConfigError(f"Unknown container {container!r}; expected one of {CONTAINERS}")
        if bucket not in BUCKETS:
            raise ConfigError(f"Unknown bucket {bucket!r}; expected one of {BUCKETS}")
        if setting_name not in config[container][bucket]:
            raise ConfigError(f"Unknown setting {container}.{bucket}.{setting_name}")

        config[container][bucket][setting_name]["value"] = parse_scalar(raw_value)

    validate_config(config)


def write_json(path: str | Path, payload: Any) -> None:
    """Write formatted JSON with a trailing newline."""

    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate frontend config JSON from argparse help text."""

    help_text = Path(args.input).read_text()
    config = generate_config_from_help(help_text)
    validate_config(config)
    write_json(args.output, config)


def cmd_emit(args: argparse.Namespace) -> None:
    """Read frontend config JSON and print env/args arrays."""

    config = load_config(args.config)
    apply_overrides(config, args.set)
    env_array, args_array = build_env_and_args(config)
    print(json.dumps({"env_array": env_array, "args_array": args_array}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate frontend_args.json")
    generate.add_argument("--input", default="frontend.json", help="argparse help text file")
    generate.add_argument("--output", default="frontend_args.json", help="config JSON output file")
    generate.set_defaults(func=cmd_generate)

    emit = subparsers.add_parser("emit", help="emit env_array and args_array from config")
    emit.add_argument("--config", default="frontend_args.json", help="unified config JSON file")
    emit.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="CONTAINER.BUCKET.SETTING=VALUE",
        help="temporary override, for example frontend.ga.http_port=9000",
    )
    emit.set_defaults(func=cmd_emit)

    return parser


def main() -> None:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except ConfigError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()

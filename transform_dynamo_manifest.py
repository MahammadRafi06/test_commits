#!/usr/bin/env python3
"""Transform a raw DynamoGraphDeployment into a deployment-ready manifest.

The raw manifest describes the workload. A named policy, loaded from SQLite,
describes deployment choices such as images, PVCs, scheduling, EFA resources,
replicas, GPU sizing, and cache mounts.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import yaml


SPDX_HEADER = """# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""

POLICY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS manifest_policies (
  name TEXT PRIMARY KEY,
  description TEXT,
  policy_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

DEFAULT_POLICY_NAME = "aws-p5-efa-vllm-disagg"

DEFAULT_POLICY: dict[str, Any] = {
    "output": {
        "include_spdx_header": True,
        "metadata": {
            "name": "vllm-disagg",
            "namespace": None,
        },
        "service_order": [
            "Frontend",
            "VllmDecodeWorker",
            "VllmPrefillWorker",
        ],
        "pvcs": [
            {"create": False, "name": "model-cache"},
            {"create": False, "name": "compilation-cache"},
        ],
    },
    "model": {
        "strip_suffixes": ["-FP8"],
    },
    "frontend": {
        "copy_env_from_secret": False,
        "envs": [
            {
                "name": "HF_HOME",
                "value": "/home/dynamo/.cache/huggingface",
            }
        ],
        "replicas": None,
        "resources": {
            "requests": {"cpu": "8"},
            "limits": {"cpu": "8"},
        },
        "volume_mounts": [
            {
                "name": "model-cache",
                "mountPoint": "/home/dynamo/.cache/huggingface",
            }
        ],
        "image": "nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.0.2",
        "working_dir": "/workspace",
        "command": "exec python3 -m dynamo.frontend --router-mode kv --router-reset-states",
        "affinity": {
            "same_zone_as_workers": True,
            "topology_key": "topology.kubernetes.io/zone",
        },
    },
    "workers": {
        "copy_env_from_secret": True,
        "replicas": {
            "decode": 2,
            "prefill": 6,
        },
        "gpu_per_worker": {
            "decode": "2",
            "prefill": "2",
        },
        "efa_per_worker": "8",
        "efa_resource_name": "vpc.amazonaws.com/efa",
        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.2-efa-amd64",
        "working_dir": "/workspace",
        "env": [
            {
                "name": "HF_HOME",
                "value": "/home/dynamo/.cache/huggingface",
            }
        ],
        "security_context": {
            "capabilities": {
                "add": ["IPC_LOCK"],
            }
        },
        "volume_mounts": [
            {
                "name": "model-cache",
                "mountPoint": "/home/dynamo/.cache/huggingface",
            },
            {
                "name": "compilation-cache",
                "mountPoint": "/home/dynamo/.cache/vllm",
                "useAsCompilationCache": True,
            },
        ],
        "node_selector": {
            "node.kubernetes.io/instance-type": "p5.48xlarge",
            "karpenter.sh/capacity-type": "reserved",
        },
        "tolerations": [
            {
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        ],
        "affinity": {
            "same_zone": True,
            "topology_key": "topology.kubernetes.io/zone",
            "anti_affinity_component": "VllmPrefillWorker",
            "anti_affinity_topology_key": "kubernetes.io/hostname",
        },
        "command": {
            "module": "dynamo.vllm",
            "tensor_parallel_size": {
                "decode": "2",
                "prefill": "2",
            },
            "kv_connector_extra_config": {
                "backends": ["LIBFABRIC"],
            },
        },
    },
}


class LiteralString(str):
    """YAML scalar that should be emitted as a literal block."""


class DynamoDumper(yaml.SafeDumper):
    """PyYAML dumper with Kubernetes-friendly list indentation."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> Any:
        return super().increase_indent(flow, False)


def _literal_string_representer(
    dumper: yaml.SafeDumper, data: LiteralString
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


DynamoDumper.add_representer(LiteralString, _literal_string_representer)


def ensure_policy_schema(conn: sqlite3.Connection) -> None:
    conn.execute(POLICY_TABLE_SQL)
    conn.commit()


def seed_default_policy(conn: sqlite3.Connection) -> None:
    upsert_policy(
        conn,
        DEFAULT_POLICY_NAME,
        "AWS p5 EFA policy for vLLM disaggregated Dynamo deployments",
        DEFAULT_POLICY,
    )


def upsert_policy(
    conn: sqlite3.Connection,
    policy_name: str,
    description: str | None,
    policy: dict[str, Any],
) -> None:
    ensure_policy_schema(conn)
    conn.execute(
        """
        INSERT INTO manifest_policies (name, description, policy_json, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(name) DO UPDATE SET
          description = excluded.description,
          policy_json = excluded.policy_json,
          updated_at = CURRENT_TIMESTAMP
        """,
        (
            policy_name,
            description,
            json.dumps(policy, indent=2),
        ),
    )
    conn.commit()


def load_policy(conn: sqlite3.Connection, policy_name: str) -> dict[str, Any]:
    ensure_policy_schema(conn)
    row = conn.execute(
        "SELECT policy_json FROM manifest_policies WHERE name = ?",
        (policy_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Policy {policy_name!r} was not found in the policy database")
    policy = json.loads(row[0])
    if not isinstance(policy, dict):
        raise ValueError(f"Policy {policy_name!r} must contain a JSON object")
    return policy


def list_policies(conn: sqlite3.Connection) -> list[tuple[str, str | None]]:
    ensure_policy_schema(conn)
    rows = conn.execute(
        "SELECT name, description FROM manifest_policies ORDER BY name"
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def show_policy(conn: sqlite3.Connection, policy_name: str) -> dict[str, Any]:
    return load_policy(conn, policy_name)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def read_policy_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a policy object")
    return data


def write_yaml(path: Path, manifest: dict[str, Any], include_spdx_header: bool) -> None:
    text = yaml.dump(
        manifest,
        Dumper=DynamoDumper,
        sort_keys=False,
        default_flow_style=False,
        width=1000,
    )
    if include_spdx_header:
        text = SPDX_HEADER + text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def nullish(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.lower() in {"", "none", "null", "~"})


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_cli_args(args: list[Any] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    if not args:
        return parsed

    tokens = [str(token) for token in args]
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
                parsed[token] = tokens[index + 1]
                index += 2
            else:
                parsed[token] = True
                index += 1
        else:
            index += 1
    return parsed


def command_module(command: list[Any] | None, fallback: str) -> str:
    if not command:
        return fallback
    tokens = [str(token) for token in command]
    for index, token in enumerate(tokens):
        if token == "-m" and index + 1 < len(tokens):
            return tokens[index + 1]
    return fallback


def apply_model_policy(model: str, policy: dict[str, Any]) -> str:
    model_policy = policy.get("model", {})
    for suffix in model_policy.get("strip_suffixes", []):
        if suffix and model.endswith(str(suffix)):
            return model[: -len(str(suffix))]
    return model


def component_role(service_name: str, service: dict[str, Any]) -> str:
    role = service.get("subComponentType")
    if role:
        return str(role)

    lowered = service_name.lower()
    if "decode" in lowered:
        return "decode"
    if "prefill" in lowered:
        return "prefill"
    return "worker"


def model_from_service(service: dict[str, Any], policy: dict[str, Any]) -> str | None:
    main_container = service.get("extraPodSpec", {}).get("mainContainer", {})
    args = parse_cli_args(main_container.get("args"))
    model = args.get("--model")
    if model is None:
        return None
    return apply_model_policy(str(model), policy)


def first_model_from_services(services: dict[str, Any], policy: dict[str, Any]) -> str | None:
    for service in services.values():
        if not isinstance(service, dict):
            continue
        model = model_from_service(service, policy)
        if model:
            return model
    return None


def copy_if_present(target: dict[str, Any], key: str, source: dict[str, Any]) -> None:
    if key in source:
        target[key] = copy.deepcopy(source[key])


def get_service_order(services: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    requested = policy.get("output", {}).get("service_order", [])
    ordered = [name for name in requested if name in services]
    ordered.extend(name for name in services if name not in ordered)
    return ordered


def build_frontend_affinity(deployment_name: str, frontend_policy: dict[str, Any]) -> dict[str, Any]:
    topology_key = frontend_policy.get("affinity", {}).get(
        "topology_key", "topology.kubernetes.io/zone"
    )
    return {
        "podAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "topologyKey": topology_key,
                    "labelSelector": {
                        "matchLabels": {
                            "nvidia.com/dynamo-graph-deployment-name": deployment_name,
                            "nvidia.com/dynamo-component-type": "worker",
                        }
                    },
                }
            ]
        }
    }


def build_worker_affinity(deployment_name: str, workers_policy: dict[str, Any]) -> dict[str, Any]:
    affinity_policy = workers_policy.get("affinity", {})
    affinity: dict[str, Any] = {}

    if affinity_policy.get("same_zone", False):
        affinity["podAffinity"] = {
            "requiredDuringSchedulingIgnoredDuringExecution": [
                {
                    "topologyKey": affinity_policy.get(
                        "topology_key", "topology.kubernetes.io/zone"
                    ),
                    "labelSelector": {
                        "matchLabels": {
                            "nvidia.com/dynamo-graph-deployment-name": deployment_name,
                        }
                    },
                }
            ]
        }

    anti_component = affinity_policy.get("anti_affinity_component")
    if anti_component:
        affinity["podAntiAffinity"] = {
            "preferredDuringSchedulingIgnoredDuringExecution": [
                {
                    "weight": 100,
                    "podAffinityTerm": {
                        "labelSelector": {
                            "matchLabels": {
                                "nvidia.com/dynamo-component": anti_component,
                            }
                        },
                        "topologyKey": affinity_policy.get(
                            "anti_affinity_topology_key", "kubernetes.io/hostname"
                        ),
                    },
                }
            ]
        }

    return affinity


def worker_resources(role: str, workers_policy: dict[str, Any], raw_service: dict[str, Any]) -> dict[str, Any]:
    resources_by_role = workers_policy.get("resources_by_role", {})
    if role in resources_by_role:
        return copy.deepcopy(resources_by_role[role])

    if "resources" in workers_policy:
        return copy.deepcopy(workers_policy["resources"])

    raw_limits = raw_service.get("resources", {}).get("limits", {})
    gpu = workers_policy.get("gpu_per_worker", {}).get(role) or raw_limits.get("gpu")
    if gpu is None:
        return copy.deepcopy(raw_service.get("resources", {}))

    requests: dict[str, Any] = {"gpu": str(gpu)}
    limits: dict[str, Any] = {"gpu": str(gpu)}

    efa = workers_policy.get("efa_per_worker")
    efa_resource_name = workers_policy.get("efa_resource_name")
    if efa is not None and efa_resource_name:
        custom = {str(efa_resource_name): str(efa)}
        requests["custom"] = copy.deepcopy(custom)
        limits["custom"] = copy.deepcopy(custom)

    return {
        "requests": requests,
        "limits": limits,
    }


def build_command_block(lines: list[str]) -> LiteralString:
    return LiteralString("\n".join(lines) + "\n")


def build_frontend_service(
    raw_service: dict[str, Any],
    deployment_name: str,
    policy: dict[str, Any],
) -> dict[str, Any]:
    frontend_policy = policy.get("frontend", {})
    raw_main = raw_service.get("extraPodSpec", {}).get("mainContainer", {})

    service: dict[str, Any] = {}
    if frontend_policy.get("copy_env_from_secret", False):
        copy_if_present(service, "envFromSecret", raw_service)

    service["componentType"] = raw_service.get("componentType", "frontend")

    if frontend_policy.get("envs"):
        service["envs"] = copy.deepcopy(frontend_policy["envs"])

    service["replicas"] = (
        frontend_policy["replicas"]
        if frontend_policy.get("replicas") is not None
        else raw_service.get("replicas", 1)
    )

    if frontend_policy.get("resources"):
        service["resources"] = copy.deepcopy(frontend_policy["resources"])

    if frontend_policy.get("volume_mounts"):
        service["volumeMounts"] = copy.deepcopy(frontend_policy["volume_mounts"])

    extra_pod_spec: dict[str, Any] = {}
    if frontend_policy.get("affinity", {}).get("same_zone_as_workers", False):
        extra_pod_spec["affinity"] = build_frontend_affinity(
            deployment_name, frontend_policy
        )

    main_container: dict[str, Any] = {
        "image": frontend_policy.get("image", raw_main.get("image")),
    }
    if frontend_policy.get("working_dir"):
        main_container["workingDir"] = frontend_policy["working_dir"]

    command = frontend_policy.get("command")
    if command:
        main_container["command"] = [
            "/bin/bash",
            "-c",
            build_command_block([str(command)]),
        ]
    elif raw_main.get("command"):
        main_container["command"] = copy.deepcopy(raw_main["command"])

    extra_pod_spec["mainContainer"] = main_container
    service["extraPodSpec"] = extra_pod_spec
    return service


def build_worker_command(
    raw_service: dict[str, Any],
    role: str,
    model: str,
    workers_policy: dict[str, Any],
) -> list[Any]:
    raw_main = raw_service.get("extraPodSpec", {}).get("mainContainer", {})
    raw_args = parse_cli_args(raw_main.get("args"))
    command_policy = workers_policy.get("command", {})
    module = command_policy.get(
        "module", command_module(raw_main.get("command"), "dynamo.vllm")
    )

    tensor_parallel_size = (
        command_policy.get("tensor_parallel_size", {}).get(role)
        or workers_policy.get("gpu_per_worker", {}).get(role)
        or raw_args.get("--tensor-parallel-size")
        or "1"
    )

    raw_kv_config = raw_args.get("--kv-transfer-config", "{}")
    try:
        kv_config = json.loads(str(raw_kv_config))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid --kv-transfer-config JSON for {role} worker: {raw_kv_config}"
        ) from exc

    extra_kv = command_policy.get("kv_connector_extra_config")
    if extra_kv:
        kv_config = deep_merge(
            kv_config,
            {"kv_connector_extra_config": copy.deepcopy(extra_kv)},
        )

    additional_args = command_policy.get("additional_args", {}).get(role, [])
    if isinstance(additional_args, dict):
        additional_arg_lines = [
            f" --{key.replace('_', '-')} {value} \\"
            for key, value in additional_args.items()
        ]
    else:
        additional_arg_lines = [f" {arg} \\" for arg in additional_args]

    kv_json = json.dumps(kv_config, separators=(",", ":"))
    lines = [
        f"exec python3 -m {module} \\",
        f" --model {model} \\",
        f" --tensor-parallel-size {tensor_parallel_size} \\",
        *additional_arg_lines,
        f" --disaggregation-mode {role} \\",
        f" --kv-transfer-config '{kv_json}'",
    ]
    return [
        "/bin/bash",
        "-c",
        build_command_block(lines),
    ]


def build_worker_service(
    service_name: str,
    raw_service: dict[str, Any],
    deployment_name: str,
    default_model: str | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    workers_policy = policy.get("workers", {})
    raw_main = raw_service.get("extraPodSpec", {}).get("mainContainer", {})
    role = component_role(service_name, raw_service)
    model = model_from_service(raw_service, policy) or default_model
    if model is None:
        raise ValueError(f"Could not find a --model argument for {service_name}")

    service: dict[str, Any] = {}
    if workers_policy.get("copy_env_from_secret", True):
        copy_if_present(service, "envFromSecret", raw_service)

    service["componentType"] = raw_service.get("componentType", "worker")
    service["subComponentType"] = role
    service["replicas"] = workers_policy.get("replicas", {}).get(
        role, raw_service.get("replicas", 1)
    )
    service["resources"] = worker_resources(role, workers_policy, raw_service)

    if workers_policy.get("volume_mounts"):
        service["volumeMounts"] = copy.deepcopy(workers_policy["volume_mounts"])

    extra_pod_spec: dict[str, Any] = {}
    if workers_policy.get("node_selector"):
        extra_pod_spec["nodeSelector"] = copy.deepcopy(workers_policy["node_selector"])
    if workers_policy.get("tolerations"):
        extra_pod_spec["tolerations"] = copy.deepcopy(workers_policy["tolerations"])

    affinity = build_worker_affinity(deployment_name, workers_policy)
    if affinity:
        extra_pod_spec["affinity"] = affinity

    main_container: dict[str, Any] = {}
    if workers_policy.get("env"):
        main_container["env"] = copy.deepcopy(workers_policy["env"])
    if workers_policy.get("security_context"):
        main_container["securityContext"] = copy.deepcopy(
            workers_policy["security_context"]
        )
    main_container["image"] = workers_policy.get("image", raw_main.get("image"))
    main_container["workingDir"] = workers_policy.get(
        "working_dir", raw_main.get("workingDir")
    )
    main_container["command"] = build_worker_command(
        raw_service,
        role,
        model,
        workers_policy,
    )

    extra_pod_spec["mainContainer"] = main_container
    service["extraPodSpec"] = extra_pod_spec
    return service


def transform_manifest(raw: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    raw_services = raw.get("spec", {}).get("services", {})
    if not isinstance(raw_services, dict):
        raise ValueError("raw manifest must contain spec.services as a mapping")

    output_policy = policy.get("output", {})
    metadata_policy = output_policy.get("metadata", {})
    deployment_name = metadata_policy.get("name") or raw.get("metadata", {}).get("name")
    if not deployment_name:
        raise ValueError("deployment name must exist in raw metadata or policy metadata")

    manifest: dict[str, Any] = {
        "apiVersion": raw.get("apiVersion"),
        "kind": raw.get("kind"),
        "metadata": {
            "name": deployment_name,
        },
    }

    namespace = metadata_policy.get("namespace", raw.get("metadata", {}).get("namespace"))
    if not nullish(namespace):
        manifest["metadata"]["namespace"] = namespace

    spec: dict[str, Any] = {}
    if output_policy.get("pvcs"):
        spec["pvcs"] = copy.deepcopy(output_policy["pvcs"])

    default_model = first_model_from_services(raw_services, policy)
    transformed_services: dict[str, Any] = {}
    for service_name in get_service_order(raw_services, policy):
        raw_service = raw_services[service_name]
        if not isinstance(raw_service, dict):
            continue

        component_type = raw_service.get("componentType")
        if component_type == "frontend":
            transformed_services[service_name] = build_frontend_service(
                raw_service,
                deployment_name,
                policy,
            )
        elif component_type == "worker":
            transformed_services[service_name] = build_worker_service(
                service_name,
                raw_service,
                deployment_name,
                default_model,
                policy,
            )
        else:
            transformed_services[service_name] = copy.deepcopy(raw_service)

    spec["services"] = transformed_services
    manifest["spec"] = spec
    return manifest


def cmd_init_db(args: argparse.Namespace) -> int:
    with sqlite3.connect(args.db) as conn:
        if args.seed_default_policy:
            seed_default_policy(conn)
        else:
            ensure_policy_schema(conn)
    return 0


def cmd_list_policies(args: argparse.Namespace) -> int:
    with sqlite3.connect(args.db) as conn:
        for name, description in list_policies(conn):
            if description:
                print(f"{name}\t{description}")
            else:
                print(name)
    return 0


def cmd_show_policy(args: argparse.Namespace) -> int:
    with sqlite3.connect(args.db) as conn:
        policy = show_policy(conn, args.policy)
    print(json.dumps(policy, indent=2))
    return 0


def cmd_upsert_policy(args: argparse.Namespace) -> int:
    policy = read_policy_file(args.file)
    with sqlite3.connect(args.db) as conn:
        upsert_policy(conn, args.policy, args.description, policy)
    return 0


def cmd_transform(args: argparse.Namespace) -> int:
    raw = read_yaml(args.input)
    with sqlite3.connect(args.db) as conn:
        policy = load_policy(conn, args.policy)

    manifest = transform_manifest(raw, policy)
    include_spdx_header = bool(policy.get("output", {}).get("include_spdx_header", False))
    write_yaml(args.output, manifest, include_spdx_header)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform raw DynamoGraphDeployment YAML using a DB-backed deployment policy."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="Create the policy DB schema")
    init_db.add_argument("--db", required=True, help="SQLite policy database path")
    init_db.add_argument(
        "--seed-default-policy",
        action="store_true",
        help=f"Upsert the sample {DEFAULT_POLICY_NAME!r} policy",
    )
    init_db.set_defaults(func=cmd_init_db)

    list_db = subparsers.add_parser("list-policies", help="List policies in the DB")
    list_db.add_argument("--db", required=True, help="SQLite policy database path")
    list_db.set_defaults(func=cmd_list_policies)

    show = subparsers.add_parser("show-policy", help="Print a policy JSON document")
    show.add_argument("--db", required=True, help="SQLite policy database path")
    show.add_argument("--policy", required=True, help="Policy name")
    show.set_defaults(func=cmd_show_policy)

    upsert = subparsers.add_parser(
        "upsert-policy", help="Insert or update a policy from JSON/YAML"
    )
    upsert.add_argument("--db", required=True, help="SQLite policy database path")
    upsert.add_argument("--policy", required=True, help="Policy name")
    upsert.add_argument("--file", required=True, type=Path, help="Policy JSON/YAML file")
    upsert.add_argument("--description", help="Optional policy description")
    upsert.set_defaults(func=cmd_upsert_policy)

    transform = subparsers.add_parser(
        "transform", help="Transform raw YAML into deployment-ready YAML"
    )
    transform.add_argument("--db", required=True, help="SQLite policy database path")
    transform.add_argument("--policy", required=True, help="Policy name")
    transform.add_argument("--input", required=True, type=Path, help="Raw manifest YAML")
    transform.add_argument(
        "--output", required=True, type=Path, help="Deployment-ready YAML"
    )
    transform.set_defaults(func=cmd_transform)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

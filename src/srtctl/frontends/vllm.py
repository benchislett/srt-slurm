# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-native frontend implementation.

For aggregated single-worker configs, the vLLM worker process serves the
OpenAI API directly on port 8000 — there is no separate frontend process
to launch, no NATS/etcd, and no Dynamo install. This sidesteps the Dynamo
NIXL init path that has been crashing with native heap corruption on the
vllm/vllm-openai:v0.19.1-cu130 image.

Scope (MVP): aggregated mode with one worker only. Multi-worker routing
and P/D disaggregation stay on the Dynamo path.
"""

import logging
from typing import TYPE_CHECKING, Any

from srtctl.core.health import WorkerHealthResult, check_vllm_native_health

if TYPE_CHECKING:
    from srtctl.core.processes import ManagedProcess
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.topology import Process

logger = logging.getLogger(__name__)


class VLLMFrontend:
    """vLLM native OpenAI server as the frontend.

    The worker process itself binds :8000 and serves /v1/* and /health, so
    start_frontends() returns an empty list.
    """

    @property
    def type(self) -> str:
        return "vllm"

    @property
    def health_endpoint(self) -> str:
        return "/health"

    def parse_health(
        self,
        response_json: dict,
        expected_prefill: int,
        expected_decode: int,
    ) -> WorkerHealthResult:
        return check_vllm_native_health(response_json, expected_prefill, expected_decode)

    def get_frontend_args_list(self, args: dict[str, Any] | None) -> list[str]:
        if not args:
            return []
        result = []
        for key, value in args.items():
            if value is True:
                result.append(f"--{key}")
            elif value is not False and value is not None:
                result.extend([f"--{key}", str(value)])
        return result

    def start_frontends(
        self,
        topology: Any,  # FrontendTopology
        runtime: "RuntimeContext",
        config: Any,  # SrtConfig
        backend: Any,  # BackendProtocol
        backend_processes: list["Process"],
    ) -> list["ManagedProcess"]:
        logger.info(
            "vLLM-native frontend: worker serves OpenAI API directly on port %d; no separate frontend process",
            topology.public_port,
        )
        return []

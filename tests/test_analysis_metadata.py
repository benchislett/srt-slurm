# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for analysis dashboard metadata loading, specifically total_gpus accounting."""

import os
import sys
import tempfile

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.srtlog.models import RunMetadata  # noqa: E402


class TestRunMetadataTotalGpus:
    def test_aggregated_prefers_agg_workers_times_gpus_per_agg(self):
        """When gpus_per_agg is set explicitly (e.g. DP=4 on an 8-GPU node)
        total_gpus should reflect actual worker GPU usage, not the whole node."""
        json_data = {
            "job_id": "123",
            "resources": {
                "gpu_type": "b300",
                "gpus_per_node": 8,
                "prefill_nodes": None,
                "decode_nodes": None,
                "prefill_workers": 0,
                "decode_workers": 0,
                "agg_nodes": 1,
                "agg_workers": 1,
                "gpus_per_agg": 4,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            md = RunMetadata.from_json(json_data, tmp)
            assert md.is_aggregated
            assert md.total_gpus == 4

    def test_aggregated_legacy_json_backfills_from_config_yaml(self):
        """Older metadata JSONs omit agg_nodes / gpus_per_agg; verify we
        pick them up from config.yaml co-located in the run dir."""
        json_data = {
            "job_id": "123",
            "resources": {
                "gpu_type": "b300",
                "gpus_per_node": 8,
                "prefill_nodes": None,
                "decode_nodes": None,
                "prefill_workers": 0,
                "decode_workers": 0,
                "agg_workers": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.yaml"), "w") as f:
                f.write(
                    "resources:\n"
                    "  gpu_type: b300\n"
                    "  gpus_per_node: 8\n"
                    "  agg_nodes: 1\n"
                    "  agg_workers: 1\n"
                    "  gpus_per_agg: 4\n"
                )
            md = RunMetadata.from_json(json_data, tmp)
            assert md.is_aggregated
            assert md.agg_nodes == 1
            assert md.gpus_per_agg == 4
            assert md.total_gpus == 4

    def test_aggregated_falls_back_to_agg_nodes_when_gpus_per_agg_missing(self):
        json_data = {
            "job_id": "123",
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "agg_nodes": 2,
                "agg_workers": 2,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            md = RunMetadata.from_json(json_data, tmp)
            assert md.total_gpus == 16

    def test_disaggregated_uses_per_worker_when_available(self):
        json_data = {
            "job_id": "123",
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "prefill_nodes": 1,
                "decode_nodes": 2,
                "prefill_workers": 2,
                "decode_workers": 4,
                "gpus_per_prefill": 4,
                "gpus_per_decode": 4,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            md = RunMetadata.from_json(json_data, tmp)
            assert not md.is_aggregated
            assert md.total_gpus == 2 * 4 + 4 * 4

    def test_disaggregated_falls_back_to_node_count(self):
        """When gpus_per_prefill/decode are absent, fall back to
        (prefill_nodes + decode_nodes) * gpus_per_node."""
        json_data = {
            "job_id": "123",
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "prefill_nodes": 1,
                "decode_nodes": 2,
                "prefill_workers": 1,
                "decode_workers": 2,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            md = RunMetadata.from_json(json_data, tmp)
            assert md.total_gpus == 3 * 8

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import json
import logging
import time

import torch

from verl.utils.device import get_torch_device

from .response_only_lm_head import _get_output_layer

logger = logging.getLogger(__name__)

PROFILE_PREFIX = "VERL_RESPONSE_ONLY_LM_HEAD_PROFILE "
_EVENT_IDS = itertools.count()


class ResponseOnlyLMHeadProfiler:
    """Profile the LM-head projection and its downstream vocabulary operations.

    This helper is intended for controlled A/B runs. On accelerator devices it
    synchronizes execution and resets peak-memory statistics immediately before
    the output projection. Normal training never constructs this object.
    """

    def __init__(
        self,
        model,
        projection_mask: torch.Tensor,
        *,
        response_only_enabled: bool,
        metadata: dict | None = None,
    ):
        self.output_layer = _get_output_layer(model) if model is not None else None
        self.projection_mask = projection_mask
        self.response_only_enabled = response_only_enabled
        self.metadata = dict(metadata or {})
        self.event_id = next(_EVENT_IDS)
        self.selected_rows = int(projection_mask.sum().item())
        self.dense_rows = projection_mask.numel()
        self._pre_handle = None
        self._post_handle = None
        self._started_at = None
        self._record = None

        device = get_torch_device()
        self._device = device
        self._tracks_accelerator_memory = bool(device.is_available()) and all(
            hasattr(device, name)
            for name in ("memory_allocated", "max_memory_allocated", "reset_peak_memory_stats", "synchronize")
        )

    @staticmethod
    def _logits_from_output(output) -> torch.Tensor:
        logits = output[0] if isinstance(output, tuple | list) else output
        if not isinstance(logits, torch.Tensor) or logits.ndim < 2:
            raise RuntimeError("Could not identify the tensor returned by the Megatron output layer")
        return logits

    def _before_projection(self, _module, _args, _kwargs):
        self.start()

    def start(self):
        """Start at the hidden-state boundary, before SP gather and row selection."""
        allocated_before = None
        if self._tracks_accelerator_memory:
            self._device.synchronize()
            allocated_before = self._device.memory_allocated()
            self._device.reset_peak_memory_stats()
        self._started_at = time.perf_counter()
        self._record = {"allocated_before_bytes": allocated_before}

    def _after_projection(self, _module, _args, _kwargs, output):
        logits = self._logits_from_output(output)
        vocab_shard = logits.shape[-1]
        projected_rows = logits.numel() // vocab_shard
        element_size = logits.element_size()
        self._record.update(
            {
                "backend": "unfused",
                "projected_rows": projected_rows,
                "vocab_shard": vocab_shard,
                "logits_dtype": str(logits.dtype),
                "element_size": element_size,
                "actual_logits_bytes": logits.numel() * element_size,
                "dense_logits_bytes": self.dense_rows * vocab_shard * element_size,
            }
        )

    def record_fused_projection(self, hidden_states, weight):
        """Record kernel input geometry without inventing materialized logits."""
        self._record.update(
            {
                "backend": "fused",
                "projected_rows": hidden_states.numel() // hidden_states.shape[-1],
                "vocab_shard": weight.shape[0],
                "logits_dtype": "none",
                "element_size": 0,
                "actual_logits_bytes": 0,
                "dense_logits_bytes": 0,
            }
        )

    def __enter__(self):
        # Register before response_only_output_projection so the timer includes
        # explicit SP gather, row selection, and the vocabulary projection.
        if self.output_layer is None:
            raise RuntimeError("Unfused LM-head profiling requires an output layer")
        self._pre_handle = self.output_layer.register_forward_pre_hook(self._before_projection, with_kwargs=True)
        self._post_handle = self.output_layer.register_forward_hook(self._after_projection, with_kwargs=True)
        return self

    def finish(self) -> dict:
        """Finish the sample after log-probability/entropy processing and emit JSON."""
        if self._record is None or self._started_at is None:
            raise RuntimeError("LM-head profiler did not observe an output-layer invocation")
        if "actual_logits_bytes" not in self._record:
            raise RuntimeError("LM-head profiler did not observe output-layer logits")

        peak_allocated = None
        peak_increment = None
        if self._tracks_accelerator_memory:
            self._device.synchronize()
            peak_allocated = self._device.max_memory_allocated()
            peak_increment = max(0, peak_allocated - self._record["allocated_before_bytes"])
        elapsed_s = time.perf_counter() - self._started_at

        dense_logits_bytes = self._record["dense_logits_bytes"]
        actual_logits_bytes = self._record["actual_logits_bytes"]
        record = {
            **self.metadata,
            "event_id": self.event_id,
            "rank": torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
            "response_only_enabled": self.response_only_enabled,
            "dense_rows": self.dense_rows,
            "selected_rows": self.selected_rows,
            "active_ratio": self.selected_rows / self.dense_rows if self.dense_rows else 0.0,
            "elapsed_s": elapsed_s,
            "peak_allocated_bytes": peak_allocated,
            "peak_increment_bytes": peak_increment,
            "logits_saved_bytes": dense_logits_bytes - actual_logits_bytes,
            **self._record,
        }
        logger.warning("%s%s", PROFILE_PREFIX, json.dumps(record, sort_keys=True))
        return record

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self._post_handle is not None:
            self._post_handle.remove()
        if self._pre_handle is not None:
            self._pre_handle.remove()

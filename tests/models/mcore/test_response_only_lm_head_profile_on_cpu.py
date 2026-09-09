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

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from verl.models.mcore import model_forward_fused as mff
from verl.models.mcore.response_only_lm_head import response_only_output_projection
from verl.models.mcore.response_only_lm_head_profile import (
    PROFILE_PREFIX,
    ResponseOnlyLMHeadProfiler,
)


class _OutputLayer(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.sequence_parallel = False
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_size))

    def forward(self, input_):
        return torch.nn.functional.linear(input_, self.weight), None


class _Model(torch.nn.Module):
    def __init__(self, output_layer):
        super().__init__()
        self.output_layer = output_layer

    def forward(self, hidden_states):
        logits, _ = self.output_layer(hidden_states)
        return logits.transpose(0, 1).contiguous()


@pytest.mark.parametrize(("enabled", "projected_rows"), [(False, 8), (True, 4)])
def test_profiler_records_projection_rows_and_logits_memory(enabled, projected_rows):
    model = _Model(_OutputLayer(3, 5))
    hidden = torch.randn(4, 2, 3)
    projection_mask = torch.tensor([[False, True, True, False], [True, False, True, False]])
    profiler = ResponseOnlyLMHeadProfiler(
        model,
        projection_mask,
        response_only_enabled=enabled,
        metadata={"role": "actor", "calculate_entropy": True},
    )
    projection = response_only_output_projection(model, projection_mask) if enabled else nullcontext()

    with patch("verl.models.mcore.response_only_lm_head_profile.logger.warning") as warning:
        with profiler, projection:
            logits = model(hidden)
        record = profiler.finish()

    warning.assert_called_once()
    fmt, prefix, payload = warning.call_args.args
    assert (fmt, prefix) == ("%s%s", PROFILE_PREFIX)
    assert json.loads(payload) == record
    assert record["response_only_enabled"] is enabled
    assert record["role"] == "actor"
    assert record["calculate_entropy"] is True
    assert record["dense_rows"] == 8
    assert record["selected_rows"] == 4
    assert record["projected_rows"] == projected_rows
    assert record["active_ratio"] == 0.5
    assert record["vocab_shard"] == 5
    assert record["dense_logits_bytes"] == 8 * 5 * logits.element_size()
    assert record["actual_logits_bytes"] == projected_rows * 5 * logits.element_size()
    assert record["logits_saved_bytes"] == (8 - projected_rows) * 5 * logits.element_size()


@pytest.mark.parametrize("mode", ["hook", "legacy"])
@pytest.mark.parametrize("enabled", [False, True])
def test_fused_engine_profiler_keeps_baseline_dense(mode, enabled):
    input_ids = torch.nested.as_nested_tensor([torch.tensor([0, 1, 2, 3])], layout=torch.jagged)
    loss_mask = torch.nested.as_nested_tensor([torch.tensor([0, 1])], layout=torch.jagged)
    hidden = torch.randn(4, 1, 3)

    class Model(_Model):
        pre_process = True
        post_process = True
        config = SimpleNamespace(fp8=None, sequence_parallel=False)
        _verl_fused_forward_mode = mode

        def _preprocess(self, **kwargs):
            return hidden, None, None, None, None

        def decoder(self, **kwargs):
            return kwargs["hidden_states"]

        def forward(self, **kwargs):
            if mode == "legacy":
                return mff._fused_GPTModel_forward(self, **kwargs)
            return kwargs["output_processor"](
                hidden_states=hidden,
                output_layer=self.output_layer,
                output_weight=None,
                labels=kwargs["labels"],
                context=kwargs["output_processor_context"],
                config=self.config,
            )

    def kernel(h, w, labels, *args):
        logits = h.reshape(-1, 3) @ w.T
        return logits.sum(-1), logits.square().sum(-1)

    with (
        patch.object(mff.parallel_state, "get_context_parallel_world_size", return_value=1),
        patch.object(mff.parallel_state, "get_context_parallel_rank", return_value=0),
        patch.object(mff.parallel_state, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(mff.parallel_state, "get_tensor_model_parallel_group", return_value=None),
        patch.object(mff, "postprocess_thd_engine", side_effect=lambda value, *args, **kwargs: value),
        patch.object(mff, "has_config_logger_enabled", return_value=False),
        patch.object(mff, "linear_cross_entropy", side_effect=kernel),
        patch("verl.models.mcore.response_only_lm_head_profile.logger.warning") as warning,
    ):
        mff.fused_forward_model_engine()(
            Model(_OutputLayer(3, 5)),
            input_ids,
            input_ids,
            {},
            1.0,
            True,
            0,
            loss_mask=loss_mask,
            response_only_lm_head_profile={"response_only_enabled": enabled},
        )
    warning.assert_called_once()
    record = json.loads(warning.call_args.args[2])
    assert record["backend"] == "fused"
    assert record["selected_rows"] == 1
    assert record["dense_rows"] == 4
    assert record["projected_rows"] == (1 if enabled else 4)
    assert record["actual_logits_bytes"] == record["dense_logits_bytes"] == record["logits_saved_bytes"] == 0

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
"""Run with: torchrun --standalone --nproc_per_node=4 -m pytest -q {file}."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu

from verl.models.mcore.model_forward_fused import _compute_fused_lm_head


@pytest.fixture(scope="module", autouse=True)
def parallel_groups():
    if not torch.cuda.is_available() or "LOCAL_RANK" not in os.environ:
        pytest.skip("Requires torchrun with four CUDA GPUs")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    assert dist.get_world_size() == 4, "Use four ranks for TP=2, CP=2"
    mpu.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)
    yield
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    ("sequence_parallel", "empty_cp_rank", "entropy_coeff", "dtype"),
    [
        (False, False, 0.0, torch.float32),
        (True, False, 0.1, torch.bfloat16),
        (False, True, 0.1, torch.bfloat16),
        (True, True, 0.1, torch.bfloat16),
    ],
)
def test_fused_head_matches_dense_reference(sequence_parallel, empty_cp_rank, entropy_coeff, dtype):
    tp_rank = mpu.get_tensor_model_parallel_rank()
    cp_rank = mpu.get_context_parallel_rank()
    rows, hidden_size, vocab_size = 128, 256, 4096
    torch.manual_seed(42)
    weight = (torch.randn(vocab_size, hidden_size, device="cuda") * 0.02).to(dtype)
    torch.manual_seed(43 + cp_rank)
    hidden = torch.randn(rows, 1, hidden_size, device="cuda", dtype=dtype)
    labels = torch.randint(vocab_size, (1, rows), device="cuda")
    mask = torch.ones_like(labels, dtype=torch.bool)
    mask[:, :96] = False
    mask[:, 100:110] = False
    if empty_cp_rank and cp_rank == 0:
        mask.zero_()
    coefficients = torch.randn(1, rows, device="cuda") * mask / rows

    ref_hidden = hidden.detach().float().requires_grad_()
    ref_weight = weight.detach().float().requires_grad_()
    ref_log_probs = (ref_hidden[:, 0] @ ref_weight.T / 0.7).log_softmax(-1)
    ref_entropy = -(ref_log_probs.exp() * ref_log_probs).sum(-1).reshape(1, rows)
    ref_log_probs = ref_log_probs.gather(-1, labels.T).T
    ((ref_log_probs + entropy_coeff * ref_entropy) * coefficients).sum().backward()

    token_slice = slice(tp_rank * (rows // 2), (tp_rank + 1) * (rows // 2)) if sequence_parallel else slice(None)
    vocab_slice = slice(tp_rank * (vocab_size // 2), (tp_rank + 1) * (vocab_size // 2))
    results = []
    for projection_mask in (None, mask):
        local_hidden = hidden[token_slice].clone().requires_grad_()
        local_weight = weight[vocab_slice].clone().requires_grad_()
        log_probs, entropy = _compute_fused_lm_head(
            local_hidden, local_weight, labels, 0.7, sequence_parallel, projection_mask
        )
        log_probs, entropy = log_probs.reshape_as(mask), entropy.reshape_as(mask)
        ((log_probs + entropy_coeff * entropy) * coefficients).sum().backward()
        torch.testing.assert_close(log_probs[mask], ref_log_probs[mask], rtol=3e-3, atol=3e-2)
        torch.testing.assert_close(entropy[mask], ref_entropy[mask], rtol=3e-3, atol=3e-2)
        torch.testing.assert_close(local_hidden.grad.float(), ref_hidden.grad[token_slice], rtol=3e-2, atol=1e-4)
        torch.testing.assert_close(local_weight.grad.float(), ref_weight.grad[vocab_slice], rtol=3e-2, atol=1e-4)
        results.append((local_hidden.grad.float(), local_weight.grad.float()))

    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1e-4)

    actual_weight_grad = results[1][1].clone()
    expected_weight_grad = ref_weight.grad[vocab_slice].contiguous()
    dist.all_reduce(actual_weight_grad, group=mpu.get_context_parallel_group())
    dist.all_reduce(expected_weight_grad, group=mpu.get_context_parallel_group())
    torch.testing.assert_close(actual_weight_grad, expected_weight_grad, rtol=3e-2, atol=1e-4)

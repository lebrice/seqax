"""Main training loop, including the model, loss function, and optimizer."""

import contextlib
import dataclasses
import datetime
import functools
import hashlib
import itertools
import logging
import math
import operator
import os
import time
from hydra_zen import hydrated_dataclass

from dataclasses import dataclass
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple
import datasets
import einops
import hydra_zen
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import rich.logging
import torch
import wandb
import wandb.wandb_run
import yaml
from jax import lax
from jax._src.distributed import initialize as jax_distributed_initialize
from jax.experimental import mesh_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import Mesh, PartitionSpec
from jax.tree_util import tree_leaves
from typeguard import typechecked
from transformers import PreTrainedTokenizerFast

import jax_extra
import shardlib.shardops as shardops
import shardlib.shardtypes as shardtypes
import training_io
from input_loader import (
    FlatTokensParams,
    HuggingFaceDataParams,
    ShufflingLoader,
    TokenBatch,
    TokenBatchParams,
)
import hydra
from jax_extra import (
    explicit_activation_checkpointing,
    fold_in_str,
    make_dataclass_from_dict,
    save_for_backward,
)
from hydra.utils import instantiate
from shardlib.shardtypes import (
    bf16,
    bool_,
    f32,
    make_shardings,
    pytree_dataclass,
    u32,
)

# NOTE: Removed this. Seems to be specific to TPUs (who even has those except Google?!)
# import env
# env.set_variables()  # noqa
SCRATCH = Path(os.environ["SCRATCH"])

# TODO: Double-check that we can get away with adding this here.
shardtypes.register_with_typeguard()  # noqa

log = logging.getLogger(__name__)


def multiply(numbers: Iterable[Any]):
    return functools.reduce(operator.mul, numbers)


omegaconf.OmegaConf.register_new_resolver("int", int, replace=True)
omegaconf.OmegaConf.register_new_resolver("eval", eval, replace=True)
omegaconf.OmegaConf.register_new_resolver(
    "mul", lambda *numbers: multiply(numbers), replace=True
)
P = PartitionSpec
PRNGKey = Any


def get_gpus_per_task(tres_per_task: str) -> int:
    """Returns the number of GPUS per task from the SLURM env variables.

    >>> get_gpus_per_task('cpu=48,gres/gpu=4')
    4
    >>> get_gpus_per_task('cpu=48,gres/gpu=h100=2')
    2
    >>> get_gpus_per_task('cpu=48')
    0
    >>> get_gpus_per_task('cpu=48,gres/gpu:h100=4')
    """
    # todo: figure out how many GPUS per task given the tres_per_task.
    # Example: 'cpu=48,gres/gpu=4' --> 4
    # Example: 'cpu=48,gres/gpu=h100:4' --> 4
    for part in tres_per_task.split(","):
        res_type, _, res_count = part.partition(":")
        if res_type == "gres/gpu":
            gpus_per_task = int(res_count.rpartition("=")[-1])
            assert gpus_per_task > 0
            return gpus_per_task
    return 0


@dataclasses.dataclass(frozen=True)
class SlurmDistributedEnv:
    global_rank: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_PROCID"])
    )
    local_rank: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_LOCALID"])
    )
    num_tasks: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_NTASKS"])
    )
    num_nodes: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_JOB_NUM_NODES"])
    )
    ntasks_per_node: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_NTASKS_PER_NODE"])
    )
    cpus_per_task: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_CPUS_PER_TASK"])
    )
    gpus_per_task: int = dataclasses.field(
        default_factory=lambda: get_gpus_per_task(os.environ["SLURM_TRES_PER_TASK"])
        or int(os.environ["SLURM_GPUS_ON_NODE"])
    )
    node_id: int = dataclasses.field(
        default_factory=lambda: int(os.environ["SLURM_NODEID"])
    )
    node_list: tuple[str, ...] = dataclasses.field(
        default_factory=lambda: tuple(os.environ["SLURM_JOB_NODELIST"].split(","))
    )


@dataclass(frozen=True)
class Hparams:
    d_model: int
    n_q_per_kv: int
    n_kv: int
    d_head: int
    layers: int
    vocab: int
    d_ff: int
    rope_max_timescale: int


@pytree_dataclass
class TransformerLayer:
    ln1: f32[" d_model/t/d"]
    ln2: f32[" d_model/t/d"]
    w_q: f32["d_model/d n_q_per_kv n_kv/t d_head"]
    w_kv: f32["2 d_model/d n_kv/t d_head"]
    w_o: f32["d_model/d n_q_per_kv n_kv/t d_head"]
    w_gate: f32["d_model/d d_ff/t"]
    w_up: f32["d_model/d d_ff/t"]
    w_down: f32["d_model/d d_ff/t"]


@pytree_dataclass
class Transformer(TransformerLayer):
    ln1: f32["layers d_model/t/d"]
    ln2: f32["layers d_model/t/d"]
    w_q: f32["layers d_model/d n_q_per_kv n_kv/t d_head"]
    w_kv: f32["layers 2 d_model/d n_kv/t d_head"]
    w_o: f32["layers d_model/d n_q_per_kv n_kv/t d_head"]
    w_gate: f32["layers d_model/d d_ff/t"]
    w_up: f32["layers d_model/d d_ff/t"]
    w_down: f32["layers d_model/d d_ff/t"]


# Transformer = Array["layers", TransformerLayer]


@pytree_dataclass
class Model:
    embed: f32["vocab/t d_model/d"]
    unembed: f32["vocab/t d_model/d"]
    transformer: Transformer
    final_layer_norm: f32[" d_model/d/t"]

    @staticmethod
    @typechecked
    def init(h: Hparams, rng: PRNGKey) -> "Model":
        embed = jax.random.normal(
            jax_extra.fold_in_str(rng, "embed"), (h.vocab, h.d_model), dtype=jnp.float32
        )
        # https://github.com/google/jax/issues/20390 for ones_like with sharding.
        ln1 = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        ln2 = jnp.ones((h.layers, h.d_model), dtype=jnp.float32)
        final_layer_norm = jnp.ones((h.d_model,), dtype=jnp.float32)

        # All of wi/wq/wo/wo/w_kv use truncated_normal initializers with 'fan_in' scaling,
        # i.e. variance set to 1.0/fan_in.
        # The constant is stddev of standard normal truncated to (-2, 2)
        truncated_normal_stddev = 0.87962566103423978

        # scale for tensors with d_model fan_in and truncated normal truncated to (-2, 2)
        d_model_scale = 1 / (math.sqrt(h.d_model) * truncated_normal_stddev)

        w_kv_scale = d_model_scale
        w_q_scale = d_model_scale / math.sqrt(h.d_head)
        total_head_dim = h.n_q_per_kv * h.n_kv * h.d_head
        w_o_scale = 1 / (math.sqrt(total_head_dim) * truncated_normal_stddev)
        w_up_scale = d_model_scale
        w_down_scale = 1 / (math.sqrt(h.d_ff) * truncated_normal_stddev)
        unembed_scale = d_model_scale

        w_kv_shape = (h.layers, 2, h.d_model, h.n_kv, h.d_head)
        w_kv = w_kv_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_kv"), -2, 2, w_kv_shape, dtype=jnp.float32
        )
        w_q_shape = (h.layers, h.d_model, h.n_q_per_kv, h.n_kv, h.d_head)
        w_q = w_q_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_q"), -2, 2, w_q_shape, dtype=jnp.float32
        )
        w_kv_shape = (h.layers, 2, h.d_model, h.n_kv, h.d_head)
        w_kv = w_kv_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_kv"), -2, 2, w_kv_shape, dtype=jnp.float32
        )
        w_o_shape = w_q_shape
        w_o = w_o_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_o"), -2, 2, w_o_shape, dtype=jnp.float32
        )

        ff_shape = (h.layers, h.d_model, h.d_ff)
        w_gate = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_gate"), -2, 2, ff_shape, dtype=jnp.float32
        )
        w_up = w_up_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_up"), -2, 2, ff_shape, dtype=jnp.float32
        )
        w_down = w_down_scale * jax.random.truncated_normal(
            fold_in_str(rng, "w_down"), -2, 2, ff_shape, dtype=jnp.float32
        )

        unembed = unembed_scale * jax.random.truncated_normal(
            fold_in_str(rng, "unembed"), -2, 2, (h.vocab, h.d_model), dtype=jnp.float32
        )
        arrays = Model(
            embed=embed,
            unembed=unembed,
            transformer=Transformer(
                ln1=ln1,
                ln2=ln2,
                w_q=w_q,
                w_kv=w_kv,
                w_o=w_o,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
            ),
            final_layer_norm=final_layer_norm,
        )
        shardings = make_shardings(Model)
        sharded = jax.tree.map(lax.with_sharding_constraint, arrays, shardings)
        assert isinstance(sharded, Model)
        return sharded

    @typechecked
    def forward_pass(
        self, h: Hparams, ids: u32[b"B/d L"], is_seq_start: bool_[b"B/d L"]
    ) -> f32[b"B/d L V/t"]:
        ##### Initial embedding lookup.
        embed = shardops.all_gather("V/t M/d -> V/t M", jnp.bfloat16(self.embed))
        x = shardops.index_unreduced("[V/t] M, B/d L -> B/d L M", embed, ids)
        x = shardops.psum_scatter("B/d L M -> B/d L M/t", x)

        L = ids.shape[1]
        segment_ids = jnp.cumsum(is_seq_start, axis=1)
        segment_mask: bool_[b"B/d L L"] = (
            segment_ids[:, :, jnp.newaxis] == segment_ids[:, jnp.newaxis, :]
        )
        segment_mask: bool_[b"B/d L L 1 1"] = segment_mask[
            ..., jnp.newaxis, jnp.newaxis
        ]  # add axes for q_per_k, num_kv_heads dimensions
        causal_mask: bool_[b"1 L L 1 1"] = jnp.tril(
            jnp.ones((L, L), dtype=jnp.bool_), 0
        )[jnp.newaxis, ..., jnp.newaxis, jnp.newaxis]
        causal_mask: bool_[b"B/d L L 1 1"] = jnp.logical_and(segment_mask, causal_mask)

        rope_table = RopeTable.create(L, h)

        ##### Transformer blocks.
        @explicit_activation_checkpointing
        @typechecked
        def loop_body(
            x: bf16[b"B/d L M/t"], layer_weights: TransformerLayer
        ) -> Tuple[bf16[b"B/d L M/t"], Tuple[()]]:
            # Pre-attention RMSNorm
            ln1 = shardops.all_gather("M/t/d -> M", jnp.float32(layer_weights.ln1))
            gx = shardops.all_gather("B/d L M/t -> B/d L M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln1)

            # Attention, using Grouped Query Attention and RoPE position embeddings.
            w_q = shardops.all_gather(
                "M/d Q K/t D -> M Q K/t D", jnp.bfloat16(layer_weights.w_q)
            )
            q = save_for_backward(
                shardops.einsum_unreduced(
                    "B/d L M, M Q K/t D -> B/d L Q K/t D", nx, w_q
                )
            )
            q = rope_table.apply("L D -> 1 L 1 1 D", q)
            w_kv = shardops.all_gather(
                "2 M/d K/t D -> 2 M K/t D", jnp.bfloat16(layer_weights.w_kv)
            )
            k, v = shardops.einsum_unreduced(
                "B/d L M, k_v M K/t D -> k_v B/d L K/t D", nx, w_kv
            )
            k = save_for_backward(k)
            v = save_for_backward(v)
            k = rope_table.apply("L d -> 1 L 1 d", k)
            logits = shardops.einsum_unreduced(
                "B/d Qlen Q K/t D, B/d Klen K/t D -> B/d Qlen Klen Q K/t",
                q,
                k,
                preferred_element_type=jnp.float32,
            )
            logits = jnp.where(causal_mask, logits, -1e10)
            probs = jnp.bfloat16(jax.nn.softmax(logits, axis=2))
            attn_out = shardops.einsum_unreduced(
                "B/d Qlen Klen Q K/t, B/d Klen K/t D -> B/d Qlen Q K/t D", probs, v
            )
            w_o = shardops.all_gather(
                "M/d Q K/t D -> M Q K/t D", jnp.bfloat16(layer_weights.w_o)
            )
            attn_out = shardops.einsum_unreduced(
                "B/d Qlen Q K/t D, M Q K/t D -> B/d Qlen M", attn_out, w_o
            )
            attn_out = shardops.psum_scatter("B/d Qlen M -> B/d Qlen M/t", attn_out)
            x = save_for_backward(x + attn_out)

            # Pre-FFN RMSNorm
            ln2 = save_for_backward(
                shardops.all_gather("M/t/d -> M", jnp.float32(layer_weights.ln2))
            )
            gx = shardops.all_gather("B/d L M/t -> B/d L M", x)
            nx = jnp.bfloat16(rms_norm(gx) * ln2)

            # FFN, using SwiGLU
            w_gate = shardops.all_gather(
                "M/d F/t -> M F/t", jnp.bfloat16(layer_weights.w_gate)
            )
            gate_proj = save_for_backward(
                shardops.einsum_unreduced("B/d L M, M F/t -> B/d L F/t", nx, w_gate)
            )
            w_up = shardops.all_gather(
                "M/d F/t -> M F/t", jnp.bfloat16(layer_weights.w_up)
            )
            up_proj = save_for_backward(
                shardops.einsum_unreduced("B/d L M, M F/t -> B/d L F/t", nx, w_up)
            )
            y = jax.nn.swish(gate_proj) * up_proj
            w_down = shardops.all_gather(
                "M/d F/t -> M F/t", jnp.bfloat16(layer_weights.w_down)
            )
            ffn_out = shardops.einsum_unreduced(
                "B/d L F/t, M F/t -> B/d L M", y, w_down
            )
            ffn_out = shardops.psum_scatter("B/d L M -> B/d L M/t", ffn_out)

            return jnp.bfloat16(x + ffn_out), ()

        x, () = jax.lax.scan(loop_body, jnp.bfloat16(x), self.transformer)

        ##### Final layernorm and output projection.
        x = shardops.all_gather("B/d L M/t -> B/d L M", x)
        ln = shardops.all_gather("M/t/d -> M", jnp.float32(self.final_layer_norm))
        x = jnp.bfloat16(rms_norm(x) * ln)
        unembed = shardops.all_gather("V/t M/d -> V/t M", jnp.bfloat16(self.unembed))
        logits = shardops.einsum_unreduced(
            "B/d L M, V/t M -> B/d L V/t",
            x,
            unembed,
            preferred_element_type=jnp.float32,
        )

        return logits

    @typechecked
    def loss(self, h: Hparams, batch: TokenBatch) -> f32[b""]:
        # Given sequence-packed targets:
        #   [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        # we want inputs:
        #   [[0, 1], [0, 3, 4], [0, 6, 7, 8]]
        # which we get by shifting the targets right by 1 and
        # masking sequence-start tokens to 0.
        inputs = jnp.pad(batch.targets[:, :-1], pad_width=((0, 0), (1, 0)))
        is_seq_start: bool_[b"batch/d len"] = batch.is_seq_start
        inputs: u32[b"batch/d len"] = jnp.where(is_seq_start, 0, inputs)

        logits: f32[b"batch/d len V/t"] = self.forward_pass(h, inputs, is_seq_start)
        max_logits: f32[b"batch/d len 1"] = lax.pmax(
            jnp.max(lax.stop_gradient(logits), axis=-1, keepdims=True), "t"
        )
        logits = logits - max_logits
        sum_logits = lax.psum(jnp.sum(jnp.exp(logits), axis=-1, keepdims=True), "t")
        logsumexp = jnp.log(sum_logits)
        logprobs: f32[b"batch/d len V/t"] = logits - logsumexp
        logprobs_at_targets = shardops.index_unreduced(
            "batch/d len [V/t], batch/d len -> batch/d len", logprobs, batch.targets
        )
        logprobs_at_targets = shardops.psum_scatter(
            "batch/d len -> batch/d len/t", logprobs_at_targets
        )
        tokens_in_global_batch = logprobs_at_targets.size * jax.lax.psum(1, ("d", "t"))
        return -jnp.sum(logprobs_at_targets) / jnp.float32(tokens_in_global_batch)


@pytree_dataclass
class RopeTable:
    sin: f32["len d_head2"]
    cos: f32["len d_head2"]

    @staticmethod
    def create(max_len: int, hparams: Hparams) -> "RopeTable":
        rope_max_timescale = hparams.rope_max_timescale
        d_head = hparams.d_head
        d = d_head // 2
        # endpoint=False is equivalent to what MaxText does. endpoint=True would be more natural, though.
        timescale = jnp.logspace(
            0, jnp.log10(jnp.float32(rope_max_timescale)), d, endpoint=False
        )
        position = jnp.arange(max_len, dtype=jnp.int32)
        sinusoid_inp = jnp.float32(position[:, jnp.newaxis]) / timescale[jnp.newaxis, :]
        sin = jnp.sin(sinusoid_inp)
        cos = jnp.cos(sinusoid_inp)
        return RopeTable(sin=sin, cos=cos)

    def apply(self, rearrange_spec, x):
        x1, x2 = jnp.split(x, 2, axis=-1)
        sin = einops.rearrange(self.sin, rearrange_spec)
        cos = einops.rearrange(self.cos, rearrange_spec)
        r1 = x1 * cos - x2 * sin
        r2 = x2 * cos + x1 * sin
        return jnp.append(r1, r2, axis=-1)


@typechecked
def rms_norm(x: bf16[b"batch/d len M"]) -> bf16[b"batch/d len M"]:
    mean2 = save_for_backward(
        jnp.mean(jax.lax.square(jnp.float32(x)), axis=-1, keepdims=True)
    )
    return jnp.bfloat16(x * jax.lax.rsqrt(mean2 + 1e-6))


@pytree_dataclass
class Metrics:
    loss: f32[b""]
    learning_rate: f32[b""]
    grad_norm: f32[b""]
    raw_grad_norm: f32[b""]


@dataclass(frozen=True)
class TrainingHparams:
    adam_b1: float
    adam_b2: float
    adam_eps: float
    adam_eps_root: float
    weight_decay: float
    warmup_steps: int
    steps: int
    steps_for_lr: int
    cosine_learning_rate_final_fraction: float
    learning_rate: float
    tokens: TokenBatchParams
    seed: int
    queue: Optional[str] = None


@pytree_dataclass
class State:
    weights: Model
    adam_mu: Model
    adam_nu: Model

    @staticmethod
    def init(hparams: Hparams, rng: PRNGKey) -> "State":
        weights = Model.init(hparams, rng)
        adam_mu = jax.tree.map(lambda p: p * 0.0, weights)
        adam_nu = jax.tree.map(lambda p: p * 0.0, weights)
        return State(weights=weights, adam_mu=adam_mu, adam_nu=adam_nu)


@partial(jax.jit, static_argnums=(2, 3), donate_argnums=(0,))
def training_step(
    state: State,
    step: u32[b""],
    h: Hparams,
    hparams: TrainingHparams,
    batch: TokenBatch,
) -> Tuple[Any, Metrics]:
    # check_rep=False for https://github.com/google/jax/issues/20335
    @partial(shardtypes.typed_shard_map, check_rep=False)
    def sharded_step(
        state: State, step: u32[b""], batch: TokenBatch
    ) -> Tuple[State, Metrics]:
        # TODO: Very ugly.
        loss, grad = jax.value_and_grad(lambda weights: weights.loss(h, batch))(
            state.weights
        )
        # Gradients have already been reduced across chips because the gradient of the weight `all_gather`
        # is weight-gradient `psum_scatter`. Loss, on the other hand, hasn't been reduced across chips: if we
        # did that inside the autodiff, we'd be double-reducing the loss, effectively multiplying it by the
        # amount of data parallelism.
        #
        # So we reduce the loss across chips _outside_ the autodiff.
        loss = jax.lax.psum(loss, ("d", "t"))

        # Other than global-norm of gradients, no other communication is needed during the weight update,
        # because weights and grads are already fully sharded, as checked below.

        # Calculate learning rate from step number.
        # We use linear warmup then cosine decay. See https://arxiv.org/pdf/2307.09288.pdf section 2.2
        warmup_lr = (
            jnp.float32(step) / jnp.float32(hparams.warmup_steps)
        ) * hparams.learning_rate
        cosine = jnp.cos(
            jnp.pi
            * (
                jnp.float32(step - hparams.warmup_steps)
                / jnp.float32(hparams.steps_for_lr - hparams.warmup_steps)
            )
        )
        cosine_lr = hparams.learning_rate * (
            hparams.cosine_learning_rate_final_fraction
            + (1 - hparams.cosine_learning_rate_final_fraction) * (cosine * 0.5 + 0.5)
        )
        lr = jnp.where(step < hparams.warmup_steps, warmup_lr, cosine_lr)

        # AdamW optimizer with global gradient clipping.
        grad_leaves, grad_treedef = jax.tree_util.tree_flatten(grad)
        global_norm_square = jnp.float32(0.0)
        for g in grad_leaves:
            assert g.dtype == jnp.float32
            global_norm_square += jnp.sum(jax.lax.square(g))
        global_norm_square = jax.lax.psum(global_norm_square, ("d", "t"))
        global_norm = jnp.sqrt(global_norm_square)
        rescale = jnp.minimum(1.0, 1.0 / global_norm)

        new_ps = []
        new_mus = []
        new_nus = []
        for p, g, mu, nu, spec in zip(
            tree_leaves(state.weights),
            grad_leaves,
            tree_leaves(state.adam_mu),
            tree_leaves(state.adam_nu),
            tree_leaves(shardtypes.make_partition_specs(State)),
        ):
            assert shardtypes.is_fully_sharded(spec), (
                "Weight update is only correctly scaled for fully sharded weights."
            )
            # Gradient clipping
            g = g * rescale
            # Adam scaling
            mu = (1 - hparams.adam_b1) * g + hparams.adam_b1 * mu
            nu = (1 - hparams.adam_b2) * jax.lax.square(g) + hparams.adam_b2 * nu
            # We need step numbers to start at 1, not 0. Otherwise the bias correction produces NaN.
            completed_steps = step + 1
            mu_hat = mu / (1 - jnp.float32(hparams.adam_b1) ** completed_steps)
            nu_hat = nu / (1 - jnp.float32(hparams.adam_b2) ** completed_steps)
            g = mu_hat / (jnp.sqrt(nu_hat + hparams.adam_eps_root) + hparams.adam_eps)
            # Weight decay
            g += hparams.weight_decay * p
            # Learning rate
            g *= lr

            # Apply update
            new_ps.append(p - g)
            new_mus.append(mu)
            new_nus.append(nu)

        new_state = State(
            weights=jax.tree_util.tree_unflatten(grad_treedef, new_ps),
            adam_mu=jax.tree_util.tree_unflatten(grad_treedef, new_mus),
            adam_nu=jax.tree_util.tree_unflatten(grad_treedef, new_nus),
        )
        metrics = Metrics(
            loss=loss,
            learning_rate=lr,
            grad_norm=global_norm * rescale,
            raw_grad_norm=global_norm,
        )
        return new_state, metrics

    return sharded_step(state, step, batch)


@dataclass(frozen=True)
class Paths:
    root_working_dir: str
    model_name: str


@dataclass(frozen=True)
class MeshConfig:
    d: int
    t: int


@hydrated_dataclass(
    wandb.init, frozen=True, zen_partial=True, populate_full_signature=True
)
class WandbConfig:
    # entity: str
    # project: str
    # run_name: str
    group: str = os.environ["SLURM_JOB_ID"]


@dataclass(frozen=True)
class Config:
    model: Hparams
    training: TrainingHparams
    paths: Paths
    num_hosts: int
    checkpoint_interval: int
    mesh: MeshConfig
    io: training_io.IOConfig

    dataset: FlatTokensParams | HuggingFaceDataParams

    logger: WandbConfig | None = dataclasses.field(default_factory=WandbConfig)

    # flat_tokens: Optional[FlatTokensParams] = None
    # hf_dataset: Optional[HuggingFaceDataParams] = None

    def __post_init__(self):
        assert self.flat_tokens is not None or self.hf_dataset is not None, (
            "Must provide either flat_tokens or hf_dataset."
        )
        assert not (self.flat_tokens is not None and self.hf_dataset is not None), (
            "Should not specify both flat_tokens and hf_dataset."
        )

    @property
    def flat_tokens(self) -> FlatTokensParams | None:
        return self.dataset if isinstance(self.dataset, FlatTokensParams) else None

    @property
    def hf_dataset(self) -> HuggingFaceDataParams | None:
        return self.dataset if isinstance(self.dataset, HuggingFaceDataParams) else None

    @cached_property
    def training_data(self):
        return self.dataset


def setup_logging(local_rank: int, num_processes: int, verbose: int):
    logging.basicConfig(
        level=logging.INFO,
        # Add the [{local_rank}/{num_processes}] prefix to log messages
        format=f"[%(asctime)s][%(levelname)s][{local_rank + 1}/{num_processes}] %(message)s",
        handlers=[rich.logging.RichHandler()],
        force=True,
    )
    if verbose == 0:
        log.setLevel(logging.ERROR)
    elif verbose == 1:
        log.setLevel(logging.WARNING)
    elif verbose == 2:
        log.setLevel(logging.INFO)
    else:
        assert verbose >= 3
        log.setLevel(logging.DEBUG)


def collate_fn(sequences: dict[str, np.ndarray], batch_size: int, max_seq_len: int):
    flat_batch = np.zeros(batch_size * max_seq_len, np.uint32)
    flat_is_start = np.zeros(batch_size * max_seq_len, np.bool_)
    start = 0
    for seq in sequences:
        seq = seq["input_ids"][0]
        end = min(start + len(seq), len(flat_batch))
        flat_is_start[start] = True
        flat_batch[start:end] = seq[: end - start]
        start += len(seq)
        if start >= len(flat_batch):
            break
    shape = (batch_size, max_seq_len)
    return flat_batch.reshape(shape), flat_is_start.reshape(shape)


# todo: the type hints here SUCK. Not using it yet.
# @auto_config
# def make_config(world_size: int):
#     return Config(...)


# TODO: Create a single-node job with only CPUs first to preprocess the dataset,
# then a multi-node job to train.


@contextlib.contextmanager
def global_main_process_first(*, global_rank: int, name: str = "sync"):
    if global_rank == 0:
        yield
        log.debug(f"Entering '{name}' barrier.")
        sync_global_devices(name)
        log.debug(f"Exiting '{name}' barrier.")
    else:
        log.debug(f"Entering '{name}' barrier.")
        sync_global_devices(name)
        log.debug(f"Exiting '{name}' barrier.")
        yield


@contextlib.contextmanager
def local_main_process_first(*, local_rank: int, name: str = "sync"):
    if local_rank == 0:
        yield
        sync_global_devices(name)
    else:
        sync_global_devices(name)
        yield


def get_tokenized_dataset(dataset_config: HuggingFaceDataParams, data_dir: Path):
    dataset_config_hash = hashlib.md5(str(dataset_config).encode()).hexdigest()
    tokenized_path = data_dir / f"tokenized_dataset_{dataset_config_hash}"
    log.debug(f"Dataset config: {dataset_config}")

    log.info(f"Getting pretrained tokenizer {dataset_config.tokenizer}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(dataset_config.tokenizer)
    max_token_id = tokenizer.vocab_size - 1

    num_proc = len(os.sched_getaffinity(0))

    if tokenized_path.exists():
        log.info(f"Using pre-tokenized dataset from {tokenized_path}")
        tokenized_dataset = datasets.load_from_disk(str(tokenized_path))
        return tokenized_dataset, max_token_id

    log.info(f"Tokenized dataset not found at {tokenized_path}.")

    # TOOD: WHY is this necessary?
    assert 0 in tokenizer.all_special_ids, "Tokenizer must have a special 0 token"

    # setup an iterator over the dataset
    tokenize = functools.partial(
        tokenizer,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_tensors="np",
    )

    log.info("Loading dataset.")
    dataset = datasets.load_dataset(
        dataset_config.path, dataset_config.name, split="train", num_proc=num_proc
    )
    log.info("Tokenizing dataset...")
    dataset = dataset.select_columns(["text"])

    assert isinstance(dataset, datasets.arrow_dataset.Dataset), type(dataset)
    tokenized_path.parent.mkdir(parents=True, exist_ok=True)

    tokenized_dataset = dataset.map(
        tokenize,
        input_columns=["text"],
        remove_columns=["text"],
        # todo: make sure this is all cached properly.
        cache_file_name=str(tokenized_path.with_suffix(".cache")),
        load_from_cache_file=True,
        num_proc=num_proc,
    )
    # TODO: Do this in a smart, distributed, multi-node fashion somehow?
    tokenized_dataset.save_to_disk(str(tokenized_path))
    return tokenized_dataset, max_token_id


def get_dataloader(config: Config, distributed_env: SlurmDistributedEnv | None):
    assert config.hf_dataset

    with (
        global_main_process_first(global_rank=distributed_env.global_rank)
        if distributed_env
        else contextlib.nullcontext()
    ):
        tokenized, max_token_id = get_tokenized_dataset(
            config.hf_dataset, data_dir=SCRATCH / "data"
        )

    batch_size = config.training.tokens.batch
    max_seq_len = config.training.tokens.len
    # # todo: simplify, replace with a hard-coded value to start.
    # sharding = shardtypes.make_shardings(TokenBatch).targets

    dataloader = torch.utils.data.DataLoader(
        tokenized,
        num_workers=config.hf_dataset.num_workers,
        collate_fn=functools.partial(
            collate_fn, batch_size=batch_size, max_seq_len=max_seq_len
        ),
        drop_last=True,
        batch_size=config.hf_dataset.sequences_packed_per_batch,
    )
    return dataloader, max_token_id


@hydra.main(config_path="configs", version_base="1.2")
def main(raw_config: omegaconf.DictConfig):
    distributed_env = SlurmDistributedEnv()
    config: Config = instantiate(raw_config)
    # this apparently doesn't work (the configs are instantiated as
    # train.Config), which is somehow not the same as `Config`?

    assert isinstance(config, Config), (config, type(config), Config)
    # BUG: Weird Hydra bug, this somehow isn't correct?! (train.Config vs __main__.Config)
    # todo: This is super weird.
    # config = make_dataclass_from_dict(Config, raw_config)

    # todo: Play around with different d / t ratios.
    config = dataclasses.replace(
        config,
        mesh=MeshConfig(
            d=distributed_env.num_tasks * distributed_env.gpus_per_task, t=1
        ),
    )
    setup_logging(distributed_env.global_rank, distributed_env.num_tasks, 3)

    if config.logger:
        # assert isinstance(config.logger, functools.partial), config.logger
        wandb_logger = hydra_zen.instantiate(config.logger)
        # assert isinstance(wandb_logger, wandb.wandb_run.Run)
        assert isinstance(wandb_logger, functools.partial)
        wandb_logger = wandb_logger()
        assert isinstance(wandb_logger, wandb.wandb_run.Run)
    else:
        wandb_logger = None

    # Rewrite of `main` / `main_contained`.
    assert distributed_env.gpus_per_task > 0, os.environ["SLURM_TRES_PER_TASK"]
    # Using `jax.distributed.initialize` shows an import warning in VsCode+Pylance saying
    # it isn't defined there.
    # BUG: Seems like it's necessary to pass this list otherwise the GPUS aren't split correctly.
    jax_distributed_initialize(
        local_device_ids=list(range(distributed_env.gpus_per_task))
    )

    log.info("Initialized.")
    if distributed_env.global_rank == 0:
        print("Distributed Config:")
        print(yaml.dump(dataclasses.asdict(distributed_env), indent=2))
        print("Config:")
        print(yaml.dump(dataclasses.asdict(config), indent=2))
        # todo: wandb init here.

    # what is this? Why is it here?
    # maybe this? https://github.com/jax-ml/jax/issues/17982
    jax.config.update("jax_threefry_partitionable", True)
    mesh = Mesh(
        mesh_utils.create_device_mesh([config.mesh.d, config.mesh.t], jax.devices()),
        ("d", "t"),
    )

    if config.hf_dataset:
        dataloader, max_token_id = get_dataloader(config, distributed_env)
        loader = iter(itertools.repeat(dataloader))
    else:
        assert config.flat_tokens
        log.info(f"Loading flat tokens from {config.flat_tokens.filespec}")
        loader = ShufflingLoader(
            "train", config.flat_tokens, config.training.tokens, mesh=mesh
        )
        max_token_id = loader.max_token_id

    assert isinstance(max_token_id, int)
    assert config.model.vocab > max_token_id, f"{config.model.vocab} vs {max_token_id}"
    model_dir = Path(config.paths.root_working_dir) / config.paths.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    mesh.__enter__()  # hacky, but nicer than indenting all the code below.
    # TODO: Pass `mesh` as an argument instead of the weird global variable hack
    # that is happening in shardlib.

    root_rng = jax.random.PRNGKey(config.training.seed)

    # todo: does this have to be in the mesh context block?
    # loader = get_loader("train", config.training_data, config.training.tokens)
    state = jax.jit(State.init, static_argnums=0)(
        config.model, fold_in_str(root_rng, "init")
    )

    # TODO: Can we just remove this for now? Otherwise we have to save the dataloader state explicitly.
    state, start_step = training_io.load_checkpoint_if_it_exists(
        str(model_dir), state, config.io
    )
    log.info(f"Starting training at step {start_step}.")

    if isinstance(loader, ShufflingLoader):
        sample_batch = loader.load(0)
    else:
        sample_batch = next(loader)

    # Explicitly compile training step, to record XLA HLO graph.
    # See https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev
    c_training_step = training_step.lower(
        state, jnp.uint32(0), config.model, config.training, sample_batch
    ).compile()
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    training_io.save_hlo_svg(
        os.path.join(model_dir, f"training_step_optimized_hlo_{date}.svg"),
        c_training_step,
    )
    profile_start = None

    for step in range(start_step, config.training.steps):
        if step % config.checkpoint_interval == 0 and step > start_step:
            training_io.save_checkpoint(str(model_dir), step, state, config.io)

        # We profile on the second step, because the first step has a long pause for XLA
        # compilation and initial shuffle buffer loading.
        if jax.process_index() == 0 and step == start_step + 1:
            jax.block_until_ready(state)
            training_io.start_profile()
            profile_start = time.time()

        # TODO: Need to do the equivalent of `loader.load(step)` with a dataloader.
        if isinstance(loader, ShufflingLoader):
            # trick to allow repeating the dataset. Apparently not supported?!
            batch = loader.load(step % loader.step_count)
        else:
            batch = next(loader)

        state, output = c_training_step(state, jnp.uint32(step), batch)

        # Run profile for two steps, to include data loading time in between them.
        if jax.process_index() == 0 and step == start_step + 2:
            jax.block_until_ready(state)
            assert profile_start is not None
            profile_duration = time.time() - profile_start
            training_io.stop_profile(model_dir)

            # Print MFU, including (one step of) data loading time.
            print(f"Profile time: {profile_duration}s for 2 steps.")
            model_params = jax.tree.reduce(
                operator.add, jax.tree.map(lambda w: w.size, state.weights)
            )
            assert isinstance(loader, ShufflingLoader)
            tokens = loader.load(step % loader.step_count).targets.size
            print(f"Model params: {model_params:_}")
            print(f"Tokens: {tokens:_}")
            device_flops = training_io.get_flops_per_device()
            num_devices = jax.device_count()
            print(
                f"MFU (projections only): {100 * (2 * 6 * model_params * tokens / (num_devices * profile_duration)) / device_flops:.2f}% MFU"
            )

        training_io.log(step=step, logger=wandb_logger, output=output)
    mesh.__exit__(None, None, None)


if __name__ == "__main__":
    main()

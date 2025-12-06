# Manual op tuning via tune-jax

## Quickstart

The individual scripts will tune the hyperparameters for these JAX ops via
[tune-jax](https://github.com/rdyro/tune-jax) - check out the README there for
more details.

The scripts are op-specific and will tune several versions of the op (e.g.,
forward and backward mode in splash attention). They will also print the full
tuning results, not just the optimal hyperparameters, in a table sorted by
runtime.

For details on how the found hyperparameters can be specified in each op, take a
look at the benchmarking script directly, they're thin wrappers.

## TPU ragged dot benchmarking script

The core benchmarking script is `ragged_dot_benchmarking.py` which takes FLAGS
for input shapes and other configs.

```bash
--num_samples=100  # upper bound on the number of samples to tune
--m=$(( 4096 * 8 ))  # number of inputs rows in gmm
--k=7168  # reduction dimension in gmm
--n=2048  # output dimension in gmm
--g=256  # number of groups/experts
--tune_modes # either "all", "fwd", "dlhs" or "drhs"
--qdtype  # either "none" for no quantization or a valid string name jax dtype
```

Run with `python3 ragged_dot_benchmarking.py`.

For each of its variants (gmm, gmm with rhs transposed and tgmm), ragged dot
takes 3 separate tile sizes and 1 input buffer count (controlling the memory
pipeline buffering)

## Splash Attention benchmarking scripts

The core benchmarking script is `splash_attention_benchmarking.py` which takes
FLAGS for input shapes and other configs to attention.

```bash
--num_samples  # how many samples to use for tuning
--q_seq_len
--kv_seq_len
--q_heads
--kv_heads
--qk_head_dim
--v_head_dim
--batch_size
--mode  # either `fwd`, `bwd` or `combined`
        # `bwd` only makes sense with --tune_pallas_only
        # `combined` only makes sense with --notune_pallas_only
--dtype  # input dtypes
--mask_name  # either `causal_mask` or `full_mask`
--(no)tune_pallas_only  # whether to tune based on the pallas calls alone
```

Run with `python3 splash_attention_benchmarking.py`.

The splash attention operator takes in a config where the found hyperparameters
can be specified.
# Packed input windows for non-CP KDA forward

This increment adapts the packed-window loading strategy from
[`primatrix/pallas-kernel` c82625a2bdadd5881b12f1ab37ec3fa2004c9780](https://github.com/primatrix/pallas-kernel/blob/c82625a2bdadd5881b12f1ab37ec3fa2004c9780/tops/ops/kda/chunk_fwd_mega.py)
to the existing fused Tokamax compute bodies. It follows forward fusion,
saved-state backward fusion, and packed-gradient correctness fixes.

## Dispatch and layouts

`Config(packed_forward=True)` enables direct packed input windows for
non-CP, fused forward with K/V divisible by 128. The option defaults to false
pending TPU validation. Fixed-length, CP, other widths and disabled forward
fusion retain existing dispatch. The public KDA API is unchanged.

The chunk grid still uses aligned sequence boundaries. Original boundaries
map each aligned chunk to its packed start. Each input BlockSpec reads 72
rows from an 8-row-aligned origin, followed by a 0..7-row static shift to
select the 64-token chunk. A guard allocation covers final-window reads.
Sequence-tail tokens become zero in VMEM; raw gates use the same -1e4
neutral sentinel as existing preprocessing. Q/K normalization remains
outside the kernel and runs on packed input values before window loading.

Output, Aqk/Akk, gate-prefix and saved-state layouts remain aligned. The
existing backward still consumes its original aligned residuals, including
Q/K normalization statistics. Thus training still retains aligned inputs;
this PR does not claim their elimination from the training memory footprint.
In a jitted forward-only call, unused aligned input preparation can be
eliminated by compiler dead-code elimination. Compilation and device-time
measurements must verify the resulting benefit.

The launcher adds a conservative allowance for packed windows and VMEM
staging refs to its hardware-derived head-group budget. It delegates DMA
pipelining to Pallas BlockSpecs rather than importing the source's manual
double-buffered DMA loop into this increment.

## Validation

Focused tests compare packed/aligned forward-only calls, VJPs and complete
gradient trees in BF16/FP32 and saved/recomputed-state modes. Boundary tests
visit every window offset, use non-multiple-of-eight input lengths, omit
initial/final states, and cover multiple head groups, batches and an empty
batch. Fixed-length calls explicitly verify fallback dispatch.

```bash
python -m pytest tokamax/_src/ops/kda/pallas_mosaic_tpu_packed_test.py -v
python -m tokamax.benchmarks.kda --skip_implementations=xla,mosaic_staged,mosaic_fwd_fused
```

Compare `mosaic_packed_inputs` with `mosaic_fused`: all other fusion options
are identical. Benchmark both forward and forward+VJP with the default
device timer and report hardware/JAX/JAXLIB/libtpu versions. CPU interpretation
is not TPU lowering or a performance measurement. TPU compilation, allocation,
device correctness and timings remain required before enabling this by default.

Packed output stores, native packed backward and CP backward fusion remain
separate increments. This PR changes the forward input boundary only.

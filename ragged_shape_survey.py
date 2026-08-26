"""Which battery shapes `mosaic_tiling.plan` blocks, now that every axis is tiled exactly."""
from tokamax._src.ops.normalization import mosaic_tiling as mt
from tokamax._src.ops.normalization import pallas_triton_config as tc
from tokamax._src.ops.normalization import arg_specs

# The CI battery: `test_base` shapes at every axis it tests, plus the vmap ones.
ci = [((64,), -1), ((128, 32), -1), ((8, 128, 32), -1), ((256, 42), -1),
      ((24, 32, 40), 0), ((24, 32, 40), 1), ((24, 32, 40), 2),
      ((3, 128, 32), -1), ((128, 32), -1)]
bench = []
for s in arg_specs.ARG_SPECS:
  x, axis = s.args['x'], s.args['axis']
  bench.append(((x.shape, axis), x.dtype.itemsize))
  bench.append((((8, *x.shape), axis + 1 if axis >= 0 else axis), x.dtype.itemsize))

print(f'{"shape":>22} {"axis":>4} {"canonical":>22} {"block":>16} {"grid":>16} {"regs":>5}')
for group, items in (('CI (float32)', [(c, 4) for c in ci]), ('benchmarks', bench)):
  print(f'--- {group}')
  for (shape, axis), itemsize in items:
    canon = tc.canonicalize_shape(shape, axis)
    try:
      p = mt.plan(canon, itemsize, block_m=32, block_n=None)
    except NotImplementedError as e:
      print(f'{str(shape):>22} {axis:>4} {str(canon):>22}  declined: {e}')
      continue
    print(f'{str(shape):>22} {axis:>4} {str(canon):>22} {str(p.block):>16}'
          f' {str(p.grid):>16} {p.tile_regs():>5}')

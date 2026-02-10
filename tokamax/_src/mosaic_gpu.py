# Copyright 2026 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Mosaic-GPU utils."""

import jax.experimental.mosaic.gpu as mgpu
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import llvm
import numpy as np


def int4_as_biased_f8e4m3fn(x, layout):
  """Encodes a packed `int4` array as an `f8e4m3fn` array with an implicit +8 bias.

  Mathematically, for any int4 value `x` in the range [-8, 7], this encoding
  ensures:
      x_f32 = (biased_fp8_value * 512.0) - 8.0)

  Implementation Details:
  -----------------------
  We map the signed int4 range [-8, 7] to the unsigned range [0, 15] by flipping
  the sign bit (XOR 0x8).
  This unsigned integer bit pattern is then reinterpreted directly as FP8 bits.

  How this works:
  1. The bit patterns for integers 0-15 correspond to f8e4m3fn values with E -6.
    - 0..7 become f8e4m3fn subnormals (E=0, M=0..7)-> Value = 0.Mantissa * 2^-6.
    - 8..15 become f8e4m3fn normals (E=1, M=0..7)  -> Value = 1.Mantissa * 2^-6.
    Note that in E4M3, the transition from Subnormal Max (0.875 * 2^-6) to
    Normal Min (1.0 * 2^-6) is linear.
  2. The hardware interprets these bits as `(Integer_Value / 8) * 2^-6`.
  3. To recover the original signed number:
    3.1 Rescale the f8e4m3fn value:
        Value * 512 = (Unsigned / 2^3 * 2^-6) * 2^9
                    = Unsigned * 2^-9 * 2^9
                    = Unsigned (0..15)
    3.2 Debias the result:
        Original_Value = Unsigned - 8

  Mapping Examples with debiasing:
      -8 (1000)->XOR 8->  0 (0000)->FP8 0.000       -> * 512 = 0.0 - 8 = -8.0
      -2 (1110)->XOR 8->  6 (0110)->FP8 0.750 * 2^-6-> * 512 = 6.0 - 8 = -2.0
      0 (0000) ->XOR 8->  8 (1000)->FP8 1.000 * 2^-6-> * 512 = 8.0 - 8 =  0.0
      7 (0111) ->XOR 8-> 15 (1111)->FP8 1.875 * 2^-6-> * 512 =15.0 - 8 =  7.0


  MMA Usage & De-biasing:
  -----------------------
  This encoding allows int4 operands to be used on FP8 tensor cores as f8e4m3fn
  with less conversion ALU overhead for upcasting. The operation `D = A @ B` can
  be recovered by adjusting the accumulator based on which operands are biased.

  Let `S=512=2^9` (Scale) and `Z=8` (Bias).

  1. B is Biased (Standard A @ Int4 B):
     Decomposes to: `S * (A @ B_enc) - Z * sum(A, axis=contracting_dim)`

  2. A is Biased (Int4 A @ Standard B):
     Decomposes to: `S * (A_enc @ B) - Z * sum(B, axis=contracting_dim)`

  3. Both Biased (Int4 A @ Int4 B):
     Decomposes to:
     `S^2(A_enc @ B_enc) - S*Z(row_sums(A_enc)) - S*Z(col_sums(B_enc)) + Z^2*K`
     where K is the size of the contraction dimension.

  *Note: For block scaling (MXFP8), the scale S=512 can be absorbed into the
  `ue8m0` exponent format scales.*

  Args:
    x: A `jax.Array` containing packed int4 values.
    layout: The `plgpu.Layout` of the fragmented array `x`.

  Returns:
    A `mgpu.FragmentedArray` containing the encoded values in biased `f8e4m3fn`
    format with the specified layout.
  """

  @plgpu.inline_mgpu(
      arg_types=(layout,),
      return_type=plgpu.ShapeDtypeStruct(x.shape, jnp.float8_e4m3fn, layout),
  )
  def encode(_, fa: mgpu.FragmentedArray):
    [vector_len] = ir.VectorType(fa.registers.flat[0].type).shape
    if vector_len != 8:
      raise NotImplementedError("Only vector_length=8 is supported")
    i32 = ir.IntegerType.get_signless(32)
    i8 = ir.IntegerType.get_signless(8)
    new_registers = np.empty_like(fa.registers)
    f8e4m3fn = ir.Float8E4M3FNType.get()
    out_vec_ty = ir.VectorType.get((vector_len,), f8e4m3fn)

    def upcast_i4_to_fp8_biased(reg: ir.Value):

      out_struct = llvm.inline_asm(
          ir.Type.parse("!llvm.struct<(i32, i32)>"),
          [reg],
          """
            {{
            .reg .b32 biased, evens, odds, odds_raw;
            xor.b32 biased, $2, 0x88888888;
            and.b32 evens, biased, 0x0F0F0F0F;
            shr.u32 odds_raw, biased, 4;
            and.b32 odds, odds_raw, 0x0F0F0F0F;
            prmt.b32 $0, evens, odds, 0x5140;
            prmt.b32 $1, evens, odds, 0x7362;
            }}
            """,
          "=r,=r,r",
      )

      lo_i32 = llvm.extractvalue(i32, out_struct, (0,))
      hi_i32 = llvm.extractvalue(i32, out_struct, (1,))

      vec_ty_4 = ir.VectorType.get((4,), i8)
      lo_vec = mgpu.utils.bitcast(lo_i32, vec_ty_4)
      hi_vec = mgpu.utils.bitcast(hi_i32, vec_ty_4)

      return mgpu.utils.vector_concat([lo_vec, hi_vec])

    for idx, reg in np.ndenumerate(fa.registers):
      reg_as_int = mgpu.utils.bitcast(reg, i32)
      new_registers[idx] = mgpu.utils.bitcast(
          upcast_i4_to_fp8_biased(reg_as_int), out_vec_ty
      )

    return mgpu.FragmentedArray(
        _registers=new_registers, _layout=layout.to_mgpu(), _is_signed=None
    )

  return encode(x)

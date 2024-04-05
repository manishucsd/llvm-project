import numpy as np
import enum
from enum import auto
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import func
from mlir.dialects import gpu
from mlir.dialects import memref
from mlir.dialects import nvgpu
from mlir.dialects import nvvm
from mlir.dialects import llvm
from mlir.dialects import builtin
from mlir.dialects import scf
from mlir.dialects import vector
from mlir.extras import types as T

DEBUG = True


######################################################################
# Helper functions
######################################################################
def debug_print(fmt, *args, predicate=None, threadNumber=-1, forcePrint=False):
    if not DEBUG and not forcePrint:
        return
    type_formats = []
    for arg in args:
        ty_format = None
        if ir.IndexType.isinstance(arg.type):
            ty_format = "%llu"
        if ir.IntegerType.isinstance(arg.type):
            width = ir.IntegerType(arg.type).width
            if width == 64:
                ty_format = "%llu"
            elif width == 32:
                ty_format = "%d"
            elif width == 1:
                ty_format = "%i"
        if ir.F32Type.isinstance(arg.type):
            ty_format = "%f"
        if ty_format is None:
            raise NotImplementedError(arg.type)
        type_formats.append(ty_format)
    if threadNumber != -1:
        tidx = gpu.thread_id(gpu.Dimension.x)
        predicate = arith.cmpi(arith.CmpIPredicate.eq, tidx, c(threadNumber))
        scf.yield_([])
    if_op = scf.IfOp(predicate)
    with ir.InsertionPoint(if_op.then_block):
        gpu.printf(fmt.format(*type_formats) + "\n", args)
        scf.yield_([])

def get_size_in_bits(ty):
  """Returns the size of the data type in bits."""
  if ir.FloatType.isinstance(ty):
      return ir.FloatType(ty).width
  if ir.IntegerType.isinstance(ty):
      return ir.IntegerType(ty).width
  raise NotImplementedError(ty)

def get_type_size(ty):
  """Returns the size of the data type in bytes."""
  return get_size_in_bits(ty) // 8


def get_mlir_ty(dtype):
  """Returns the MLIR type for the given numpy dtype."""
  if dtype == np.float16:
      return T.f16()
  if dtype == np.float32:
      return T.f32()
  if dtype == np.float64:
      return T.f64()
  if dtype == np.int32:
      return T.i32()
  if dtype == np.int64:
      return T.i64()
  raise NotImplementedError(dtype)

def get_numpy_dtype(mlir_ty):
  """Returns the numpy dtype for the given MLIR type."""
  if T.F16Type.isinstance(mlir_ty):
      return np.float16
  if T.F32Type.isinstance(mlir_ty):
      return np.float32
  if T.F64Type.isinstance(mlir_ty):
      return np.float64
  if T.I32Type.isinstance(mlir_ty):
      return np.int32
  if T.I64Type.isinstance(mlir_ty):
      return np.int64
  raise NotImplementedError(mlir_ty)

######################################################################
# Enums
######################################################################

class LayoutType(enum.Enum):
  ColumnMajor = auto()
  RowMajor = auto()


# cuBLAS/cuDNN layout type names convention is followed for the layout names.
# https://docs.nvidia.com/cuda/cublas/index.html#cublasoperation-t
ShortLayoutTypeName = {
    LayoutType.ColumnMajor: "n",
    LayoutType.RowMajor: "t",
}

# Layout names used for nvvm attributes.
LayoutTypeName = {
    LayoutType.ColumnMajor: "col",
    LayoutType.RowMajor: "row",
}
######################################################################


# Note that WGMMA swizzle enum values are different from the ones in the
# used for TMA swizzle. Thus, we need to define a separate enum for WGMMA
# swizzle.
class GmmaSwizzle(enum.Enum):
  Interleave = 0
  Swizzle128B = auto()
  Swizzle64B = auto()
  Swizzle32B = auto()

GmmaSwizzleName = {
   GmmaSwizzle.Interleave: "interleave",
   GmmaSwizzle.Swizzle128B: "swizzle128b",
   GmmaSwizzle.Swizzle64B: "swizzle64b",
   GmmaSwizzle.Swizzle32B: "swizzle32b",
}

######################################################################

# C++ uses this value to understand whether it's dynamic or not.
MLIR_DYNAMIC = -9223372036854775808

######################################################################
class SmArchTag(enum.Enum):
  Sm80 = 80
  Sm90 = 90

SmArchTagName = {
    SmArchTag.Sm80: "ampere",
    SmArchTag.Sm90: "hopper",
}

SharedMemCapacityPerSM = {
    80: 163,  # 163KB of SMEM - 1KB reserved for the driver
    90: 227,  # 227KB of SMEM - 1KB reserved for the driver
}
######################################################################

class MainloopVariant(enum.Enum):
  """
  Mainloop variants for the mainloop that performs the GEMM computation.
  """
  Singlestage = auto()                  # for debugging purposes
  Multistage = auto()
  WarpSpecializedCooperative = auto()
  WarpSpecializedPingPong = auto()

ShortMainloopVariantName = {
  MainloopVariant.Singlestage: "singlestage",
  MainloopVariant.Multistage: "multistage",
  MainloopVariant.WarpSpecializedCooperative: "warp_specialized_cooperative",
  MainloopVariant.WarpSpecializedPingPong: "warp_specialized_pingpong",
}
######################################################################


class TensorDescription:
  """
  A class for tensor description capturing tensor datatype and layout.
  """

  def __init__(self, datatype, layout):
    self.datatype = datatype  # mlir built-in type
    self.layout = layout      # LayoutType enum

  def name(self):
    return "%s%s" % (self.datatype, ShortLayoutTypeName[self.layout])
######################################################################
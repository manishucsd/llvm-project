from mlir import ir
from mlir.dialects import arith


class Builder:
  """
  Top-level builder class contains kernel-agnositc peices of the kernel builder.
  Specifically, the Builder owns mlir.context, built-in types, and attributes.

  * Guidlines for writing IR builders *
   -  Create `mlir` types and attributes under the singular 
      ir.Context() object. The `mlir` types and attributes suffixed 
      with `_ty` with `_attr`, respectively. We separate the creation
      of types and attributes from the operations for the following:
      (a.) GEMM Kernel-agnositc: created in the `Builder`.
      (b.) GEMM Kernel-specific: created in the `KernelModuleBuilder`.
   - Create all `mlir` operations suffixed with `_op` under the
      module object and are inserted under the insertion point 
      of the `module` in the `KernelModuleBuilder` __call__.
  """

  def __init__(self):

    # create builtin datatypes
    self.f16_ty = ir.F16Type.get()
    self.f32_ty = ir.F32Type.get()
    self.f64_ty = ir.F64Type.get()
    self.index_ty = ir.IndexType.get()
    self.i1_ty = ir.IntegerType.get_signless(1)
    self.i8_ty = ir.IntegerType.get_signless(8)
    self.i16_ty = ir.IntegerType.get_signless(16)
    self.i32_ty = ir.IntegerType.get_signless(32)
    self.i64_ty = ir.IntegerType.get_signless(64)

    # create llvm pointer types
    self.llvm_smem_ptr_ty = ir.Type.parse("!llvm.ptr<3>")

    # create attributes
    self.smem_addr_space_attr = ir.Attribute.parse("#gpu.address_space<workgroup>")

    # addtional types
    self.gpu_token_ty = ir.Type.parse("!gpu.async.token")


  def const_index_op(self, value):
    """Returns an arithmetic constant op (index type)."""
    return arith.ConstantOp(self.index_ty, ir.IntegerAttr.get(self.index_ty, value))

  def const_int_op(self, value, int_ty):
    """Returns an arithmetic constant op (integer type)."""
    return arith.ConstantOp(int_ty, ir.IntegerAttr.get(int_ty, value))
######################################################################
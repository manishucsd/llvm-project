# RUN:   %PYTHON %s 

import numpy as np
import pathlib
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import func
from mlir.dialects import gpu
from mlir.dialects import memref
from mlir.dialects import nvgpu
from mlir.dialects import nvvm
from mlir.dialects import scf
from mlir.dialects import vector
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager

DYNAMIC = -9223372036854775808

with ir.Context() as ctx, ir.Location.unknown():
  
  f32_ty = ir.F32Type.get()
  index_ty = ir.IndexType.get()
  async_token_ty = ir.Type.parse("!gpu.async.token")

  module = ir.Module.create()
  with ir.InsertionPoint(module.body):

    @func.FuncOp.from_py_func()
    def main():
      a_ty = b_ty = c_ty = ir.MemRefType.get([128], f32_ty)
      a_host = memref.AllocOp(a_ty, [], [])
      b_host = memref.AllocOp(b_ty, [], [])
      c_host = memref.AllocOp(c_ty, [], [])

      token = gpu.WaitOp(async_token_ty, [])
      a_device = gpu.AllocOp(a_ty, async_token_ty, [token], [], []).results[0]
      b_device = gpu.AllocOp(b_ty, async_token_ty, [token], [], []).results[0]
      c_device = gpu.AllocOp(c_ty, async_token_ty, [token], [], []).results[0]

      gpu.MemcpyOp(async_token_ty, [token], a_device, a_host)
      gpu.MemcpyOp(async_token_ty, [token], b_device, b_host)

  main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
  module.operation.verify()
  print(module)

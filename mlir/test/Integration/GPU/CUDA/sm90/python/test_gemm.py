import errno
import numpy as np
import subprocess
import ctypes
from tools import nvgpucompiler
from tools import builder
from tools.gemm import *
from tools.utils import *
from tools.gemm_builder import *
import contextlib
import os
import sys
import pathlib
import ctypes
from mlir import ir
from mlir import runtime as rt

problem_shape = GemmShape(128, 128, 512)
with ir.Context() as ctx, ir.Location.unknown():
  b = builder.Builder()
  # Create a GEMM description
  gemm_desc = GemmDescription(TensorDescription(b.f16_ty, LayoutType.RowMajor),    # a or lhs tensor
                              TensorDescription(b.f16_ty, LayoutType.RowMajor),    # b or rhs tensor
                              TensorDescription(b.f32_ty, LayoutType.RowMajor),    # c or result tensor
                              b.f32_ty,                                            # accumulation element datatype
                              problem_shape,                                       # gemm problem shape (in number of elements)
                              GemmShape(1, 1, 1),                                  # cga shape (in number of cta)
                              GemmShape(128, 128, 64),                             # cta shape (in number of elements)
                              GemmShape(4, 1, 1),                                  # warp shape (in number of warps)
                              GemmShape(64, 128, 16),                              # wgmma shape (int number of elements)
                              3,                                                   # number of smem stages
                              MainloopVariant.Multistage,
                              SmArchTag.Sm90)
  
  # Create a GEMM kernel module
  gemm_module_builder = GemmKernelBuilder(gemm_desc)
  gemm_module = gemm_module_builder()

  # Compile the GEMM module
  support_lib = os.getenv("SUPPORT_LIB")
  if not os.path.exists(support_lib):
      raise FileNotFoundError(
          errno.ENOENT, os.strerror(errno.ENOENT), support_lib)
  
  options=f"cubin-chip=sm_90a cubin-features=+ptx80 opt-level=3"
  compiler = nvgpucompiler.NvgpuCompiler(
      options, opt_level=3, shared_libs=[support_lib]
  )

  engine = compiler.compile_and_jit(gemm_module)
  
  # Allocate input/output tensors
  dtype_a = get_numpy_dtype(gemm_desc.a.datatype)
  dtype_b = get_numpy_dtype(gemm_desc.b.datatype)
  dtype_c = get_numpy_dtype(gemm_desc.c.datatype)

  host_c = np.zeros((gemm_desc.problem_shape.m, gemm_desc.problem_shape.n), dtype_c)
  host_a = np.random.randn(gemm_desc.problem_shape.m, gemm_desc.problem_shape.k).astype(dtype_a)
  host_b = np.random.randn(gemm_desc.problem_shape.k, gemm_desc.problem_shape.n).astype(dtype_b)
  a_ptr = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(host_a)))
  b_ptr = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(host_b)))
  c_ptr = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(host_c)))

  # Launch the GEMM kernel
  engine.invoke(gemm_desc.name(), a_ptr, b_ptr, c_ptr)  


  
  # Run the reference
  ref_c = host_a.astype(dtype_a) @ host_b.astype(dtype_b)

  float_formatter = "{:.2f}".format
  np.set_printoptions(formatter={"float_kind": float_formatter})
  print("host_c (computed)")
  print(host_c)
  print("ref_c (reference)")
  print(ref_c)

  # Check the result
  np.testing.assert_allclose(host_c, ref_c, rtol=5e-03, atol=1e-01)



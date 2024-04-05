import numpy as np
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
from utils import *
from builder import Builder

class TmaDescriptorBuilder:
  """A class that builds a TMA descriptor."""

  def __init__(self, swizzle, l2promo, oob, interleave, 
               tma_box_shape, memref_ty):
    self.swizzle = swizzle       # mlir.nvgpu.TensorMapSwizzleKind
    self.l2promo = l2promo       # mlir.nvgpu.TensorMapL2PromoKind
    self.oob = oob               # mlir.nvgpu.TensorMapOOBKind
    self.interleave = interleave # mlir.nvgpu.TensorMapInterleaveKind
    self.tma_box_shape = tma_box_shape
    self.memref_ty = memref_ty   # MemRefType

  @property
  def tensormap_descriptor_ty(self):
    """Returns a tensormap descriptor type."""
    memref_str = f"memref<{self.tma_box_shape[0]}x{self.tma_box_shape[1]}x{self.memref_ty.element_type}, 3>"
    parse_str = f"!nvgpu.tensormap.descriptor<tensor = {memref_str},\
                                              swizzle = {self.swizzle},\
                                              l2promo = {self.l2promo},\
                                              oob = {self.oob},\
                                              interleave = {self.interleave}>"
    return ir.Type.parse(parse_str)

  def tma_descriptor_op(self, device_ptr):
    """Returns a tensormap descriptor op."""
    b = Builder()
    tma_descriptor_ty = self.tensormap_descriptor_ty
    device_unranked_memref = memref.CastOp(
       ir.UnrankedMemRefType.get(self.memref_ty.element_type, self.memref_ty.memory_space), device_ptr)
    tma_descriptor_op = nvgpu.TmaCreateDescriptorOp(
        tma_descriptor_ty, device_unranked_memref, map(b.const_index_op,
                                                      self.tma_box_shape))
    return tma_descriptor_op.result
######################################################################
  

class GemmKernelBuilder():
  def __init__(self, gemm_description) -> None:
    self.gemm_description = gemm_description
    
    # Create kernel-specific types and attribute required for the GEMM kernel
    b = Builder()

    #
    # Global memory.
    #
    self.a_memref_ty = ir.MemRefType.get([self.gemm_description.problem_shape.m,
                                          self.gemm_description.problem_shape.k],
                                          self.gemm_description.a.datatype)
    self.b_memref_ty = ir.MemRefType.get([self.gemm_description.problem_shape.k,
                                          self.gemm_description.problem_shape.n],
                                          self.gemm_description.b.datatype)
    self.c_memref_ty = ir.MemRefType.get([self.gemm_description.problem_shape.m,
                                          self.gemm_description.problem_shape.n],
                                          self.gemm_description.c.datatype)
    #
    # Shared memory.
    #
    self.mbarrier_group_ty = ir.Type.parse(f"!nvgpu.mbarrier.group<\
                                        memorySpace={b.smem_addr_space_attr},\
                                        num_barriers={gemm_description.num_stages}\
                                      >")

    # Create smem memref type for dynamically sized shared memory 
    # for eg. memref<?xi8, #gpu.address_space<workgroup>>; Note that DYNAMIC == ?
    self.dynamic_smemref_ty =  ir.MemRefType.get((MLIR_DYNAMIC,), 
                                                 b.i8_ty, 
                                                 memory_space = b.smem_addr_space_attr)
    
    # Create tma shapes for A and B operands
    # Note as of right now, the tuple format (strided, strided, ..., contiguous)
    self.a_smem_stage_shape = (self.gemm_description.cta_shape.m, 
                               self.gemm_description.cta_shape.k)
    self.b_smem_stage_shape = (self.gemm_description.cta_shape.k, 
                               self.gemm_description.cta_shape.n)
    self.a_smem_stage_smemref_ty = ir.MemRefType.get(self.a_smem_stage_shape,
                                                     self.gemm_description.a.datatype,
                                                     memory_space = b.smem_addr_space_attr)
    self.b_smem_stage_smemref_ty = ir.MemRefType.get(self.b_smem_stage_shape,
                                                     self.gemm_description.b.datatype,
                                                     memory_space = b.smem_addr_space_attr)

    # Create the memref types for A, B, and C in the shared memory.
    # Note as of right now, the tuple format (strided, strided, ..., contiguous)
    self.a_tma_box_shape = (self.gemm_description.cta_shape.m, 64) # Row-major A
    self.b_tma_box_shape = (self.gemm_description.cta_shape.k, 64) # Row-major B
    
    self.a_tma_box_bytes = np.product(self.a_tma_box_shape) * self.gemm_description.a_bitwidth // 8
    self.b_tma_box_bytes = np.product(self.b_tma_box_shape) * self.gemm_description.b_bitwidth // 8

    self.a_tma_box_smemref_ty = ir.MemRefType.get(self.a_tma_box_shape,
                                                  self.gemm_description.a.datatype,
                                                  memory_space = b.smem_addr_space_attr)
    self.b_tma_box_smemref_ty = ir.MemRefType.get(self.b_tma_box_shape,
                                                  self.gemm_description.b.datatype,
                                                  memory_space = b.smem_addr_space_attr)
    # 
    # Hopper Tensor Cores. 
    #
    vector_frag_str = f"vector<{gemm_description.cta_shape.m}x{gemm_description.cta_shape.n}x"\
                             f"{gemm_description.accum_ty}>"

    self.accum_fragmented_ty = ir.Type.parse(f"!nvgpu.warpgroup.accumulator<fragmented = {vector_frag_str}>")
    self.a_wgmma_desc_ty = ir.Type.parse(f"!nvgpu.warpgroup.descriptor<tensor=memref<\
                                          {gemm_description.cta_shape.m}x{gemm_description.cta_shape.k}x{gemm_description.a.datatype}, 3>>")
    self.b_wgmma_desc_ty = ir.Type.parse(f"!nvgpu.warpgroup.descriptor<tensor=memref<\
                                          {gemm_description.cta_shape.k}x{gemm_description.cta_shape.n}x{gemm_description.b.datatype}, 3>>")
    
    #
    # Epilogue
    #
    self.accum_smem_ty = ir.MemRefType.get((self.gemm_description.cta_shape.m, self.gemm_description.cta_shape.n), 
                                           self.gemm_description.accum_ty, memory_space = b.smem_addr_space_attr)
    self.accum_linearized_smem_ty = ir.MemRefType.get([self.gemm_description.cta_shape.m * self.gemm_description.cta_shape.n],
                                                      self.gemm_description.accum_ty, memory_space = b.smem_addr_space_attr)
    self.output_gmem_ty = ir.MemRefType.get((self.gemm_description.cta_shape.m, self.gemm_description.cta_shape.n),
                                                  self.gemm_description.c.datatype,
                                                  ir.Attribute.parse(f"strided<[{self.gemm_description.problem_shape.n}, 1], offset: ?>"))
    
    print(f"gemm_description : {self.gemm_description.name()}")

  @property
  def gemm_iteration_k(self):
    """Returns the number of iterations in the k-dimension."""
    return self.gemm_description.problem_shape.k // self.gemm_description.cta_shape.k
  
  @property
  def dim3_grid(self):
    """Returns the grid shape as dim3 tuple in number of cta launched in ms-, n-, k-dim."""
    grid_m = ((self.gemm_description.problem_shape.m + self.gemm_description.cta_shape.m - 1) // self.gemm_description.cta_shape.m)
    grid_n = ((self.gemm_description.problem_shape.n + self.gemm_description.cta_shape.n - 1) // self.gemm_description.cta_shape.n)
    dim3_grid = (grid_m, grid_n, 1)
    return dim3_grid

  @property
  def dim3_block(self):
    """Returns the block shape as dim3 tuple in number of threads in m-, n-, k-dim."""
    warp_thread_count = 32
    return (self.gemm_description.warp_shape.m * warp_thread_count,
            self.gemm_description.warp_shape.n,
            self.gemm_description.warp_shape.k)
  
  @property
  def block_size(self):
    """Returns the block size in number of threads."""
    return np.product(self.dim3_block)
  
  def __call__(self):
    """Builds and returns a GEMM kernel module."""
    if self.gemm_description.mainloop_variant == MainloopVariant.Multistage:
      return self.multistage_gemm_kernel()
    elif self.gemm_description.mainloop_variant == MainloopVariant.WarpSpecializedCooperative:
      return self.warp_specialized_cooperative_gemm_kernel()
    else:
      raise NotImplementedError(self.gemm_description.mainloop_variant)
    
  def multistage_gemm_kernel(self):
    """Builds and returns a multistage GEMM kernel module."""
    b = Builder()
    # Create a module.
    module = ir.Module.create()

    with ir.InsertionPoint(module.body):
      func_op = func.FuncOp(self.gemm_description.name(),
                            ([self.a_memref_ty, self.b_memref_ty, self.c_memref_ty],
                             []))
      with ir.InsertionPoint(func_op.add_entry_block()):

        # create commonly used constants
        c0_op = b.const_index_op(0)
        c1_op = b.const_index_op(1)
        ticks_op = b.const_index_op(10000000)
        phase_bit_op = b.const_int_op(0, b.i1_ty)
        gemm_iteration_k_op = b.const_index_op(self.gemm_iteration_k)
        num_stages_op = b.const_index_op(self.gemm_description.num_stages)

        # create smem start constant indices for partitioning dynamic smem for a and b 
        a_smem_start_byte_op = c0_op
        b_smem_start_byte_op = b.const_index_op(self.gemm_description.a_smem_bytes)
        smem_txcount_bytes_op = b.const_index_op(self.gemm_description.smem_bytes_per_stage)
        a_tma_box_bytes_op = b.const_index_op(self.a_tma_box_bytes)
        b_tma_box_bytes_op = b.const_index_op(self.b_tma_box_bytes)
        a_smem_bytes_per_stage_op = b.const_index_op(self.gemm_description.a_smem_bytes_per_stage)
        b_smem_bytes_per_stage_op = b.const_index_op(self.gemm_description.b_smem_bytes_per_stage)

        a_host = func_op.arguments[0]
        b_host = func_op.arguments[1]
        c_host = func_op.arguments[2]

        # Step 1. Allocate device memory and memcpy
        t1 = gpu.wait(b.gpu_token_ty, [])
        a_device, t2 = gpu.alloc(self.a_memref_ty, b.gpu_token_ty, [t1], [], [])
        b_device, t3 = gpu.alloc(self.b_memref_ty, b.gpu_token_ty, [t2], [], [])
        c_device, t4 = gpu.alloc(self.c_memref_ty, b.gpu_token_ty, [t3], [], [])
        t5 = gpu.memcpy(b.gpu_token_ty, [t4], a_device, a_host)
        t6 = gpu.memcpy(b.gpu_token_ty, [t5], b_device, b_host)
        t7 = gpu.wait(b.gpu_token_ty, [t6])

        # Step 2. Create a TMA descriptor for A and B operands
        a_tma_desc = TmaDescriptorBuilder(nvgpu.TensorMapSwizzleKind.SWIZZLE_128B,
                                          nvgpu.TensorMapL2PromoKind.L2PROMO_NONE,
                                          nvgpu.TensorMapOOBKind.OOB_ZERO,
                                          nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
                                          self.a_tma_box_shape,
                                          self.a_memref_ty)

        b_tma_desc = TmaDescriptorBuilder(nvgpu.TensorMapSwizzleKind.SWIZZLE_128B,
                                          nvgpu.TensorMapL2PromoKind.L2PROMO_NONE,
                                          nvgpu.TensorMapOOBKind.OOB_ZERO,
                                          nvgpu.TensorMapInterleaveKind.INTERLEAVE_NONE,
                                          self.b_tma_box_shape,
                                          self.b_memref_ty)

        a_tma_desc_op = a_tma_desc.tma_descriptor_op(a_device)
        b_tma_desc_op = b_tma_desc.tma_descriptor_op(b_device)        

        # create smem size constant int32 op for dynamic shared memory 
        smem_size_const_i32_op = b.const_int_op(self.gemm_description.smem_bytes, b.i32_ty)
        
        # create kernel launch
        launch_op = gpu.LaunchOp(b.gpu_token_ty, [t7], 
                              *map(b.const_index_op, self.dim3_grid),
                              *map(b.const_index_op, self.dim3_block),
                              dynamicSharedMemorySize = smem_size_const_i32_op)
        
        # Kernel body
        launch_op.body.blocks.append(*([T.index()] * 12))
        with ir.InsertionPoint(launch_op.body.blocks[0]):
            
          # GPU Step 0. Bootstrapping
          memref.assume_alignment(c_device, 16)
          dynamic_smem_op = gpu.dynamic_shared_memory(self.dynamic_smemref_ty)
          tidx_op = gpu.ThreadIdOp(gpu.Dimension.x)

          bidx_op = gpu.block_id(gpu.Dimension.x)
          bidy_op = gpu.block_id(gpu.Dimension.y)
          # x-dim is mapped to problem m
          offset_m_op = arith.muli(bidx_op, b.const_index_op(self.gemm_description.cta_shape.m))
          # y-dim is mapped to problem n
          offset_n_op = arith.muli(bidy_op, b.const_index_op(self.gemm_description.cta_shape.n)) 
          
          # GPU Step 1. Initialize barriers
          is_leader_op = arith.CmpIOp(arith.CmpIPredicate.eq, tidx_op, c0_op)
          mbarrier_group_op = nvgpu.mbarrier_create(self.mbarrier_group_ty)

          for i in range(self.gemm_description.num_stages):
            nvgpu.mbarrier_init(mbarrier_group_op, c1_op, b.const_index_op(i), predicate=is_leader_op)
          gpu.barrier()

          # GPU Step 2. Prefetch TMA descriptors
          nvgpu.tma_prefetch_descriptor(a_tma_desc_op, predicate=is_leader_op)
          nvgpu.tma_prefetch_descriptor(b_tma_desc_op, predicate=is_leader_op)

          # GPU Step 3. Prologue (global memory -> shared memory)
          prologue_stages_op = b.const_index_op(self.gemm_description.num_stages - 1)

          a_smem_bytes_per_stage_op = b.const_index_op(self.gemm_description.a_smem_bytes_per_stage)
          b_smem_bytes_per_stage_op = b.const_index_op(self.gemm_description.b_smem_bytes_per_stage)

          for_op = scf.ForOp(c0_op, prologue_stages_op, c1_op)
          with ir.InsertionPoint(for_op.body):
            stage_op = for_op.induction_variable
            a_smem_byte_offset_op = arith.muli(stage_op, a_smem_bytes_per_stage_op)
            a_tma_box_view_op = memref.view(self.a_tma_box_smemref_ty, dynamic_smem_op, a_smem_byte_offset_op, [])
            offset_k_op = arith.muli(b.const_index_op(self.gemm_description.cta_shape.k), stage_op)

            nvgpu.TmaAsyncLoadOp(a_tma_box_view_op, mbarrier_group_op, a_tma_desc_op, 
                                 coordinates = [offset_k_op, offset_m_op], # contiguous, strided (Row-Major A)
                                 mbarId = stage_op,
                                 predicate = is_leader_op)

            debug_print(
                "[Prologue] stage={}, A TMA Load a_smem_byte_offset_op={} @ a=({},{})",
                stage_op,
                a_smem_byte_offset_op,
                offset_k_op,
                offset_m_op,
                predicate=is_leader_op,
            )

            b_tma_1_offset_n_op = offset_n_op
            b_smem_byte_offset_op = arith.addi(b_smem_start_byte_op, arith.muli(stage_op, b_smem_bytes_per_stage_op))
            b_tma_box_view_op = memref.view(self.b_tma_box_smemref_ty, dynamic_smem_op, b_smem_byte_offset_op, [])
            nvgpu.TmaAsyncLoadOp(b_tma_box_view_op, mbarrier_group_op, b_tma_desc_op,
                                 coordinates = [b_tma_1_offset_n_op, offset_k_op], # contiguous, strided (Row-Major B)
                                 mbarId = stage_op,
                                 predicate = is_leader_op)

            debug_print(
                "[Prologue] stage {}, B TMA Load b_smem_byte_offset_op={} @ b=({},{})",
                stage_op,
                b_smem_byte_offset_op,
                b_tma_1_offset_n_op,
                offset_k_op,
                predicate=is_leader_op,
            )

            b_tma_2_offset_n_op = arith.addi(b_tma_1_offset_n_op, b.const_index_op(64))
            b_smem_byte_offset_op = arith.addi(b_smem_byte_offset_op, b_tma_box_bytes_op)
            b_tma_box_view_op = memref.view(self.b_tma_box_smemref_ty, dynamic_smem_op, b_smem_byte_offset_op, [])
            nvgpu.TmaAsyncLoadOp(b_tma_box_view_op, mbarrier_group_op, b_tma_desc_op,
                                 coordinates = [b_tma_2_offset_n_op, offset_k_op], # contiguous, strided (Row-Major B)
                                 mbarId = stage_op,
                                 predicate = is_leader_op)

            debug_print(
                "[Prologue] stage {}, B TMA Load b_smem_byte_offset_op={} @ b=({},{})",
                stage_op,
                b_smem_byte_offset_op,
                b_tma_2_offset_n_op,
                offset_k_op,
                predicate=is_leader_op,
            )

            nvgpu.mbarrier_arrive_expect_tx(mbarrier_group_op, smem_txcount_bytes_op, stage_op, predicate = is_leader_op)
            scf.yield_([])

          # initialize accumulators
          accum_op = nvgpu.warpgroup_mma_init_accumulator(self.accum_fragmented_ty)

          # GPU Step 4. Mainloop
          for_op = scf.ForOp(c0_op, gemm_iteration_k_op, c1_op, [accum_op, arith.constant(T.bool(), 0)])
          with ir.InsertionPoint(for_op.body):
            # Step 4.1 Wait on the mbarrier
            phase_bit_op = for_op.inner_iter_args[1]
            iteration_k_op = for_op.induction_variable
            smem_read_stage_op = arith.remui(iteration_k_op, num_stages_op)

            debug_print("[Mainloop] iteration={}, smem_write_stage_op={}, phase_bit_op={}", 
                        iteration_k_op, smem_read_stage_op, phase_bit_op, predicate=is_leader_op)
            
            nvgpu.MBarrierTryWaitParityOp(mbarrier_group_op, phase_bit_op, ticks_op, mbarId=smem_read_stage_op)
            
            debug_print("[Mainloop] iteration={}, smem_write_stage_op={}, phase_bit_op={} [done]", 
                        iteration_k_op, smem_read_stage_op, phase_bit_op, predicate=is_leader_op)
            
            # 4.2 Create WGMMA Descriptors
            a_smem_byte_offset_op = arith.muli(smem_read_stage_op, a_smem_bytes_per_stage_op)
            a_tile_view_op = memref.view(self.a_smem_stage_smemref_ty, dynamic_smem_op, a_smem_byte_offset_op, [])

            b_smem_byte_offset_op = arith.addi(arith.muli(smem_read_stage_op, b_smem_bytes_per_stage_op), b_smem_start_byte_op)
            b_tile_view_op = memref.view(self.b_smem_stage_smemref_ty, dynamic_smem_op, b_smem_byte_offset_op, [])

            a_wgmma_desc_op = nvgpu.WarpgroupGenerateDescriptorOp(self.a_wgmma_desc_ty, a_tile_view_op, a_tma_desc_op)
            b_wgmma_desc_op = nvgpu.WarpgroupGenerateDescriptorOp(self.b_wgmma_desc_ty, b_tile_view_op, b_tma_desc_op)

            debug_print("[Mainloop] iteration={}, smem_read_stage_op={}, WGMMA (SS) @ a_smem_byte_offset_op={}, b_smem_byte_offset_op={}", 
                        iteration_k_op,
                        smem_read_stage_op, 
                        a_smem_byte_offset_op,
                        b_smem_byte_offset_op,
                        predicate=is_leader_op)

            # 4.3 Execute WGMMA
            carry_accum_op = for_op.inner_iter_args[0]
            new_accum_op = nvgpu.WarpgroupMmaOp(accum_op.type, a_wgmma_desc_op, b_wgmma_desc_op, carry_accum_op, transposeB=True)

            
            # 4.4 Issue TMA for the next stage
            write_iteration_op = arith.addi(iteration_k_op, prologue_stages_op)
            is_oob_op = arith.cmpi(arith.CmpIPredicate.ult, write_iteration_op, gemm_iteration_k_op)
            pred_op = arith.andi(is_oob_op, is_leader_op)
            smem_write_stage_op = arith.remui(write_iteration_op, num_stages_op)

            a_smem_byte_offset_op = arith.addi(a_smem_start_byte_op, arith.muli(smem_write_stage_op, a_smem_bytes_per_stage_op))
            a_tma_box_view_op = memref.view(self.a_tma_box_smemref_ty, dynamic_smem_op, a_smem_byte_offset_op, []) 
            offset_k_op = arith.muli(b.const_index_op(self.gemm_description.cta_shape.k), write_iteration_op)
            
            nvgpu.mbarrier_arrive_expect_tx(mbarrier_group_op, smem_txcount_bytes_op, smem_write_stage_op, predicate = pred_op)

            debug_print(
                "[Mainloop] smem_write_stage_op={}, A TMA Load a_smem_byte_offset_op={} @ a=({},{})",
                smem_write_stage_op,
                a_smem_byte_offset_op,
                offset_k_op,
                offset_m_op,
                predicate=is_leader_op,
            )
            
            nvgpu.TmaAsyncLoadOp(a_tma_box_view_op, mbarrier_group_op, a_tma_desc_op,
                                  coordinates = [offset_k_op, offset_m_op], # contiguous, strided (Row-Major A)
                                  mbarId = smem_write_stage_op,
                                  predicate = pred_op)
            
            b_smem_byte_offset_op = arith.addi(b_smem_start_byte_op, arith.muli(smem_write_stage_op, b_smem_bytes_per_stage_op))
            b_tma_box_view_op = memref.view(self.b_tma_box_smemref_ty, dynamic_smem_op, b_smem_byte_offset_op, [])
            b_tma_1_offset_n_op = offset_n_op

            debug_print(
                "[Mainloop] smem_write_stage_op={}, B TMA Load a_smem_byte_offset_op={} @ b=({},{})",
                smem_write_stage_op,
                b_smem_byte_offset_op,
                b_tma_1_offset_n_op,
                offset_k_op,
                predicate=is_leader_op,
            )

            nvgpu.TmaAsyncLoadOp(b_tma_box_view_op, mbarrier_group_op, b_tma_desc_op,
                                  coordinates = [b_tma_1_offset_n_op, offset_k_op], # contiguous, strided (Row-Major B)
                                  mbarId = smem_write_stage_op,
                                  predicate = pred_op)
            
            b_tma_2_offset_n_op = arith.addi(b_tma_1_offset_n_op, b.const_index_op(64))
            b_smem_byte_offset_op = arith.addi(b_smem_byte_offset_op, b_tma_box_bytes_op)
            b_tma_box_view_op = memref.view(self.b_tma_box_smemref_ty, dynamic_smem_op, b_smem_byte_offset_op, [])

            debug_print(
                "[Mainloop] smem_write_stage_op={}, B TMA Load a_smem_byte_offset_op={} @ b=({},{})",
                smem_write_stage_op,
                b_smem_byte_offset_op,
                b_tma_2_offset_n_op,
                offset_k_op,
                predicate=is_leader_op,
            )
            
            nvgpu.TmaAsyncLoadOp(b_tma_box_view_op, mbarrier_group_op, b_tma_desc_op,
                                  coordinates = [b_tma_2_offset_n_op, offset_k_op], # contiguous, strided (Row-Major B)
                                  mbarId = smem_write_stage_op,
                                  predicate = pred_op)
            
            # Step 4.5 Flip the phase bit
            pred_op = arith.cmpi(arith.CmpIPredicate.eq, smem_read_stage_op, b.const_index_op(self.gemm_description.num_stages - 1))
            phase_bit_op = arith.select(pred_op, arith.xori(phase_bit_op, arith.constant(T.bool(), 1)), phase_bit_op)


            # Step 4.6 Loop back. Yield accumulators and phase bit
            scf.yield_([new_accum_op, phase_bit_op])

          # Step 5. Wait for all the WGMMA to drain.
          nvvm.WgmmaWaitGroupSyncOp(0)

          # Print the accumulators for a few threads
          # print(for_op.results[0].fragmented)

          # Step 6. Epilogue (shared memory -> global memory)
          accum_smem_view_op = memref.view(self.accum_smem_ty, dynamic_smem_op, c0_op, [])
          nvgpu.WarpgroupMmaStoreOp(for_op.results[0], accum_smem_view_op)
          gpu.barrier()
          
          accum_smem_linearized_view_op = memref.view(self.accum_linearized_smem_ty, dynamic_smem_op, c0_op, [])

          output_gmem_view_op = memref.SubViewOp(
                                            self.output_gmem_ty, 
                                            c_device,
                                            [offset_m_op, offset_n_op],
                                            [],
                                            [],
                                            [MLIR_DYNAMIC, MLIR_DYNAMIC],
                                            [self.gemm_description.cta_shape.m, self.gemm_description.cta_shape.n],
                                            [1, 1],)
          
          vlen = 1
          for_op = scf.ForOp(
            tidx_op,
            b.const_index_op(self.gemm_description.cta_shape.m * self.gemm_description.cta_shape.n),
            b.const_index_op(vlen * self.block_size),
          )
          with ir.InsertionPoint(for_op.body):
            coord_m = arith.divui(for_op.induction_variable, b.const_index_op(self.gemm_description.cta_shape.m))
            coord_n = arith.remui(for_op.induction_variable, b.const_index_op(self.gemm_description.cta_shape.n))
            val = vector.load(
              ir.VectorType.get((vlen,), self.gemm_description.c.datatype),
              accum_smem_linearized_view_op,
              [for_op.induction_variable],
            )
            val_data = vector.extractelement(val, position=c0_op)
            debug_print("[Epilogue] coord_m={}, coord_n={}, val={}", coord_m, coord_n, val_data, predicate=is_leader_op)
            
            vector.store(val, output_gmem_view_op, [coord_m, coord_n])
            scf.yield_([])
          gpu.terminator()

        # Step 7. Copy output back to host and free device memory
        t8 = gpu.wait(b.gpu_token_ty, [launch_op])
        t9 = gpu.memcpy(b.gpu_token_ty, [t8], c_host, c_device)
        gpu.dealloc(b.gpu_token_ty, [t8], a_device)
        gpu.dealloc(b.gpu_token_ty, [t8], b_device)
        gpu.wait(b.gpu_token_ty, [t9])
        gpu.dealloc(b.gpu_token_ty, [t9], c_device)
        func.ReturnOp([])

    func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    module.operation.verify()
    print(module)
    return module
//===- OptimizeSharedMemory.cpp - MLIR NVGPU pass implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to enable 1xtf32 and 3xtf32 nvgpu.mma sync 
// operations on f32 input datatype
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Passes.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <iostream>
using namespace mlir;
using namespace mlir::nvgpu;


namespace {

struct MmaSyncF32ToT32
    : public OpRewritePattern<nvgpu::MmaSyncOp> {

  using OpRewritePattern<nvgpu::MmaSyncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(nvgpu::MmaSyncOp mmasyncOp,
                                PatternRewriter &rewrite) const override {
    
    std::cout << "Entering MmaSyncF32ToT32 " << std::endl;

    return success();
  }
};

} // namespace

void mlir::nvgpu::populateMmaSyncF32ToT32Patterns(
  RewritePatternSet &patterns) {

  patterns.add<MmaSyncF32ToT32>(patterns.getContext());
} 
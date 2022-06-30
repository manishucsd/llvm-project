//===- TestVectorTransforms.cpp - Test Vector transforms and lowerings ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::nvgpu;

namespace {

struct TestMmaSyncF32ToT32Patterns
    : public PassWrapper<TestMmaSyncF32ToT32Patterns,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestMmaSyncF32ToT32Patterns)

  StringRef getArgument() const final {
    return "test-nvgpu-mmasync-f32-to-tf32-patterns";
  }
  StringRef getDescription() const final {
    return "Test patterns to convert mma.sync on f32 with tf32 precision";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMmaSyncF32ToT32Patterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestNvgpuLowerings() {
  PassRegistration<TestMmaSyncF32ToT32Patterns>();    
}

} // namespace test
} // namespace mlir
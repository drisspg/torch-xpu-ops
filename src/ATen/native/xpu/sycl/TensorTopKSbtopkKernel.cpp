/*
 * Subgroup top-k dispatch logic.
 *
 * Routes to per-K compilation units (TensorTopKSbtopkKernel_k*.cpp)
 * for the actual kernel instantiations.
 *
 * Splitting per-K into separate files parallelizes AOT compilation
 * across build targets.
 */

#include <ATen/native/xpu/sycl/TensorTopKSbtopkKernel.h>
#include <ATen/native/xpu/sycl/TensorTopKSbtopkKernelImpl.h>
#include <c10/util/llvmMathExtras.h>
#include <comm/DeviceProperties.h>

namespace at::native::xpu {

// Subgroup top-k: dispatch on K (rounded up to power of two).
static bool subgroup_topk_try_launch(
    const at::Tensor& self,
    int64_t nsegments,
    int64_t nelements,
    int64_t k,
    bool largest,
    const at::Tensor& values,
    const at::Tensor& indices) {
  if (k > 16) {
    return false;
  }

  int K_sel = std::min<int>(
      static_cast<int>(c10::llvm::PowerOf2Ceil(static_cast<uint64_t>(k))), 16);

#define SBTOPK_LAUNCH(KVAL) \
  sbtopk_k##KVAL##_launch(  \
      self,                 \
      nsegments,            \
      nelements,            \
      static_cast<int>(k),  \
      largest,              \
      values,               \
      indices)

  switch (K_sel) {
    case 1:
      SBTOPK_LAUNCH(1);
      break;
    case 2:
      SBTOPK_LAUNCH(2);
      break;
    case 4:
      SBTOPK_LAUNCH(4);
      break;
    case 8:
      SBTOPK_LAUNCH(8);
      break;
    default:
      SBTOPK_LAUNCH(16);
      break;
  }

#undef SBTOPK_LAUNCH

  return true;
}

// ================================================================
// Dispatch: subgroup top-k vs original
//
//   - dim < 32: original (need at least SG_SIZE elements)
//   - dim >= 32, large batch, k <= 16: subgroup top-k
// ================================================================
SbtopkResult sbtopk_try_launch(
    const at::Tensor& self,
    int64_t nsegments,
    int64_t nelements,
    int64_t k,
    bool largest,
    const at::Tensor& values,
    const at::Tensor& indices) {
  // Subgroup kernel needs at least SG_SIZE (32) elements per slice
  if (nelements < 32) {
    return SbtopkResult::FAILED;
  }

  // Subgroup top-k: best for large batch, k<=16.
  // Output is ALREADY SORTED (descending for largest, ascending for smallest).
  //
  // Threshold: nsegments >= thread_slots / 4.
  //   Subgroup top-k uses 1 sub-group per slice (reading data once), while
  //   the original kernel reads data multiple times (~3 radix passes). So
  //   subgroup top-k reaches memory-BW saturation at much lower occupancy.
  //   thread_slots/4 is the conservative cutoff.
  //
  // On B580: thread_slots = 160 EU * 8 HW threads = 1280, threshold = 320.
  int64_t thread_slots =
      ::xpu::sycl::syclGpuEuCount() * ::xpu::sycl::syclGpuHWThreadsPerEU();
  int64_t sg_threshold = thread_slots / 4;
  if (k <= 16 && nsegments >= sg_threshold) {
    if (subgroup_topk_try_launch(
            self, nsegments, nelements, k, largest, values, indices)) {
      return SbtopkResult::SORTED;
    }
    return SbtopkResult::FAILED;
  }

  return SbtopkResult::FAILED;
}

} // namespace at::native::xpu

/*
 * Subgroup top-k kernel -- K=16 instantiations.
 *
 * Splitting per-K into separate files to parallelize AOT compilation.
 * See TensorTopKSbtopkKernelImpl.h for the shared template code.
 */

#include <ATen/native/xpu/sycl/TensorTopKSbtopkKernelImpl.h>

namespace at::native::xpu {

template <typename scalar_t, typename IndexT>
static void sbtopk_k16_typed(
    const scalar_t* input,
    scalar_t* topK,
    int64_t* indices,
    IndexT numSlices,
    int64_t sliceSize,
    int k,
    bool largest) {
  if (largest) {
    sbtopk_launch_vec_dispatch<scalar_t, 16, true, IndexT>(
        input, topK, indices, numSlices, sliceSize, k);
  } else {
    sbtopk_launch_vec_dispatch<scalar_t, 16, false, IndexT>(
        input, topK, indices, numSlices, sliceSize, k);
  }
}

void sbtopk_k16_launch(
    const at::Tensor& self,
    int64_t nsegments,
    int64_t nelements,
    int k,
    bool largest,
    const at::Tensor& values,
    const at::Tensor& indices) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "sbtopk_k16",
      [&]() {
        const auto* input = static_cast<const scalar_t*>(self.const_data_ptr());
        auto* topK = static_cast<scalar_t*>(values.data_ptr());
        auto* idx = static_cast<int64_t*>(indices.data_ptr());

        if (nsegments <=
            static_cast<int64_t>(std::numeric_limits<int>::max())) {
          sbtopk_k16_typed<scalar_t, int>(
              input,
              topK,
              idx,
              static_cast<int>(nsegments),
              nelements,
              k,
              largest);
        } else {
          sbtopk_k16_typed<scalar_t, int64_t>(
              input, topK, idx, nsegments, nelements, k, largest);
        }
      });
}

} // namespace at::native::xpu

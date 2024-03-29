/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

#include <complex>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/common_runtime/ve/ve_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#endif

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace tensorflow {
namespace {

template <typename T, bool conjugate>
void TransposeSimple(const CPUDevice& device, const Tensor& in,
                     const gtl::ArraySlice<int32> perm, Tensor* out) {
  const int ndims = in.dims();
  gtl::InlinedVector<int64, 8> in_strides = ComputeStride<int64>(in.shape());
  gtl::InlinedVector<int64, 8> out_strides = ComputeStride<int64>(out->shape());
  const T* p = reinterpret_cast<const T*>(in.tensor_data().data());
  T* q = reinterpret_cast<T*>(const_cast<char*>((out->tensor_data().data())));
  auto transpose_fn = [=, &in_strides, &out_strides, &perm](int64 begin,
                                                            int64 end) {
    for (int64 o_idx = begin; o_idx < end; ++o_idx) {
      int64 i_idx = 0;
      int64 t = o_idx;
      for (int i = 0; i < ndims; ++i) {
        const int64 ratio = t / out_strides[i];
        t -= ratio * out_strides[i];
        i_idx += ratio * in_strides[perm[i]];
      }
      if (conjugate) {
        q[o_idx] = Eigen::numext::conj(p[i_idx]);
      } else {
        q[o_idx] = p[i_idx];
      }
    }
  };
  double cycles_per_element =
      (conjugate ? 1 : 0) + ndims * (Eigen::TensorOpCost::DivCost<int64>() +
                                     2 * Eigen::TensorOpCost::MulCost<int64>() +
                                     2 * Eigen::TensorOpCost::AddCost<int64>());
  Eigen::TensorOpCost cost(/*bytes_loaded=*/sizeof(T),
                           /*bytes_stored=*/sizeof(T), cycles_per_element);
  device.parallelFor(in.NumElements(), cost, std::move(transpose_fn));
}

}  // namespace

template <typename T, bool conjugate>
struct Transpose<CPUDevice, T, conjugate> {
  static void run(const CPUDevice& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out) {
    switch (in.dims()) {
      case 2:
        internal::TransposeUsingEigen<CPUDevice, T, 2>(d, in, perm, conjugate,
                                                       out);
        break;
      case 3:
        internal::TransposeUsingEigen<CPUDevice, T, 3>(d, in, perm, conjugate,
                                                       out);
        break;
      case 4:
        internal::TransposeUsingEigen<CPUDevice, T, 4>(d, in, perm, conjugate,
                                                       out);
        break;
      case 5:
        internal::TransposeUsingEigen<CPUDevice, T, 5>(d, in, perm, conjugate,
                                                       out);
        break;
      case 6:
        internal::TransposeUsingEigen<CPUDevice, T, 6>(d, in, perm, conjugate,
                                                       out);
        break;
      case 7:
        internal::TransposeUsingEigen<CPUDevice, T, 7>(d, in, perm, conjugate,
                                                       out);
        break;
      case 8:
        internal::TransposeUsingEigen<CPUDevice, T, 8>(d, in, perm, conjugate,
                                                       out);
        break;
      default:
        TransposeSimple<T, conjugate>(d, in, perm, out);
        break;
    }
  }
};

#define INSTANTIATE(DEVICE)                                                 \
  template <>                                                               \
  Status DoTranspose(const DEVICE& device, const Tensor& in,                \
                     const gtl::ArraySlice<int32> perm, Tensor* out) {      \
    return internal::DoTransposeImpl(device, in, perm, /*conjugate=*/false, \
                                     out);                                  \
  }                                                                         \
  template <>                                                               \
  Status DoConjugateTranspose(const DEVICE& device, const Tensor& in,       \
                              const gtl::ArraySlice<int32> perm,            \
                              Tensor* out) {                                \
    return internal::DoTransposeImpl(device, in, perm, /*conjugate=*/true,  \
                                     out);                                  \
  }                                                                         \
  template <>                                                               \
  Status DoMatrixTranspose(const DEVICE& device, const Tensor& in,          \
                           Tensor* out) {                                   \
    return internal::DoMatrixTransposeImpl(device, in, /*conjugate=*/false, \
                                           out);                            \
  }                                                                         \
  template <>                                                               \
  Status DoConjugateMatrixTranspose(const DEVICE& device, const Tensor& in, \
                                    Tensor* out) {                          \
    return internal::DoMatrixTransposeImpl(device, in, /*conjugate=*/true,  \
                                           out);                            \
  }

INSTANTIATE(CPUDevice)


#ifdef TENSORFLOW_USE_VE
struct VETransposeArgs {
  int dtype;
  uint64_t in;
  uint64_t out;
  int size;
  int conjugate ;	// boolean
  int32_t dim_size[8];
  int32_t perm[8];
} ;

Status VEDoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) {
  if (perm.size() > 8) {
    return errors::InvalidArgument("transpose with perm.size > 8 "
                                   "is not supported on VE. perm.size is ",
                                   perm.size());
  }

  struct VETransposeArgs args ;

  args.dtype = in.dtype();
  args.in = (uint64_t)DMAHelper::base(&in);
  args.out = (uint64_t)DMAHelper::base(out);
  args.size = perm.size();
  args.conjugate = 0 ;	// false
  for (int i = 0; i < perm.size(); ++i) {
    args.dim_size[i] = in.dim_size(i);
    args.perm[i] = perm[i];
  }

  VEDeviceContext* vectx = ctx->op_device_context<VEDeviceContext>();
  Status s = vectx->Compute("Transpose", (void*)&args, sizeof(args));
  return s;
}

Status VEDoConjugateTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) {
  if (perm.size() > 8) {
    return errors::InvalidArgument("transpose with perm.size > 8 "
                                   "is not supported on VE. perm.size is ",
                                   perm.size());
  }

  struct VETransposeArgs args ;

  args.dtype = in.dtype();
  args.in = (uint64_t)DMAHelper::base(&in);
  args.out = (uint64_t)DMAHelper::base(out);
  args.size = perm.size();
  args.conjugate = 1 ;	// false
  for (int i = 0; i < perm.size(); ++i) {
    args.dim_size[i] = in.dim_size(i);
    args.perm[i] = perm[i];
  }

  VEDeviceContext* vectx = ctx->op_device_context<VEDeviceContext>();
  Status s = vectx->Compute("Transpose", (void*)&args, sizeof(args));
  return s;
}

#endif

}  // namespace tensorflow

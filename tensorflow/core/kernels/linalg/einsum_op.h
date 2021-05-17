/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/framework/ve_ops_common.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {
namespace functor {

template <typename Device, typename T, int N>
struct StrideFunctor {
  void operator()(const Device& d, typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    output.device(d) = input.stride(strides);
  }
};

template <typename Device, typename T, int N>
struct InflateFunctor {
  void operator()(const Device& d, typename TTypes<T, N>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, N>& strides,
                  typename TTypes<T, N>::Tensor output) {
    output.device(d) = input.inflate(strides);
  }
};

#ifdef TENSORFLOW_USE_VE
struct VEStrideFunctor {
  void operator()(OpKernelContext* ctx, Tensor &input,
                  const gtl::InlinedVector<int64, 8>& strides,
                  Tensor &output, const int N) {

    VEOpKernelHelper::ArgsImpl<> args;
    args.addArg<int64>(N);
    args.addArg<Tensor>(input);
    args.addArg<Tensor>(output);
    for(int i=0; i<N; i++) args.addArg<int64>(strides[i]) ;

    VEOpKernelHelper::Call(ctx, "EinsumStride", args);
  }
};

struct VEInflateFunctor {
  void operator()(OpKernelContext* ctx, Tensor &input,
                  const gtl::InlinedVector<int64, 8>& strides,
                  Tensor &output, const int N) {
    VEOpKernelHelper::ArgsImpl<> args;
    args.addArg<int64>(N);
    args.addArg<Tensor>(input);
    args.addArg<Tensor>(output);
    for(int i=0; i<N; i++) args.addArg<int64>(strides[i]) ;

    VEOpKernelHelper::Call(ctx, "EinsumInflate", args);
  }
};
#endif	// TENSORFLOW_USE_VE

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_EINSUM_OP_H_

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

#if !defined(MLIR_GENERATED_EXPERIMENTAL_KERNELS_ENABLED)
REGISTER8(UnaryOp, CPU, "Square", functor::square, float, Eigen::half, double,
          int32, int64, complex64, complex128, bfloat16);
#else
REGISTER(UnaryOp, CPU, "Square", functor::square, bfloat16);
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER4(UnaryOp, GPU, "Square", functor::square, float, Eigen::half, double,
          int64);
#endif

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Square")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        UnaryOp<CPUDevice, functor::square<int32>>);
#endif

#ifdef TENSORFLOW_USE_VE
REGISTER_VE_UNARY_OP(Square, float);
REGISTER_KERNEL_BUILDER(Name("Square")
                            .Device(DEVICE_VE)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        UnaryOp<CPUDevice, functor::square<int32>>);
#endif  // TENSORFLOW_USE_SYC
}  // namespace tensorflow

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

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {
REGISTER6(UnaryOp, CPU, "Log1p", functor::log1p, float, Eigen::half, bfloat16,
          double, complex64, complex128);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if !defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
REGISTER3(UnaryOp, GPU, "Log1p", functor::log1p, float, Eigen::half, double);
#endif
#endif

#ifdef TENSORFLOW_USE_VE
REGISTER_VE_UNARY_OP(Log1p, float);
#endif  // TENSORFLOW_USE_VE
}  // namespace tensorflow

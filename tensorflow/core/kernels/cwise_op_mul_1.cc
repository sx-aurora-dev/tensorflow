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

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/common_runtime/ve/ve_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#endif

namespace tensorflow {

REGISTER6(BinaryOp, CPU, "Mul", functor::mul, float, Eigen::half, double, uint8,
          int32, bfloat16);
#if defined(__ANDROID_TYPES_SLIM__)
// We only register the first type when we have multi-argument calls in the
// case where we're trying to reduce executable size, but it turns out that the
// int32 version of this op is needed, so explicitly include it.
REGISTER(BinaryOp, CPU, "Mul", functor::mul, int32);
#endif  // __ANDROID_TYPES_SLIM__

#if GOOGLE_CUDA
REGISTER4(BinaryOp, GPU, "Mul", functor::mul, float, Eigen::half, double,
          uint8);
// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Mul")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::mul<int32>>);
#endif

#ifdef TENSORFLOW_USE_SYCL
REGISTER3(BinaryOp, SYCL, "Mul", functor::mul, float, double, uint8);
REGISTER_KERNEL_BUILDER(Name("Mul")
                            .Device(DEVICE_SYCL)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::mul<int32>>);
#endif  // TENSORFLOW_USE_SYCL

#ifdef TENSORFLOW_USE_VE
template <typename T>
class VEMulOp : public BinaryOpShared {
  public:
    explicit VEMulOp(OpKernelConstruction* context) 
      : BinaryOpShared(context, DataTypeToEnum<T>::v(), DataTypeToEnum<T>::v()) {}
    void Compute(OpKernelContext* context) override {
      BinaryOpState state(context);
      if (!context->status().ok()) return;

      VLOG(2) << "VEMulOp:"
        << " in0.shape=" << state.in0.shape().DebugString()
        << " in1.shape=" << state.in1.shape().DebugString()
        << " out.shape=" << state.out->shape().DebugString();

      struct {
        int dtype;
        uint64_t in0;
        uint64_t in1;
        uint64_t out;
        int32_t dims_in0;
        int32_t dims_in1;
        int32_t dims_out;
        int64_t nelems_in0;
        int64_t nelems_in1;
        int64_t nelems_out;
        int64_t dim_size_in0[8];
        int64_t dim_size_in1[8];
        int64_t dim_size_out[8];
      } args;

      args.dtype = state.in0.dtype();
      args.in0 = (uint64_t)DMAHelper::base(&state.in0);
      args.in1 = (uint64_t)DMAHelper::base(&state.in1);
      args.out = (uint64_t)DMAHelper::base(state.out);
      args.dims_in0 = state.in0.dims();
      args.dims_in1 = state.in1.dims();
      args.dims_out = state.out->dims();
      args.nelems_in0 = state.in0.NumElements();
      args.nelems_in1 = state.in1.NumElements();
      args.nelems_out = state.out->NumElements();

      if (args.dims_in0 > 8 || args.dims_in1 > 8 || args.dims_out > 8) {
        context->SetStatus(errors::Unimplemented(
                "dims_in0 > 8 || dims_in1 > 8 || dims_out > 8"
                " is not supported by VEMulOp"));
        return;
      }

      for (int i = 0; i < args.dims_in0; ++i)
        args.dim_size_in0[i] = state.in0.dim_size(i);
      for (int i = 0; i < args.dims_in1; ++i)
        args.dim_size_in1[i] = state.in1.dim_size(i);
      for (int i = 0; i < args.dims_out; ++i)
        args.dim_size_out[i] = state.out->dim_size(i);

      VEDeviceContext* vectx = context->op_device_context<VEDeviceContext>();
      Status s = vectx->Compute("Mul", (void*)&args, sizeof(args));
      if (!s.ok())
        context->SetStatus(s);
    }
};

REGISTER_KERNEL_BUILDER(Name("Mul")
                        .Device(DEVICE_VE)
                        .TypeConstraint<float>("T"),
                        VEMulOp<float>);

REGISTER_KERNEL_BUILDER(Name("Mul")
                            .Device(DEVICE_VE)
                            .HostMemory("x")
                            .HostMemory("y")
                            .HostMemory("z")
                            .TypeConstraint<int32>("T"),
                        BinaryOp<CPUDevice, functor::mul<int32>>);
#endif
}  // namespace tensorflow

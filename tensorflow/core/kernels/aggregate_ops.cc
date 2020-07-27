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

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/aggregate_ops.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/aggregate_ops_cpu.h"

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/common_runtime/ve/ve_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL
#ifdef TENSORFLOW_USE_VE
typedef Eigen::VeDevice VEDevice;
#endif  // TENSORFLOW_USE_VE

#define REGISTER_ADDN(type, dev)                                   \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AddN").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      AddNOp<dev##Device, type, OpKernel, OpKernelConstruction,    \
             OpKernelContext>)

#define REGISTER_ADDN_CPU(type) REGISTER_ADDN(type, CPU)

TF_CALL_NUMBER_TYPES(REGISTER_ADDN_CPU);
REGISTER_ADDN_CPU(Variant);

#undef REGISTER_ADDN_CPU

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)
#define REGISTER_ADDN_GPU(type) REGISTER_ADDN(type, GPU)
TF_CALL_int64(REGISTER_ADDN_GPU);
TF_CALL_variant(REGISTER_ADDN_GPU);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_ADDN_GPU);
TF_CALL_COMPLEX_TYPES(REGISTER_ADDN_GPU);
#undef REGISTER_ADDN_GPU

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(
    Name("AddN")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("T")
        .HostMemory("inputs")
        .HostMemory("sum"),
    AddNOp<CPUDevice, int32, OpKernel, OpKernelConstruction, OpKernelContext>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
REGISTER_ADDN(float, SYCL);
REGISTER_ADDN(double, SYCL);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(
    Name("AddN")
        .Device(DEVICE_SYCL)
        .TypeConstraint<int32>("T")
        .HostMemory("inputs")
        .HostMemory("sum"),
    AddNOp<CPUDevice, int32, OpKernel, OpKernelConstruction, OpKernelContext>);
#endif  // TENSORFLOW_USE_SYCL

#ifdef TENSORFLOW_USE_VE

template <typename T, class OpKernelT,
          class OpKernelConstructionT, class OpKernelContextT>
class AddNOp<VEDevice, T, OpKernelT, OpKernelConstructionT, OpKernelContextT> : public OpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    const Tensor& input0 = ctx->input(0);
    const int num = ctx->num_inputs();

    if (num == 1) {
      ctx->set_output(0, input0);
      return;
    } else {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input0.shape(), &output));

      struct Args {
        int output_type;
        uint64_t out;
        size_t num_elems;
        size_t num_inputs;
        uint64_t in[1];	// variable length
      } ;

      size_t argSize = sizeof(struct Args) + sizeof(uint64_t)*(num-1) ;
      struct Args *args = (struct Args *) malloc(argSize) ;

      args->output_type = DataTypeToEnum<T>::v();
      args->out = (uint64_t)DMAHelper::base(output);
      args->num_elems = input0.NumElements();
      args->num_inputs = num;
      for (int i = 0; i < num; ++i) {
        OP_REQUIRES(ctx, ctx->input(i).dtype() == args->output_type,
                    errors::InvalidArgument("type mismatch"));
        args->in[i] = (uint64_t)DMAHelper::base(&ctx->input(i));
      }

      VEDeviceContext* vectx = ctx->op_device_context<VEDeviceContext>();
      Status s = vectx->Compute("AddN", (void*)args, argSize);
      if (!s.ok())
        ctx->SetStatus(s);

      free(args) ;
    }
  }
};

REGISTER_ADDN(float, VE);
// REGISTER_ADDN(double, VE);
#endif // TENSORFLOW_USE_VE

#undef REGISTER_ADDN

}  // namespace tensorflow

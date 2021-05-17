/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/array_ops.cc.
#include "tensorflow/core/kernels/snapshot_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/common_runtime/ve/ve_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#endif

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Scalar>
class SnapshotOp : public OpKernel {
 public:
  explicit SnapshotOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    // Try to use buffer forwarding to avoid an explicit copy.
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (!output->SharesBufferWith(input)) {
      functor::Snapshot<Device, Scalar> functor;
      functor(context->eigen_device<Device>(), input.flat<Scalar>(),
              output->flat<Scalar>());
    }
  }
};

#define REGISTER_KERNEL(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Snapshot").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      SnapshotOp<CPUDevice, TYPE>);

TF_CALL_POD_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_KERNEL(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Snapshot").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      SnapshotOp<GPUDevice, TYPE>);

TF_CALL_POD_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif


#if TENSORFLOW_USE_VE
typedef Eigen::VeDevice VeDevice;

template <typename Scalar>
class SnapshotOp<VeDevice, Scalar> : public OpKernel {
 public:
  explicit SnapshotOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    Tensor* output = nullptr;
    // Try to use buffer forwarding to avoid an explicit copy.
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));
    if (!output->SharesBufferWith(input)) {
      struct {
        uint64_t dst;
        uint64_t src;
        size_t size;
      } args;

      args.src = (uint64_t)DMAHelper::base(&input);
      args.dst = (uint64_t)DMAHelper::base(output);
      args.size = input.NumElements() * sizeof(Scalar);

      VEDeviceContext* vectx = context->op_device_context<VEDeviceContext>();
      Status s = vectx->Compute("Snapshot", (void*)&args, sizeof(args));
      if (!s.ok())
        context->SetStatus(s);
    }
  }
};

#define REGISTER_VE_KERNEL(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("Snapshot").Device(DEVICE_VE).TypeConstraint<TYPE>("T"),   \
      SnapshotOp<VeDevice, TYPE>);

TF_CALL_POD_TYPES(REGISTER_VE_KERNEL);

#undef REGISTER_VE_KERNEL
#endif  // TENSORFLOW_USE_VE

}  // namespace tensorflow

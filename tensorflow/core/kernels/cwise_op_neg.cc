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
REGISTER7(UnaryOp, CPU, "Neg", functor::neg, float, Eigen::half, double, int32,
          complex64, int64, complex128);

#ifdef TENSORFLOW_USE_SYCL
REGISTER3(UnaryOp, SYCL, "Neg", functor::neg, float, double, int64);
REGISTER_KERNEL_BUILDER(Name("Neg")
                            .Device(DEVICE_SYCL)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        UnaryOp<CPUDevice, functor::neg<int32>>);
#endif  // TENSORFLOW_USE_SYCL

#ifdef TENSORFLOW_USE_VE
template <typename T>
class VEUnaryOp : public OpKernel {
  public:
    explicit VEUnaryOp(OpKernelConstruction* ctx, std::string name) 
      : OpKernel(ctx), name_(name) {}

    void Compute(OpKernelContext* ctx) override {
      const Tensor& inp = ctx->input(0);
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
              {0}, 0, inp.shape(), &out));

      struct _Tensor {
        int dtype;
        int data_format;
        uint64_t addr;
        int32_t dims;
        int64_t nelems;
        int64_t dim_size[8];
      };

      struct {
        _Tensor in;
        _Tensor out;
      } args;

      args.in.dtype = DataTypeToEnum<T>::v();
      args.in.addr = (uint64_t)DMAHelper::base(&inp);
      args.in.nelems = inp.NumElements();
      args.out.addr = (uint64_t)DMAHelper::base(out);

      VEDeviceContext* vectx = ctx->op_device_context<VEDeviceContext>();
      Status s = vectx->Compute(name_.c_str(), (void*)&args, sizeof(args));
      if (!s.ok())
        ctx->SetStatus(s);
    }

  private:
    std::string name_;
};

template<typename T>
class VENegOp : public VEUnaryOp<T> {
  public:
    explicit VENegOp(OpKernelConstruction* ctx) 
      : VEUnaryOp<T>(ctx, "Neg") {}
};

REGISTER_KERNEL_BUILDER(Name("Neg")
                        .Device(DEVICE_VE)
                        .TypeConstraint<float>("T"),
                        VENegOp<float>);

//REGISTER3(VEUnaryOp, VE, "Neg", functor::neg, float, double, int64);
REGISTER_KERNEL_BUILDER(Name("Neg")
                            .Device(DEVICE_VE)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        UnaryOp<CPUDevice, functor::neg<int32>>);
#endif  // TENSORFLOW_USE_VE

#if GOOGLE_CUDA
REGISTER6(UnaryOp, GPU, "Neg", functor::neg, float, Eigen::half, double, int64,
          complex64, complex128);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Neg")
                            .Device(DEVICE_GPU)
                            .HostMemory("x")
                            .HostMemory("y")
                            .TypeConstraint<int32>("T"),
                        UnaryOp<CPUDevice, functor::neg<int32>>);
#endif
}  // namespace tensorflow

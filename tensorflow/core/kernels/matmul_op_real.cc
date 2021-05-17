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

#include "tensorflow/core/kernels/matmul_op_impl.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/common_runtime/ve/ve_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#endif

namespace tensorflow {

TF_CALL_FLOAT_TYPES(REGISTER_BATCH_MATMUL_CPU);
TF_CALL_int16(REGISTER_BATCH_MATMUL_CPU);
TF_CALL_int32(REGISTER_BATCH_MATMUL_CPU);
TF_CALL_int64(REGISTER_BATCH_MATMUL_CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_GPU_NUMBER_TYPES(REGISTER_BATCH_MATMUL_GPU);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_VE
TF_CALL_float(REGISTER_BATCH_MATMUL_VE);
//TF_CALL_double(REGISTER_BATCH_MATMUL_VE);
//TF_CALL_half(REGISTER_BATCH_MATMUL_VE);

template <typename T>
class VEMatMulOp : public OpKernel {
  public:
    explicit VEMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor& a = ctx->input(0);
      const Tensor& b = ctx->input(1);

      // Check that the dimensions of the two matrices are valid.
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsMatrix(a.shape()),
          errors::InvalidArgument("In[0] is not a matrix. Instead it has shape ",
                                  a.shape().DebugString()));
      OP_REQUIRES(
          ctx, TensorShapeUtils::IsMatrix(b.shape()),
          errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                  b.shape().DebugString()));

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0].first = transpose_a_ ? 0 : 1;
      dim_pair[0].second = transpose_b_ ? 1 : 0;

      OP_REQUIRES(
          ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
          errors::InvalidArgument(
              "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
              ", In[1]: ", b.shape().DebugString()));
      int a_dim_remaining = 1 - dim_pair[0].first;
      int b_dim_remaining = 1 - dim_pair[0].second;
      TensorShape out_shape(
          {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

      if (out->NumElements() == 0) {
        // If a has shape [0, x] or b has shape [x, 0], the output shape
        // is a 0-element matrix, so there is nothing to do.
        return;
      }

      if (a.NumElements() == 0 || b.NumElements() == 0) {
        // If a has shape [x, 0] and b has shape [0, y], the
        // output shape is [x, y] where x and y are non-zero, so we fill
        // the output with zeros.
#if 0
        functor::SetZeroFunctor<Device, T> f;
        f(ctx->eigen_device<Device>(), out->flat<T>());
#endif
        ctx->SetStatus(errors::Unimplemented(
                "MatMul to fill zero is not supported on VE"));
        return;
      }

      VLOG(2) << "VEMatMulOp::Compute:"
        << " a=" << a.shape().DebugString()
        << " b=" << b.shape().DebugString()
        << " out=" << out->shape().DebugString()
        << " a.dtype=" << a.dtype()
        << " b.dtype=" << a.dtype();

      struct {
        int dtype;
        uint64_t a;
        uint64_t b;
        uint64_t out;
        int64_t dim_size_a[2];
        int64_t dim_size_b[2];
        int32_t transpose_a;
        int32_t transpose_b;
      } args;

      args.dtype = a.dtype();
      args.a = (uint64_t)DMAHelper::base(&a);
      args.b = (uint64_t)DMAHelper::base(&b);
      args.out = (uint64_t)DMAHelper::base(out);
      args.dim_size_a[0] = a.dim_size(0);
      args.dim_size_a[1] = a.dim_size(1);
      args.dim_size_b[0] = b.dim_size(0);
      args.dim_size_b[1] = b.dim_size(1);
      args.transpose_a = transpose_a_;
      args.transpose_b = transpose_b_;

      VEDeviceContext* vectx = ctx->op_device_context<VEDeviceContext>();
      Status s = vectx->Compute("MatMul", (void*)&args, sizeof(args));
      if (!s.ok())
        ctx->SetStatus(s);
    }

  private:
    bool transpose_a_;
    bool transpose_b_;
};

#define REGISTER_VE(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("MatMul")                         \
                              .Device(DEVICE_VE)               \
                              .TypeConstraint<T>("T"),            \
                          VEMatMulOp<T>)

TF_CALL_float(REGISTER_VE);
// TF_CALL_double(REGISTER_VE);
#endif // TENSORFLOW_USE_VE

}  // namespace tensorflow

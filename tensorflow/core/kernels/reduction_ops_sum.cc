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

#include "tensorflow/core/kernels/reduction_ops_common.h"

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/common_runtime/ve/ve_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#endif

namespace tensorflow {

#define REGISTER_CPU_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tidx"),                                      \
      ReductionOp<CPUDevice, type, int32, Eigen::internal::SumReducer<type>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_CPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tidx"),                                      \
      ReductionOp<CPUDevice, type, int64, Eigen::internal::SumReducer<type>>);
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA

#define REGISTER_GPU_KERNELS(type)                                             \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int32>("Tidx")                                       \
          .HostMemory("reduction_indices"),                                    \
      ReductionOp<GPUDevice, type, int32, Eigen::internal::SumReducer<type>>); \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("Sum")                                                              \
          .Device(DEVICE_GPU)                                                  \
          .TypeConstraint<type>("T")                                           \
          .TypeConstraint<int64>("Tidx")                                       \
          .HostMemory("reduction_indices"),                                    \
      ReductionOp<GPUDevice, type, int64, Eigen::internal::SumReducer<type>>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_complex64(REGISTER_GPU_KERNELS);
TF_CALL_complex128(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("T")
        .TypeConstraint<int32>("Tidx")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, int32, Eigen::internal::SumReducer<int32>>);
REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_GPU)
        .TypeConstraint<int32>("T")
        .TypeConstraint<int64>("Tidx")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, int64, Eigen::internal::SumReducer<int32>>);

#endif

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(Name("Sum")                                      \
                              .Device(DEVICE_SYCL)                         \
                              .TypeConstraint<type>("T")                   \
                              .TypeConstraint<int32>("Tidx")               \
                              .HostMemory("reduction_indices"),            \
                          ReductionOp<SYCLDevice, type, int32,             \
                                      Eigen::internal::SumReducer<type>>); \
  REGISTER_KERNEL_BUILDER(Name("Sum")                                      \
                              .Device(DEVICE_SYCL)                         \
                              .TypeConstraint<type>("T")                   \
                              .TypeConstraint<int64>("Tidx")               \
                              .HostMemory("reduction_indices"),            \
                          ReductionOp<SYCLDevice, type, int64,             \
                                      Eigen::internal::SumReducer<type>>);
REGISTER_SYCL_KERNELS(float);
REGISTER_SYCL_KERNELS(double);

REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_SYCL)
        .TypeConstraint<int32>("T")
        .TypeConstraint<int32>("Tidx")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, int32, Eigen::internal::SumReducer<int32>>);
REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_SYCL)
        .TypeConstraint<int32>("T")
        .TypeConstraint<int64>("Tidx")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, int64, Eigen::internal::SumReducer<int32>>);
#undef REGISTER_SYCL_KERNELS
#endif  // TENSORFLOW_USE_SYCL

#ifdef TENSORFLOW_USE_VE
template <typename T, typename Tperm>
class VEReductionOp : public OpKernel {
 public:
  explicit VEReductionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    const DataType dt = DataTypeToEnum<T>::v();
    const DataType pt = DataTypeToEnum<Tperm>::v();
    OP_REQUIRES_OK(ctx, ctx->MatchSignature({dt, pt}, {dt}));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("keep_dims", &keep_dims_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& data = ctx->input(0);
    const Tensor& axes = ctx->input(1);
    VLOG(1) << "data shape: " << data.shape().DebugString();
    VLOG(1) << "axes      : " << axes.SummarizeValue(10);

    ReductionHelper helper;
    OP_REQUIRES_OK(ctx, helper.Simplify(data, axes, keep_dims_));
    CHECK_GE(helper.ndims(), 0);

    if (helper.ndims() == 0 ||
        (helper.ndims() == 1 && !helper.reduce_first_axis())) {
      // Special case. Reduces nothing.  It is unclear why this is
      // necessary, but tests fail without it.  Look into why this
      // case occurs.
      Tensor out;
      if (!out.CopyFrom(data, helper.out_shape())) {
        ctx->SetStatus(errors::Internal("Error during reduction copy."));
      }
      ctx->set_output(0, out);
      return;
    }

    // We must allocate temp tensors using the same alloc attr as
    // output(0) because it is returned as output(0) in the end.
    const AllocatorAttributes alloc_attr = ctx->output_alloc_attr(0);

    // A temporary tensor whose size matches the size of the reduced
    // output.
    Tensor tmp_out;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(ctx->expected_output_dtype(0),
                                helper.out_reshape(), &tmp_out, alloc_attr));

#if 1
#if 0
    typedef functor::ReduceFunctor<Device, Reducer> Functor;
    Constants<Device> constants;
    const Device& d = ctx->eigen_device<Device>();
    Reducer reducer;
#endif

    if (tmp_out.NumElements() == 0) {
      // Nothing to do, fall through to final reshaping.
#if 0
    } else if (data.NumElements() == 0) {
      // Degenerate reduction where the input is empty but the output is
      // nonempty (thus tmp_out.NumElements() > 0), and we must fill the output
      // with identity elements.  Example: tf.reduce_sum(tf.zeros((0, 3)), [0]).
      // Eigen sometimes crashes in this case, so we do it manually.
      Functor::FillIdentity(d, tmp_out.flat<T>(), reducer);
    } else if ((helper.ndims() == 1) && helper.reduce_first_axis()) {
      // Reduce to a scalar.
      Functor::Reduce(ctx, helper.out<T, 0>(&tmp_out), helper.in<T, 1>(data),
                      constants.kZero, reducer);
    } else if ((helper.ndims() == 2) && helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a matrix along 1st dimension.
      Functor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                      constants.kZero, reducer);
#endif
    } else if ((helper.ndims() == 2) && !helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a matrix along 2nd dimension.
#if 0
      Functor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 2>(data),
                      constants.kOne, reducer);
#else
      struct {
        int dtype;
        int ndims;
        uint64_t in;
        uint64_t out;
        int64_t dim_size[2];
        int axis;
      } args;

      args.dtype = data.dtype();
      args.ndims = helper.ndims();
      args.in = (uint64_t)DMAHelper::base(&data);
      args.out = (uint64_t)DMAHelper::base(&tmp_out);
      args.dim_size[0] = helper.data_reshape().dim_size(0);
      args.dim_size[1] = helper.data_reshape().dim_size(1);
      args.axis = 1;

      VEDeviceContext* vectx = ctx->op_device_context<VEDeviceContext>();
      Status s = vectx->Compute("Sum", (void*)&args, sizeof(args));
      if (!s.ok())
        ctx->SetStatus(s);
#endif
#if 0
    } else if ((helper.ndims() == 3) && helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a 3D tensor along 1st and 3rd
      // dimensions.
      Functor::Reduce(ctx, helper.out<T, 1>(&tmp_out), helper.in<T, 3>(data),
                      constants.kZeroTwo, reducer);
    } else if ((helper.ndims() == 3) && !helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a 3D tensor along 2nd dimension.
      Functor::Reduce(ctx, helper.out<T, 2>(&tmp_out), helper.in<T, 3>(data),
                      constants.kOne, reducer);
#endif
    } else {
#if 0
      // If we don't hit one of the cases above, transpose the data so that
      // all reduced dimensions are last and reuse the 2-D -> 1-D case.
      Tensor data_reshaped;
      CHECK(data_reshaped.CopyFrom(data, helper.data_reshape()));
      Tensor shuffled;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             helper.shuffled_shape(), &shuffled,
                                             alloc_attr));
      OP_REQUIRES_OK(
          ctx, DoTranspose(d, data_reshaped, helper.permutation(), &shuffled));
      const int64 unreduced = tmp_out.NumElements();
      const int64 reduced = shuffled.NumElements() / unreduced;
      const Tensor& const_shuffled = shuffled;
      Functor::Reduce(ctx, tmp_out.flat<T>(),
                      const_shuffled.shaped<T, 2>({unreduced, reduced}),
                      constants.kOne, reducer);
#else
      ctx->SetStatus(errors::Unimplemented("Unsupported reduction on VE"));
#endif
    }
#endif

    // Set the real output using the contents of the reduction but the
    // real expected output shape.  The number of elements should
    // match between the two shapes.
    Tensor out;
    if (!out.CopyFrom(tmp_out, helper.out_shape())) {
      ctx->SetStatus(errors::Internal("Error during reduction copy."));
    }
    ctx->set_output(0, out);
  }

 private:
  // True if the number of dimensions should be maintained.
  bool keep_dims_;
};

#define REGISTER_VE_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(Name("Sum")                                      \
                              .Device(DEVICE_VE)                         \
                              .TypeConstraint<type>("T")                   \
                              .TypeConstraint<int32>("Tidx")               \
                              .HostMemory("reduction_indices"),            \
                          VEReductionOp<type, int32>);                     \
  REGISTER_KERNEL_BUILDER(Name("Sum")                                      \
                              .Device(DEVICE_VE)                         \
                              .TypeConstraint<type>("T")                   \
                              .TypeConstraint<int64>("Tidx")               \
                              .HostMemory("reduction_indices"),            \
                          VEReductionOp<type, int64>);
REGISTER_VE_KERNELS(float);
REGISTER_VE_KERNELS(double);

REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_VE)
        .TypeConstraint<int32>("T")
        .TypeConstraint<int32>("Tidx")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, int32, Eigen::internal::SumReducer<int32>>);
REGISTER_KERNEL_BUILDER(
    Name("Sum")
        .Device(DEVICE_VE)
        .TypeConstraint<int32>("T")
        .TypeConstraint<int64>("Tidx")
        .HostMemory("input")
        .HostMemory("output")
        .HostMemory("reduction_indices"),
    ReductionOp<CPUDevice, int32, int64, Eigen::internal::SumReducer<int32>>);
#undef REGISTER_VE_KERNELS
#endif  // TENSORFLOW_USE_VE

}  // namespace tensorflow

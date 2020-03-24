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

// See docs in ../ops/nn_ops.cc.

#include "tensorflow/core/lib/strings/str_util.h"
#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/softmax_op_functor.h"

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/framework/ve_ops_common.h"
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

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from SoftmaxEigenImpl.
namespace functor {
template <typename Device, typename T>
struct SoftmaxFunctorBase {
  void operator()(const Device& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::Matrix softmax, const bool log) {
    SoftmaxEigenImpl<Device, T>::Compute(d, logits, softmax, log);
  }
};
template <typename T>
struct SoftmaxFunctor<CPUDevice, T> : SoftmaxFunctorBase<CPUDevice, T> {};

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
struct SoftmaxFunctor<SYCLDevice, T> : SoftmaxFunctorBase<SYCLDevice, T> {};
#endif  // TENSORFLOW_USE_SYCL
}  // namespace functor

template <typename Device, typename T>
class SoftmaxOp : public OpKernel {
 public:
  explicit SoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {
    log_ = absl::StartsWith(type_string(), "Log");
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(logits_in.shape()),
                errors::InvalidArgument("logits must have >= 1 dimension, got ",
                                        logits_in.shape().DebugString()));
    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, logits_in.shape(), &softmax_out));
    if (logits_in.NumElements() > 0) {
      functor::SoftmaxFunctor<Device, T> functor;
      functor(context->eigen_device<Device>(), logits_in.flat_inner_dims<T>(),
              softmax_out->flat_inner_dims<T>(), log_);
    }
  }

 private:
  bool log_;
};

#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Softmax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SoftmaxOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU
#define REGISTER_CPU(T)                                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("LogSoftmax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SoftmaxOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU);
TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_SYCL).TypeConstraint<float>("T"),
    SoftmaxOp<SYCLDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_SYCL).TypeConstraint<double>("T"),
    SoftmaxOp<SYCLDevice, double>);
#endif  // TENSORFLOW_USE_SYCL

#ifdef TENSORFLOW_USE_VE

template <typename T>
class SoftmaxOp<VEDevice, T> : public VEOpKernel {
 public:
  explicit SoftmaxOp(OpKernelConstruction* context) : VEOpKernel(context) {
    log_ = str_util::StartsWith(type_string(), "Log");
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(logits_in.shape()),
                errors::InvalidArgument("logits must have >= 1 dimension, got ",
                                        logits_in.shape().DebugString()));
    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, logits_in.shape(), &softmax_out));

    ArgsImpl<> Args = ArgsImpl<>() ;
    Args.addArg<Tensor>(logits_in) ;
    Args.addArg<Tensor>(*softmax_out) ;
    Args.addArg<int64_t>(log_?1:0) ;

    Call(context, "Softmax", Args) ;

  }

 private:
  bool log_;

};


REGISTER_KERNEL_BUILDER(
    Name("Softmax").Device(DEVICE_VE).TypeConstraint<float>("T"),
    SoftmaxOp<VEDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("LogSoftmax").Device(DEVICE_VE).TypeConstraint<float>("T"),
    SoftmaxOp<VEDevice, float>);

#endif // TENSORFLOW_USE_VE


}  // namespace tensorflow

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/relu_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/common_runtime/ve/ve_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/ve_ops_common.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_VE
typedef Eigen::VeDevice VEDevice;
#endif  // TENSORFLOW_USE_VE

#define REGISTER_RELU_KERNELS(type)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      ReluOp<CPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      ReluGradOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6").Device(DEVICE_CPU).TypeConstraint<type>("T"),         \
      Relu6Op<CPUDevice, type>);                                          \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6Grad").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      Relu6GradOp<CPUDevice, type>)                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyRelu").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      LeakyReluOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      LeakyReluGradOp<CPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_RELU_KERNELS);
#undef REGISTER_RELU_KERNELS

#define REGISTER_ELU_KERNELS(type)                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Elu").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      EluOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("EluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
      EluGradOp<CPUDevice, type>);                                   \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Selu").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      SeluOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SeluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SeluGradOp<CPUDevice, type>)

// Elu and Selu only make sense with float or double.
TF_CALL_FLOAT_TYPES(REGISTER_ELU_KERNELS);
#undef REGISTER_ELU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void Relu<GPUDevice, T>::operator()(                                         \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Relu<GPUDevice, T>;                                   \
                                                                               \
  template <>                                                                  \
  void ReluGrad<GPUDevice, T>::operator()(                                     \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features,                                \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct ReluGrad<GPUDevice, T>;                               \
                                                                               \
  template <>                                                                  \
  void Relu6<GPUDevice, T>::operator()(                                        \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Relu6<GPUDevice, T>;                                  \
                                                                               \
  template <>                                                                  \
  void Relu6Grad<GPUDevice, T>::operator()(                                    \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features,                                \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct Relu6Grad<GPUDevice, T>;                              \
                                                                               \
  template <>                                                                  \
  void LeakyRelu<GPUDevice, T>::operator()(LeakyReluArgs args);                \
  extern template struct LeakyRelu<GPUDevice, T>;                              \
                                                                               \
  template <>                                                                  \
  void LeakyReluGrad<GPUDevice, T>::operator()(                                \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor features, T alpha,                       \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct LeakyReluGrad<GPUDevice, T>;                          \
                                                                               \
  template <>                                                                  \
  void Elu<GPUDevice, T>::operator()(const GPUDevice& d,                       \
                                     typename TTypes<T>::ConstTensor features, \
                                     typename TTypes<T>::Tensor activations);  \
  extern template struct Elu<GPUDevice, T>;                                    \
                                                                               \
  template <>                                                                  \
  void EluGrad<GPUDevice, T>::operator()(                                      \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor activations,                             \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct EluGrad<GPUDevice, T>;                                \
                                                                               \
  template <>                                                                  \
  void Selu<GPUDevice, T>::operator()(                                         \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features,            \
      typename TTypes<T>::Tensor activations);                                 \
  extern template struct Selu<GPUDevice, T>;                                   \
                                                                               \
  template <>                                                                  \
  void SeluGrad<GPUDevice, T>::operator()(                                     \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,           \
      typename TTypes<T>::ConstTensor activations,                             \
      typename TTypes<T>::Tensor backprops);                                   \
  extern template struct SeluGrad<GPUDevice, T>;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <>
void Relu<GPUDevice, qint8>::operator()(
    const GPUDevice& d, typename TTypes<qint8>::ConstTensor features,
    typename TTypes<qint8>::Tensor activations);
extern template struct Relu<GPUDevice, qint8>;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu").Device(DEVICE_GPU).TypeConstraint<type>("T"),          \
      ReluOp<GPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      ReluGradOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6").Device(DEVICE_GPU).TypeConstraint<type>("T"),         \
      Relu6Op<GPUDevice, type>);                                          \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Relu6Grad").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      Relu6GradOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyRelu").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      LeakyReluOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      LeakyReluGradOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Elu").Device(DEVICE_GPU).TypeConstraint<type>("T"),           \
      EluOp<GPUDevice, type>);                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("EluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),       \
      EluGradOp<GPUDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Selu").Device(DEVICE_GPU).TypeConstraint<type>("T"),          \
      SeluOp<GPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SeluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      SeluGradOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

template <typename Device>
class ReluOp<Device, qint8>
    : public UnaryElementWiseOp<qint8, ReluOp<Device, qint8>> {
 public:
  using UnaryElementWiseOp<qint8, ReluOp<Device, qint8>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    auto flat_input = input.flat<qint8>();
    OP_REQUIRES(context, (flat_input.size() % 4) == 0,
                errors::InvalidArgument(
                    "Tensor size must be a multiple of 4 for Relu<qint8>. Got ",
                    flat_input.size()));
    functor::Relu<Device, qint8> func;
    func(context->eigen_device<Device>(), flat_input, output->flat<qint8>());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Relu").Device(DEVICE_GPU).TypeConstraint<qint8>("T"),
    ReluOp<GPUDevice, qint8>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_VE

template <typename T>
class ReluOp<VEDevice, T> : public UnaryElementWiseOp<T, ReluOp<VEDevice, T>> {
  public:
    using UnaryElementWiseOp<T, ReluOp<VEDevice, T>>::UnaryElementWiseOp;

    void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
      struct {
        int dtype;
        uint64_t in;
        uint64_t out;
        uint64_t num_elems;
      } args;

      args.dtype = DataTypeToEnum<T>::v();
      args.in = (uint64_t)DMAHelper::base(&input);
      args.out = (uint64_t)DMAHelper::base(output);
      args.num_elems = input.NumElements();

      VEDeviceContext* vectx = context->op_device_context<VEDeviceContext>();
      Status s = vectx->Compute("Relu", (void*)&args, sizeof(args));
      if (!s.ok())
        context->SetStatus(s);
    }
};

template <typename T>
class ReluGradOp<VEDevice, T> : public BinaryElementWiseOp<T, ReluGradOp<VEDevice, T>> {
  public:
    using BinaryElementWiseOp<T, ReluGradOp<VEDevice, T>>::BinaryElementWiseOp;

    template <int NDIMS>
      void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
                   Tensor* output) {
      struct {
        int dtype;
        uint64_t g;
        uint64_t a;
        uint64_t output;
        uint64_t num_elems;
      } args;

      args.dtype = DataTypeToEnum<T>::v();
      args.g = (uint64_t)DMAHelper::base(&g);
      args.a = (uint64_t)DMAHelper::base(&a);
      args.output = (uint64_t)DMAHelper::base(output);
      args.num_elems = g.NumElements();

      VEDeviceContext* vectx = context->op_device_context<VEDeviceContext>();
      Status s = vectx->Compute("ReluGrad", (void*)&args, sizeof(args));
      if (!s.ok())
        context->SetStatus(s);
      }
};

#define REGISTER_VE_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Relu").Device(DEVICE_VE).TypeConstraint<type>("T"),      \
      ReluOp<VEDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("ReluGrad").Device(DEVICE_VE).TypeConstraint<type>("T"),  \
      ReluGradOp<VEDevice, type>);
#if 0
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_VE_KERNELS);
#else
TF_CALL_float(REGISTER_VE_KERNELS)
//TF_CALL_double(REGISTER_VE_KERNELS)
#endif
#undef REGISTER_VE_KERNELS


template <typename T>
class Relu6Op<VEDevice, T> : public UnaryElementWiseOp<T, Relu6Op<VEDevice, T>> {
 public:
  using UnaryElementWiseOp<T, Relu6Op<VEDevice, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
      VEOpKernelHelper::ArgsImpl<> args;
      args.addArg<Tensor>(input);
      args.addArg<Tensor>(*output);
      VEOpKernelHelper::Call(context, "Relu6", args);
  }
};

template <typename T>
class Relu6GradOp<VEDevice, T> : public BinaryElementWiseOp<T, Relu6GradOp<VEDevice, T>> {
 public:
  using BinaryElementWiseOp<T, Relu6GradOp<VEDevice, T>>::BinaryElementWiseOp;

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (inputs): inputs that were passed to Relu6Op()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
      VEOpKernelHelper::ArgsImpl<> args;
      args.addArg<Tensor>(g);
      args.addArg<Tensor>(a);
      args.addArg<Tensor>(*output);
      VEOpKernelHelper::Call(context, "Relu6Grad", args);
  }
};

#define REGISTER_VE_KERNELS(type)                                    	     \
  REGISTER_KERNEL_BUILDER(                                                   \
	  Name("Relu6").Device(DEVICE_VE).TypeConstraint<type>("T"),         \
	  Relu6Op<VEDevice, type>);                                          \
  REGISTER_KERNEL_BUILDER(                                                   \
	  Name("Relu6Grad").Device(DEVICE_VE).TypeConstraint<type>("T"),     \
	  Relu6GradOp<VEDevice, type>);

#if 0
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_VE_KERNELS);
#else
TF_CALL_float(REGISTER_VE_KERNELS)
//TF_CALL_double(REGISTER_VE_KERNELS)
#endif
#undef REGISTER_VE_KERNELS



template <typename T>
class LeakyReluOp<VEDevice, T> : public UnaryElementWiseOp<T, LeakyReluOp<VEDevice, T>> {
  public:
    //using UnaryElementWiseOp<T, ReluOp<VEDevice, T>>::UnaryElementWiseOp;

    explicit LeakyReluOp(OpKernelConstruction* context)
      : UnaryElementWiseOp<T, LeakyReluOp<VEDevice, T>>(context) {
        float alpha_tmp;
        OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_tmp));
        alpha_ = T(alpha_tmp);
      }

    void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
      VEOpKernelHelper::ArgsImpl<> args;

      args.addArg<Tensor>(input);
      args.addArg<Tensor>(*output);
      args.addArg<double>((double)alpha_);

      VEOpKernelHelper::Call(context, "LeakyRelu", args);
    }

  private:
    T alpha_;
};

template <typename T>
class LeakyReluGradOp<VEDevice, T> 
        : public BinaryElementWiseOp<T, LeakyReluGradOp<VEDevice, T>> {
  public:
    //using BinaryElementWiseOp<T, ReluGradOp<VEDevice, T>>::BinaryElementWiseOp;
    explicit LeakyReluGradOp(OpKernelConstruction* context)
      : BinaryElementWiseOp<T, LeakyReluGradOp<VEDevice, T>>(context) {
        float alpha_tmp;
        OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha_tmp));
        alpha_ = T(alpha_tmp);
      }

    template <int NDIMS>
      void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
                   Tensor* output) {
        VEOpKernelHelper::ArgsImpl<> args;
        args.addArg<Tensor>(g);
        args.addArg<Tensor>(a);
        args.addArg<Tensor>(*output);
        args.addArg<double>((double)alpha_);

        VEOpKernelHelper::Call(context, "LeakyReluGrad", args);
      }

  private:
    T alpha_;
};

#define REGISTER_VE_KERNELS(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyRelu").Device(DEVICE_VE).TypeConstraint<type>("T"),      \
      LeakyReluOp<VEDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("LeakyReluGrad").Device(DEVICE_VE).TypeConstraint<type>("T"),  \
      LeakyReluGradOp<VEDevice, type>);
#if 0
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_VE_KERNELS);
#else
TF_CALL_float(REGISTER_VE_KERNELS)
//TF_CALL_double(REGISTER_VE_KERNELS)
#endif
#undef REGISTER_VE_KERNELS


template <typename T>
class EluOp<VEDevice,T> : public UnaryElementWiseOp<T, EluOp<VEDevice, T>> {
 public:
  using UnaryElementWiseOp<T, EluOp<VEDevice, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    VEOpKernelHelper::ArgsImpl<> args;

    args.addArg<Tensor>(input);
    args.addArg<Tensor>(*output);

    VEOpKernelHelper::Call(context, "Elu", args);
  }
};

template <typename T>
class EluGradOp<VEDevice,T> : public BinaryElementWiseOp<T, EluGradOp<VEDevice, T>> {
 public:
  using BinaryElementWiseOp<T, EluGradOp<VEDevice, T>>::BinaryElementWiseOp;

  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (outputs): outputs of the EluOp()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    VEOpKernelHelper::ArgsImpl<> args;
    args.addArg<Tensor>(g);
    args.addArg<Tensor>(a);
    args.addArg<Tensor>(*output);

    VEOpKernelHelper::Call(context, "EluGrad", args);
  }
};

#define REGISTER_VE_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Elu").Device(DEVICE_VE).TypeConstraint<type>("T"),       \
      EluOp<VEDevice, type>);                                        \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("EluGrad").Device(DEVICE_VE).TypeConstraint<type>("T"),   \
      EluGradOp<VEDevice, type>)
#if 0
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_VE_KERNELS);
#else
TF_CALL_float(REGISTER_VE_KERNELS)
//TF_CALL_double(REGISTER_VE_KERNELS)
#endif


template <typename T>
class SeluOp<VEDevice, T> : public UnaryElementWiseOp<T, SeluOp<VEDevice, T>> {
 public:
  using UnaryElementWiseOp<T, SeluOp<VEDevice, T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    VEOpKernelHelper::ArgsImpl<> args;

    args.addArg<Tensor>(input);
    args.addArg<Tensor>(*output);

    VEOpKernelHelper::Call(context, "Selu", args);
  }
};

template <typename T>
class SeluGradOp<VEDevice, T> : public BinaryElementWiseOp<T, SeluGradOp<VEDevice, T>> {
 public:
  using BinaryElementWiseOp<T, SeluGradOp<VEDevice, T>>::BinaryElementWiseOp;


  // INPUTS:
  //   g (gradients): backpropagated gradients
  //   a (outputs): outputs of the SeluOp()
  // OUTPUT:
  //   gradients to backprop
  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    VEOpKernelHelper::ArgsImpl<> args;
    args.addArg<Tensor>(g);
    args.addArg<Tensor>(a);
    args.addArg<Tensor>(*output);

    VEOpKernelHelper::Call(context, "SeluGrad", args);
  }
};

#define REGISTER_VE_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("Selu").Device(DEVICE_VE).TypeConstraint<type>("T"),      \
      SeluOp<VEDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("SeluGrad").Device(DEVICE_VE).TypeConstraint<type>("T"),  \
      SeluGradOp<VEDevice, type>)
#if 0
TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_VE_KERNELS);
#else
TF_CALL_float(REGISTER_VE_KERNELS)
//TF_CALL_double(REGISTER_VE_KERNELS)
#endif
#undef REGISTER_VE_KERNELS

#endif // TENSORFLOW_USE_VE

}  // namespace tensorflow

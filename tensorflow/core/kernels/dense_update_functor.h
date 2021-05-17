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

#ifndef TENSORFLOW_CORE_KERNELS_DENSE_UPDATE_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_DENSE_UPDATE_FUNCTOR_H_

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/common_runtime/ve/ve_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


enum DenseUpdateType { ADD, SUB, ASSIGN };

namespace functor {

template <typename Device, typename T, DenseUpdateType OP>
struct DenseUpdate {
  void operator()(const Device& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update);
};

template <typename T>
struct DenseUpdate<CPUDevice, T, ADD> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) += update;
  }
};

template <typename T>
struct DenseUpdate<CPUDevice, T, SUB> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) -= update;
  }
};

template <typename T>
struct DenseUpdate<CPUDevice, T, ASSIGN> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat params,
                  typename TTypes<T>::ConstFlat update) {
    params.device(d) = update;
  }
};


#ifdef TENSORFLOW_USE_VE
template <typename T, DenseUpdateType OP>
struct VEDenseUpdate {
  void operator()(OpKernelContext* ctx,
                  Tensor* params,
                  const Tensor* update);
};

// [todo] impl ADD, SUB
template <typename T>
struct VEDenseUpdate<T, ADD> {
  void operator()(OpKernelContext* ctx,
                  Tensor* params,
                  const Tensor* update) {
    struct {
      int dtype;
      int64_t num_elements ;
      uint64_t dst_ptr, src_ptr ;
    } args;

    args.dtype = update->dtype() ;
    args.num_elements = update->NumElements() ;
    args.dst_ptr = (uint64_t) DMAHelper::base(params) ;
    args.src_ptr = (uint64_t) DMAHelper::base(update) ;

    VEDeviceContext* vectx = ctx->op_device_context<VEDeviceContext>();
    Status s = vectx->Compute("DenseUpdateAssignAdd", (void*)&args, sizeof(args));
    if (!s.ok())
      ctx->SetStatus(s);
  }
};

template <typename T>
struct VEDenseUpdate<T, SUB> {
  void operator()(OpKernelContext* ctx,
                  Tensor* params,
                  const Tensor* update) {
    struct {
      int dtype;
      int64_t num_elements ;
      uint64_t dst_ptr, src_ptr ;
    } args;

    args.dtype = update->dtype() ;
    args.num_elements = update->NumElements() ;
    args.dst_ptr = (uint64_t) DMAHelper::base(params) ;
    args.src_ptr = (uint64_t) DMAHelper::base(update) ;

    VEDeviceContext* vectx = ctx->op_device_context<VEDeviceContext>();
    Status s = vectx->Compute("DenseUpdateAssignSub", (void*)&args, sizeof(args));
    if (!s.ok())
      ctx->SetStatus(s);
  }
};

template <typename T>
struct VEDenseUpdate<T, ASSIGN> {
  void operator()(OpKernelContext* ctx,
                  Tensor* params,
                  const Tensor* update) {

    struct {
      int dtype;
      int64_t num_elements ;
      uint64_t dst_ptr, src_ptr ;
    } args;

    args.dtype = update->dtype() ;
    args.num_elements = update->NumElements() ;
    args.dst_ptr = (uint64_t) DMAHelper::base(params) ;
    args.src_ptr = (uint64_t) DMAHelper::base(update) ;

    VEDeviceContext* vectx = ctx->op_device_context<VEDeviceContext>();
    Status s = vectx->Compute("DenseUpdateAssign", (void*)&args, sizeof(args));
    if (!s.ok())
      ctx->SetStatus(s);
  }

};

#endif // TENSORFLOW_USE_VE

}  // end namespace functor

template <typename Device>
Status VariantCopyFn(OpKernelContext* context, const Tensor& from, Tensor* to);

template <>
Status VariantCopyFn<CPUDevice>(OpKernelContext* context, const Tensor& from,
                                Tensor* to);
template <>
Status VariantCopyFn<GPUDevice>(OpKernelContext* context, const Tensor& from,
                                Tensor* to);

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DENSE_UPDATE_FUNCTOR_H_

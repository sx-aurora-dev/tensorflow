#ifndef TENSORFLOW_CORE_KERNELS_VE_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_VE_OPS_COMMON_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

#ifdef TENSORFLOW_USE_VE
#include "tensorflow/core/common_runtime/ve/ve_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

namespace tensorflow {

class VEOpKernelHelper {
  private:
    struct _Tensor {
      int32_t dtype;
      uint64_t addr;
      int32_t dims;
      int64_t nelems;
      int64_t dim_size[1];

      static size_t size(int dims_) {
        return sizeof(_Tensor) + sizeof(int64_t) * (dims_ - 1);
      }

      size_t size() const {
        return size(dims);
        //return sizeof(_Tensor) + sizeof(int64_t) * (dims - 1);
      }

      Status init(const Tensor& t) {
        dtype = t.dtype();
        addr = (uint64_t)DMAHelper::base(&t);
        dims = t.dims();
        nelems = t.NumElements();
        for (int i = 0; i < dims; ++i) {
          dim_size[i] = t.dim_size(i);
        }
        return Status::OK();
      }
    } __attribute__((__packed__));


    class ArgsBase {
      public:
        virtual const void* buf() const = 0;
        virtual size_t size() const = 0;
    };

  protected:
    template<int MAX_BUF_SIZE = 1024>
    class Args : public ArgsBase {
      public:
        Args() {
          curr_ = reinterpret_cast<uintptr_t>(buf_) + sizeof(int64_t);
          end_ = reinterpret_cast<uintptr_t>(buf_) + MAX_BUF_SIZE;
          pHeader_ = reinterpret_cast<Header*>(buf_);
          pHeader_->nTensors = 0;
        }

        Status addTensor(const Tensor& t) {
          _Tensor* p = reinterpret_cast<_Tensor*>(curr_);
          size_t size = _Tensor::size(t.dims());
          if (curr_ + size >= end_)
            return errors::Internal("buffer is too small");

          p->init(t);

          curr_ += size;
          ++pHeader_->nTensors;

          return Status::OK();
        }

        const void* buf() const { return reinterpret_cast<const void*>(buf_); }
        size_t size() const { return curr_ - reinterpret_cast<uintptr_t>(buf_); }

      private:
        char buf_[MAX_BUF_SIZE];
        uintptr_t curr_;
        uintptr_t end_;
        struct Header {
          int64_t nTensors;
        };
        Header* pHeader_;
    };

  public:
    void Call(OpKernelContext* context, const std::string& name, ArgsBase& buf) {
      VEDeviceContext* vectx = context->op_device_context<VEDeviceContext>();
      Status s = vectx->Compute(name.c_str(), buf.buf(), buf.size());
      if (!s.ok())
        context->SetStatus(s);
    }
};

class VEOpKernel : public OpKernel, public VEOpKernelHelper {
  public:
    VEOpKernel(OpKernelConstruction* context) : OpKernel(context) {}
};

}; // namespace tensorflow

#endif

#endif

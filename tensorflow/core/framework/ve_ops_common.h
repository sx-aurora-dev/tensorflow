#ifndef TENSORFLOW_CORE_KERNELS_VE_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_VE_OPS_COMMON_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

#ifdef TENSORFLOW_USE_VE

namespace tensorflow {

class VEOpKernelHelper {
  public:
    class Args {
      public:
        Args(void* buf, size_t size) : buf_(buf) {
          curr_ = reinterpret_cast<uintptr_t>(buf_) + sizeof(Header);
          end_ = reinterpret_cast<uintptr_t>(buf_) + size;
          pHeader_ = reinterpret_cast<Header*>(buf_);
          pHeader_->nVariables = 0;
        }

        template<typename T>
        Status addArg(const T& v) ;

        const void* buf() const { return buf_; }
        size_t size() const { return curr_ - reinterpret_cast<uintptr_t>(buf_); }

      private:
        void* buf_;
        uintptr_t curr_;
        uintptr_t end_;
        struct Header {
          int64_t nVariables;
        };
        Header* pHeader_;
    };

    template<int MAX_BUF_SIZE = 1024>
      class ArgsImpl : public Args {
        public:
          ArgsImpl() : Args(buf_, MAX_BUF_SIZE) {}

          // shortcuts
          ArgsImpl(const Tensor& t0) : Args(buf_, MAX_BUF_SIZE) {
            addArg<Tensor>(t0);
          }

          ArgsImpl(const Tensor& t0, const Tensor& t1) : Args(buf_, MAX_BUF_SIZE) {
            addArg<Tensor>(t0);
            addArg<Tensor>(t1);
          }

        private:
          char buf_[MAX_BUF_SIZE];
      };

  public:
    static void Call(OpKernelContext* context, const std::string& name, const Args& buf,
                     const OpKernel* op = NULL);
    // shortcuts
    static void Call(OpKernelContext* context, const std::string& name,
              const Tensor& t0, const OpKernel* op = NULL) {
      Call(context, name, ArgsImpl<>(t0), op);
    }
    static void Call(OpKernelContext* context, const std::string& name,
              const Tensor& t0, const Tensor& t1, const OpKernel* op = NULL)
    {
      Call(context, name, ArgsImpl<>(t0, t1), op);
    }
};

class VEOpKernel : public OpKernel, public VEOpKernelHelper {
  public:
    VEOpKernel(OpKernelConstruction* context) : OpKernel(context) {}
};

}; // namespace tensorflow

#endif

#endif

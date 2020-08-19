#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_VE_VE_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_VE_VE_DEVICE_H_

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class VEDeviceContext : public DeviceContext {
  public:
    virtual ~VEDeviceContext() {}

    virtual Status Compute(const std::string& name, const void* arg, size_t len,
                           const OpKernel* op = nullptr) = 0;
};

}

#endif

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_VE_VE_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_VE_VE_DEVICE_H_

#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

class VEDeviceContext : public DeviceContext {
  public:
    virtual ~VEDeviceContext() {}

    virtual void Compute(const std::string& name, const void* arg, size_t len) = 0;
};

}

#endif

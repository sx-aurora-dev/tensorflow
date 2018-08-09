#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_VE_VE_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_VE_VE_DEVICE_H_

#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

class VEO;

class VEDeviceContext : public DeviceContext {
  public:
    VEDeviceContext(VEO* veo) : veo_(veo) {}

    void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
        Tensor* device_tensor,
        StatusCallback done) const override;

    void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
        Device* device, Tensor* cpu_tensor,
        StatusCallback done) override;

    void conv2d(const Tensor& input, const Tensor& filter, Tensor* output);

  private:
    VEO* veo_;
};

}

#endif

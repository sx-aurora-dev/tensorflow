#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/visitable_allocator.h"
#include "tensorflow/core/common_runtime/process_state.h"

namespace tensorflow {

class VEDeviceContext : public DeviceContext {
  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor,
                             StatusCallback done) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override;

};

void VEDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                                            Tensor* device_tensor,
                                            StatusCallback done) const {
  VLOG(2) << "VEDeviceContext::CopyCPUTensorToDevice";
  done(Status::OK());
}

void VEDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                                            Device* device, Tensor* cpu_tensor,
                                            StatusCallback done) {
  VLOG(2) << "VEDeviceContext::CopyDeviceTensorToCPU";
  done(Status::OK());
}

class VEDevice : public LocalDevice {
  public:
    VEDevice(const SessionOptions& options, const string name) :
      LocalDevice(options,
                  Device::BuildDeviceAttributes(name, "VE",
                                                Bytes(256 << 20),
                                                DeviceLocality())) {
        int numa_node = 0;
        cpu_allocator_ = ProcessState::singleton()->GetCPUAllocator(numa_node);
      }

    ~VEDevice() override;

    Status Init(const SessionOptions& options);
    Status Sync() override { return Status::OK(); }

    Allocator* GetAllocator(AllocatorAttributes attr) override {
      return cpu_allocator_; 
    }

  protected:
    Allocator* cpu_allocator_; // not owned

  private:
    GpuDeviceInfo* gpu_device_info_;
    std::vector<VEDeviceContext*> device_contexts_;
};

VEDevice::~VEDevice() {
  delete gpu_device_info_;
  for (auto ctx : device_contexts_) ctx->Unref();
}

Status VEDevice::Init(const SessionOptions& options) {
  VLOG(2) << "VEDevice::Init";
  device_contexts_.push_back(new VEDeviceContext);

  gpu_device_info_ = new GpuDeviceInfo;
  //gpu_device_info_->stream = streams_[0]->compute;
  gpu_device_info_->default_context = device_contexts_[0];
  set_tensorflow_gpu_device_info(gpu_device_info_);

  return Status::OK();
}

class VEDeviceFactory : public DeviceFactory {
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override {
    const string device_name = strings::StrCat(name_prefix, "/device:VE:0");
    VLOG(2) << "VEDeviceFactory::CreateDevices: " << device_name;
    VEDevice* device = new VEDevice(options, device_name);
    TF_RETURN_IF_ERROR(device->Init(options));
    devices->push_back(device);
    return Status::OK();
  }
};

#if 1
REGISTER_LOCAL_DEVICE_FACTORY("VE", VEDeviceFactory, 220);
#endif

} // namespace tensorflow


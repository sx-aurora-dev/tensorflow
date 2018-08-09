#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/visitable_allocator.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
//#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

#include "tensorflow/core/common_runtime/ve/ve_device.h"

#include "ve_offload.h"

extern "C" {
extern void set_fake_tid(long) __attribute__((weak));
}

namespace tensorflow {

class VEO {
  class Fake {
    public:
      Fake(int pid) { if (set_fake_tid) set_fake_tid(pid); }
      ~Fake() { if (set_fake_tid) set_fake_tid(0); }
  };

  public:
    VEO() {}
    virtual ~VEO();

#define FAKE() Fake fake(proc_pid_);

    void* alloc_mem(size_t size) {
      VLOG(2) << "VEO::alloc_mem: " << pthread_self();
      VLOG(2) << "VEO::alloc_mem: proc_=" << proc_ << " size=" << size;
      uint64_t addr;

      int ret;
      {
        FAKE();
        ret = veo_alloc_mem(proc_, &addr, size);
      }

      VLOG(2) << "VEO::alloc_mem: ret=" << ret << " addr=" << std::hex << addr;
      return (void*)addr;
    }

    int write_mem(uint64_t ve_addr, const void* vh_buff, size_t len) {
      FAKE();
      return veo_write_mem(proc_, ve_addr, vh_buff, len);
    }

    int read_mem(void* vh_buff, uint64_t ve_addr, size_t len) {
      FAKE();
      return veo_read_mem(proc_, vh_buff, ve_addr, len);
    }

    int conv2d(uint64_t addr, uint64_t len) {
      VLOG(2) << "VEO::conv2d";
      struct veo_args *argp = veo_args_alloc();
      veo_args_set_i64(argp, 0, addr);
      veo_args_set_i64(argp, 1, len);
      VLOG(2) << "VEO::call: ctx_=" << ctx_;

      uint64_t id = veo_call_async(ctx_, sym_, argp);
      VLOG(2) << "VEO::call: VEO request ID = " << id;

#if 0
      uint64_t buffer = 0;
      uint64_t bufptr = veo_get_sym(proc, handle, "buffer");
      int ret;
      ret = veo_read_mem(proc, &buffer, bufptr, sizeof(buffer));
      printf("veo_read_mem() returned %d\n", ret);
      printf("%016lx\n", buffer);
      buffer = 0xc0ffee;
      ret = veo_write_mem(proc, bufptr, &buffer, sizeof(buffer));
      printf("veo_write_mem() returned %d\n", ret);
      uint64_t sym2 = veo_get_sym(proc, handle, "print_buffer");
      uint64_t id2 = veo_call_async(ctx, sym2, argp);
      uint64_t retval;
#endif
      uint64_t retval;
      int ret = veo_call_wait_result(ctx_, id, &retval);
      VLOG(2) << "VEO::call: id=" << id << " ret=" << ret << " retval=" << retval;
    }

#undef FAKE

    void init();

  private:
    pid_t proc_pid_;
    struct veo_proc_handle* proc_;
    uint64_t lib_id_;
    uint64_t sym_;
    struct veo_thr_ctxt *ctx_;
};

void VEO::init() {
  const char* filename = getenv("VEO_LIB");
  VLOG(2) << "VEO::init: filename=" << filename;
  if (filename == NULL)
    return; // error

  int nodeid = 0;
  proc_ = veo_proc_create(nodeid);
  VLOG(2) << "VEO::init: proc_=" << proc_;

  proc_pid_ = getpid();

#if 0
  {
    uint64_t addr;
    VLOG(2) << "veo_alloc_mem: " << pthread_self();
    VLOG(2) << "veo_alloc_mem";
    int rc = veo_alloc_mem(proc_, &addr, 1024);
    VLOG(2) << "veo_alloc_mem: rc=" << rc;
  }
#endif

  lib_id_ = veo_load_library(proc_, filename);
  VLOG(2) << "VEO::init: lib_id_=" << lib_id_;

  //sym_ = veo_get_sym(proc_, lib_id_, "hello");
  sym_ = veo_get_sym(proc_, lib_id_, "conv2d");
  VLOG(2) << "VEO::init: sym_=" << (void*)sym_;

  ctx_ = veo_context_open(proc_);
  VLOG(2) << "VEO::init: ctx_=" << ctx_;
}

VEO::~VEO() {
  VLOG(2) << "VEO::~VEO";
  veo_context_close(ctx_);
  veo_proc_destroy(proc_);
}

namespace {

class VEMemAllocator : public SubAllocator {
  public:
    VEMemAllocator(VEO* veo) : veo_(veo) {}
    ~VEMemAllocator();
    void* Alloc(size_t alignments, size_t num_bytes);
    void Free(void* ptr, size_t num_bytes);

  private:
    VEO* veo_;
};

VEMemAllocator::~VEMemAllocator() {}

void* VEMemAllocator::Alloc(size_t alignments, size_t num_bytes) {
  VLOG(2) << "VEMemAllocator::Alloc: alignments=" << alignments << " num_bytes=" << num_bytes;
  return veo_->alloc_mem(num_bytes);
}

void VEMemAllocator::Free(void* ptr, size_t num_bytes) {
}

class VEBFCAllocator : public BFCAllocator {
  public:
    VEBFCAllocator(size_t total_memory, bool allow_growth, const string& name, 
        VEO* veo);

  private:
};

VEBFCAllocator::VEBFCAllocator(size_t total_memory, bool allow_growth, const string& name, VEO* veo) :
  BFCAllocator(new VEMemAllocator(veo), total_memory, allow_growth, name) {
  }

class VEDevice : public LocalDevice {
  public:
    VEDevice(const SessionOptions& options, const string name,
        Allocator* ve_allocator) :
      LocalDevice(options,
          Device::BuildDeviceAttributes(name, "VE",
            Bytes(256 << 20),
            DeviceLocality())),
      ve_allocator_(ve_allocator) {}

    ~VEDevice() override;

    Status Init(const SessionOptions& options, VEO* veo);
    Status Sync() override { return Status::OK(); }

    Allocator* GetAllocator(AllocatorAttributes attr) override {
      return ve_allocator_;
    }

  protected:
    Allocator* ve_allocator_;

  private:
    GpuDeviceInfo* gpu_device_info_;
    std::vector<VEDeviceContext*> device_contexts_;
};

VEDevice::~VEDevice() {
  delete gpu_device_info_;
  for (auto ctx : device_contexts_) ctx->Unref();
}

Status VEDevice::Init(const SessionOptions& options, VEO* veo) {
  VLOG(2) << "VEDevice::Init";
  //VEO& veo = initVEO();
  device_contexts_.push_back(new VEDeviceContext(veo));

  VLOG(2) << "VEDevice::Init DeviceContext=" << device_contexts_.back();

  gpu_device_info_ = new GpuDeviceInfo;
#if 0
  gpu_device_info_->stream = stream;
#endif
  gpu_device_info_->default_context = device_contexts_[0];
  set_tensorflow_gpu_device_info(gpu_device_info_);

  return Status::OK();
}

class VEDeviceFactory : public DeviceFactory {
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
      std::vector<Device*>* devices) override {
    const string device_name = strings::StrCat(name_prefix, "/device:VE:0");
    VLOG(2) << "VEDeviceFactory::CreateDevices: " << device_name;

    VEO* veo = new VEO();
    veo->init();

    size_t total_memory = 20UL*1024*1024*1024;
    Allocator* ve_allocator = new VEBFCAllocator(total_memory, true, "VE_0_bfc", veo);

    VEDevice* device = new VEDevice(options, device_name, ve_allocator);
    TF_RETURN_IF_ERROR(device->Init(options, veo));
    devices->push_back(device);
    return Status::OK();
  }
};

}

REGISTER_LOCAL_DEVICE_FACTORY("VE", VEDeviceFactory, 220);

void VEDeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
    Tensor* device_tensor,
    StatusCallback done) const {
  VLOG(2) << "VEDeviceContext::CopyCPUTensorToDevice";

  const void* in = DMAHelper::base(cpu_tensor);
  void* out = DMAHelper::base(device_tensor);

  VLOG(2) << "VEDeviceContext::CopyCPUTensorToDevice: in=" << in << " out=" << out;

  int rc = veo_->write_mem((uint64_t)out, in, cpu_tensor->TotalBytes());
  VLOG(2) << "VEDeviceContext::CopyCPUTensorToDevice: rc=" << rc;

  done(Status::OK());
}

void VEDeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
    Device* device, Tensor* cpu_tensor,
    StatusCallback done) {
  VLOG(2) << "VEDeviceContext::CopyDeviceTensorToCPU";

  const void* in = DMAHelper::base(device_tensor);
  void* out = DMAHelper::base(cpu_tensor);

  int rc = veo_->read_mem(out, (uint64_t)in, device_tensor->TotalBytes());

  VLOG(2) << "VEDeviceContext::CopyDeviceTensorToCPU: rc=" << rc;

  VLOG(2) << "VEDeviceContext::CopyDeviceTensorToCPU: " << (char*)out;

  done(Status::OK());
}

void VEDeviceContext::conv2d(const Tensor& input, const Tensor& filter, Tensor* output) {
  VLOG(2) << "VEDeviceContext::conv2d";
  void* out = DMAHelper::base(output);
  VLOG(2) << "VEDeviceContext::conv2d: out=" << out;

  veo_->conv2d((uint64_t)out, output->TotalBytes());
}

} // namespace tensorflow


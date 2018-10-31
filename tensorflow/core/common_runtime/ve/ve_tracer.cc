#include <stdlib.h>
#include <memory>

#include "tensorflow/core/platform/device_tracer.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#if 0
#include "tensorflow/core/lib/core/errors.h"
#endif
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#if 0
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#endif

namespace tensorflow {

typedef void (*cb_t)(int nodeid,
                     const std::vector<std::string>& kernel_names,
                     const void* buf,
                     void* data);

extern Status ve_get_timestamp(int nodeid, uint64_t* ts, double* resolution);
extern Status ve_set_trace_callback(int nodeid, cb_t cb, void* data);

class VEDeviceTracer : public DeviceTracer {
  public:
    VEDeviceTracer();

    ~VEDeviceTracer() { VLOG(2) << "~VEDeviceTracer"; }

    Status Start() override;
    Status Stop() override;
    Status Collect(StepStatsCollector *collector) override;

  private:
    struct KernelStats {
      int nodeid;
      std::string name;
      uint64_t t0;
      uint64_t t1;
    };

    int64 start_;
    uint64_t ve_start_timestamp_;
    double ve_resolution_;
    std::vector<KernelStats> stats_;
    mutex lock_;

    void callback(int nodeid, 
                  const std::vector<std::string>& kernel_names,
                  const void* buf);

    static void cb(int nodeid,
                   const std::vector<std::string>& kernel_names,
                   const void* buf,
                   void* self) {
      //VLOG(2) << "VEDeviceTracer::cb: self=" << self;
      reinterpret_cast<VEDeviceTracer*>(self)
        ->callback(nodeid, kernel_names, buf);
    }
};

void VEDeviceTracer::callback(int nodeid,
                                const std::vector<std::string>& kernel_names,
                                const void* buf)
{
  mutex_lock guard(lock_);
  VLOG(2) << "VEDeviceTracer::callback: stats_.size=" << stats_.size()
    << " kernel_names.size=" << kernel_names.size();

  uint64_t* pcyc = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uintptr_t>(buf) + sizeof(double));
  int n = kernel_names.size();
  for (int i = 0; i < n; ++i) {
    uint64_t t0 = pcyc[i*2];
    uint64_t t1 = pcyc[i*2+1];
#if 0
    VLOG(2) << "VEDeviceTracer::callback: kernel=" 
      << kernel_names[i] << " time=" << (t1-t0)*1e6/hz << " us";
#endif

    stats_.push_back(KernelStats{nodeid, kernel_names[i], t0, t1});
  }
}

VEDeviceTracer::VEDeviceTracer() {
#if 0
  VLOG(2) << "VEDeviceTracer::VEDeviceTracer:"
    " cb=" << reinterpret_cast<void*>(cb) << " this=" << this;
#endif
  int nodeid = 0; // FIXME
  ve_set_trace_callback(nodeid, cb, (void*)this);
  //VLOG(2) << "VEDeviceTracer::VEDeviceTracer done";
}

Status VEDeviceTracer::Start() { 
  //VLOG(2) << "VEDeviceTracer::Start";
  start_ = Env::Default()->NowMicros();
  int nodeid = 0; // FIXME
  Status s = ve_get_timestamp(nodeid, &ve_start_timestamp_, &ve_resolution_);
  VLOG(2) << "VEDeviceTracer::Start:"
    << " ve_start_timestamp_=" << ve_start_timestamp_
    << " ve_resolution_=" << ve_resolution_;
  if (!s.ok())
    return s;
  return Status::OK(); 
}

Status VEDeviceTracer::Stop() {
  VLOG(2) << "VEDeviceTracer::Stop";
  ve_set_trace_callback(0, nullptr, nullptr);
  return Status::OK(); 
}

Status VEDeviceTracer::Collect(StepStatsCollector *collector) {
  VLOG(2) << "VEDeviceTracer::Collect: stats_.size=" << stats_.size();

  mutex_lock guard(lock_);

  const string prefix = "";
  double mhz = ve_resolution_ / 1e6;

#if 0
  VLOG(2) << "VEDeviceTracer::Collect:"
    << " start_=" << start_
    << " ve_start_timestamp_=" << ve_start_timestamp_
    << " ve_resolution_=" << ve_resolution_;
#endif

  for (auto s : stats_) {
#if 0
    VLOG(2) << "VEDeviceTracer::Collect:"
      << " name=" << s.name
      << " t0=" << s.t0 << " t1=" << s.t1
      << " dur=" << (s.t1 - s.t0) / mhz;
#endif
    NodeExecStats *ns = new NodeExecStats;
    ns->set_all_start_micros(start_ + (s.t0 - ve_start_timestamp_) / mhz);
    ns->set_op_start_rel_micros(0);
    auto elapsed_us = (s.t1 - s.t0) / mhz;
    ns->set_op_end_rel_micros(elapsed_us);
    ns->set_all_end_rel_micros(elapsed_us);
    ns->set_node_name(strings::StrCat(s.name, ":", s.name));
#if 0
    VLOG(2) << "VEDeviceTracer::Collect:"
      << " start=" << ns->all_start_micros()
      << " end=" << ns->all_start_micros() + ns->all_end_rel_micros();
#endif
    const string stream_device =
      strings::StrCat(prefix, "/device:VE:", s.nodeid, "/stream:");
    collector->Save(strings::StrCat(stream_device, "all"), ns);
  }

  stats_.clear();
  //VLOG(2) << "VEDeviceTracer::Collect: after clear " << stats_.size();

  return Status::OK();
}


std::unique_ptr<DeviceTracer> CreateDeviceTracer() {
  VLOG(2) << "CreateDeviceTracer(VE)";
  std::unique_ptr<DeviceTracer> tracer(new VEDeviceTracer());
  return tracer;
}

} // tensorflow

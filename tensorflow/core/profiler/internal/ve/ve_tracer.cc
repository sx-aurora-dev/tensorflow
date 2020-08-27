#include <stdlib.h>
#include <memory>

#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/profiler/internal/profiler_factory.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

typedef void (*cb_t)(int nodeid, int kind, const void* data, void* self);

extern Status ve_get_timestamp(uint64_t* ts, double* resolution);
extern Status ve_set_trace_callback(cb_t cb, void* data);
extern Status ve_sync();

namespace profiler {

class VEDeviceTracer : public profiler::ProfilerInterface {
  public:
    VEDeviceTracer();

    ~VEDeviceTracer() override { VLOG(2) << "~VEDeviceTracer"; }

    Status Start() override;
    Status Stop() override;
    Status Export(XSpace* space);
    Status CollectData(RunMetadata* run_metadata) override;
    Status CollectData(XSpace* space) override;

  private:
    struct KernelRecords {
      int nodeid;
      std::string name;
      uint64_t start_ve_clock;
      uint64_t end_ve_clock;
    };

    struct MemcpyRecords {
      int nodeid;
      int type;
      uint64_t start_walltime_ns;
      uint64_t end_walltime_ns;
    };

    uint64 start_walltime_ns_;
    uint64_t ve_start_clock_;
    double ve_resolution_;
    std::vector<KernelRecords> kernel_records_;
    std::vector<MemcpyRecords> memcpy_records_;
    mutex lock_;

    void callback(int nodeid, int kind, const void* data);

    static void cb(int nodeid, int kind, const void* data, void* self) {
      //VLOG(2) << "VEDeviceTracer::cb: self=" << self;
      reinterpret_cast<VEDeviceTracer*>(self)
        ->callback(nodeid, kind, data);
    }
};

void VEDeviceTracer::callback(int nodeid, int kind, const void* data)
{
  mutex_lock guard(lock_);
  if (kind == 0) { // kenrel
    struct Tmp {
      const std::vector<std::string>* kernel_names;
      const void* buf;
    };
    const Tmp* tmp = reinterpret_cast<const Tmp*>(data);
    const std::vector<std::string>& kernel_names = *tmp->kernel_names;
    const void* buf = tmp->buf;

    VLOG(2) << "VEDeviceTracer::callback: kernel_records_.size=" << kernel_records_.size()
      << " kernel_names.size=" << kernel_names.size();

    const uint64_t* pcyc = reinterpret_cast<const uint64_t*>(buf);
    int n = kernel_names.size();
    for (int i = 0; i < n; ++i) {
      uint64_t t0 = pcyc[i*2];
      uint64_t t1 = pcyc[i*2+1];
#if 0
      VLOG(2) << "VEDeviceTracer::callback: kernel=" 
        << kernel_names[i] << " t0=" << t0 << " t1=" << t1;
#endif

      kernel_records_.push_back(KernelRecords{nodeid, kernel_names[i], t0, t1});
    }
  }
  else if (kind == 1) { // mempcy
    struct Tmp {
      uint64_t start;
      uint64_t end;
      uint64_t type;
    };
    const Tmp* tmp = reinterpret_cast<const Tmp*>(data);
    memcpy_records_.push_back(MemcpyRecords{nodeid, (int)tmp->type, tmp->start, tmp->end});
  }
}

VEDeviceTracer::VEDeviceTracer() {
#if 0
  VLOG(2) << "VEDeviceTracer::VEDeviceTracer:"
    " cb=" << reinterpret_cast<void*>(cb) << " this=" << this;
#endif
  ve_set_trace_callback(cb, (void*)this);
  //VLOG(2) << "VEDeviceTracer::VEDeviceTracer done";
}

Status VEDeviceTracer::Start() { 
  //VLOG(2) << "VEDeviceTracer::Start";
  start_walltime_ns_ = tensorflow::EnvTime::NowNanos();
  Status s = ve_get_timestamp(&ve_start_clock_, &ve_resolution_);
  VLOG(2) << "VEDeviceTracer::Start:"
    << " ve_start_clock_=" << ve_start_clock_
    << " ve_resolution_=" << ve_resolution_
    << " start_walltime_ns_=" << start_walltime_ns_;
  if (!s.ok())
    return s;


  return Status::OK(); 
}

Status VEDeviceTracer::Stop() {
  VLOG(2) << "VEDeviceTracer::Stop";
  tensorflow::ve_sync();
  ve_set_trace_callback(nullptr, nullptr);
  return Status::OK(); 
}

Status VEDeviceTracer::Export(XSpace* space) {
  LOG(INFO) << "VEDeviceTracer has collected " << kernel_records_.size()
            << " events.";
  const absl::string_view kVePlanePrefix = "/device:VE:";
  const int32 kVePlaneBaseId = 0; // OK?

  mutex_lock guard(lock_);

  int device_ordinal = 0;
  double ghz = ve_resolution_ / 1e9;

  std::string name = absl::StrCat(kVePlanePrefix, device_ordinal);
  XPlaneBuilder device_plane(FindOrAddMutablePlaneWithName(space, name));
  device_plane.SetId(kVePlaneBaseId + device_ordinal);

  // Flush
  int64 line_id = 0; // stream_id in GPU
  XLineBuilder line = device_plane.GetOrCreateLine(line_id);
  line.SetName("Kernel");
  line.SetTimestampNs(start_walltime_ns_);

  for (auto s : kernel_records_) {
    XEventMetadata* event_metadata =
        device_plane.GetOrCreateEventMetadata(s.name);
    XEventBuilder xevent = line.AddEvent(*event_metadata);
    VLOG(2) << "VEDeviceTracer::Export: " << s.start_ve_clock
      << " " << s.end_ve_clock;
    uint64_t ve_start_ns = (s.start_ve_clock - ve_start_clock_) / ghz;
    uint64_t ve_end_ns = (s.end_ve_clock - ve_start_clock_) / ghz;
    uint64_t walltime_start_ns = start_walltime_ns_ + ve_start_ns;
    uint64_t walltime_end_ns = start_walltime_ns_ + ve_end_ns;
    xevent.SetTimestampNs(walltime_start_ns);
    xevent.SetEndTimestampNs(walltime_end_ns);
  }

  {
    std::string str_type[] = {"MemcpyH2D", "MemcpyD2H"};

    for (auto s : memcpy_records_) {
      VLOG(2) << "VEDeviceTracer::Export: memcpy "
        << s.type << " " << s.start_walltime_ns << " " << s.end_walltime_ns;

      XLineBuilder line = device_plane.GetOrCreateLine(1 + s.type);
      line.SetName(str_type[s.type]);
      line.SetTimestampNs(start_walltime_ns_);

      XEventMetadata* event_metadata =
        device_plane.GetOrCreateEventMetadata(str_type[s.type]);
      XEventBuilder xevent = line.AddEvent(*event_metadata);
      xevent.SetTimestampNs(s.start_walltime_ns);
      xevent.SetEndTimestampNs(s.end_walltime_ns);
    }
  }

  return Status::OK();
}

Status VEDeviceTracer::CollectData(RunMetadata* run_metadata) {
  VLOG(2) << "VEDeviceTracer::CollectData(RunMetadata)";
  return errors::Unimplemented("Collect data into RunMetadata not implemented");
}

Status VEDeviceTracer::CollectData(XSpace* space) {
  VLOG(2) << "VEDeviceTracer::CollectData(XSpace)";
  Export(space);
  return Status::OK();
}

} // namespace profiler

std::unique_ptr<profiler::ProfilerInterface>
CreateVeTracer(const ProfileOptions&) {
  if (const char* tmp = getenv("VE_NODE_NUMBER")) {
    if (atoi(tmp) < 0) {
      return nullptr;
    }
  }
  
  VLOG(2) << "CreateVeTracer";
  return absl::make_unique<profiler::VEDeviceTracer>();
}

auto register_device_tracer_factory = [] {
  RegisterProfilerFactory(&CreateVeTracer);
  return 0;
}();

} // tensorflow

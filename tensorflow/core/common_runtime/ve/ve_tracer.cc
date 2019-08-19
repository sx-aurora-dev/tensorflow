#include <stdlib.h>
#include <memory>

//#include "tensorflow/core/platform/device_tracer.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
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
#endif
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

typedef void (*cb_t)(int nodeid, int kind, const void* data, void* self);

extern Status ve_get_timestamp(int nodeid, uint64_t* ts, double* resolution);
extern Status ve_set_trace_callback(int nodeid, cb_t cb, void* data);

#if 0 // copy from device_tracer.cc
#define TF_STATIC_THREAD_LOCAL_POD(_Type_, _var_)                  \
  static __thread _Type_ s_obj_##_var_;                            \
  namespace {                                                      \
  class ThreadLocal_##_var_ {                                      \
   public:                                                         \
    ThreadLocal_##_var_() {}                                       \
    void Init() {}                                                 \
    inline _Type_ *pointer() const { return &s_obj_##_var_; }      \
    inline _Type_ *safe_pointer() const { return &s_obj_##_var_; } \
    _Type_ &get() const { return s_obj_##_var_; }                  \
    bool is_native_tls() const { return true; }                    \
                                                                   \
   private:                                                        \
    TF_DISALLOW_COPY_AND_ASSIGN(ThreadLocal_##_var_);              \
  } _var_;                                                         \
  }  // namespace

// Thread-local state recording the most recent annotation (if any).
// When non-null, this points to a string in the active annotation
// of the current thread.  The annotation is guaranteed to remain live
// for the duration of the CUPTI API callback.
TF_STATIC_THREAD_LOCAL_POD(const char *, tls_current_annotation);

namespace {

class TraceCollectorImpl : public tracing::TraceCollector {
 public:
  TraceCollectorImpl() { tracing::SetTraceCollector(this); }

  ~TraceCollectorImpl() override {
    DCHECK(!active_trace_session_)
        << "Unexpected active trace session detected. ";
  }

  // Note the method can be called after a call to Stop().
  virtual std::unique_ptr<Handle> CreateAnnotationHandle(
      StringPiece name_part1, StringPiece name_part2) const {
    struct Impl : public tracing::TraceCollector::Handle {
      string annotation;
      explicit Impl(string &&name_scope) : annotation(name_scope) {
        VLOG(2) << "CreateAnnotationHandle " << annotation;
        // Remember the most recent ScopedAnnotation for each thread.
        tls_current_annotation.get() = annotation.c_str();
      }
      ~Impl() override { tls_current_annotation.get() = nullptr; }
    };
    return std::unique_ptr<Handle>(
        new Impl{ConcatenateNames(name_part1, name_part2)});
  }

  virtual std::unique_ptr<Handle> CreateActivityHandle(StringPiece, StringPiece,
                                                       bool) const {
    // We don't do anything with 'Activities' yet.
    return nullptr;
  }

  bool IsEnabledForAnnotations() const override {
    return active_trace_session_.load(std::memory_order_relaxed);
  }

  void Start() {
    DCHECK(!active_trace_session_)
        << "Unexpected active trace session detected. ";
    active_trace_session_ = true;
  }

  void Stop() {
    DCHECK(active_trace_session_) << "No active trace session detected. ";
    active_trace_session_ = false;
  }

 private:
  std::atomic<bool> active_trace_session_;
};

TraceCollectorImpl *GlobalDefaultTraceCollector() {
  static auto *instance = new TraceCollectorImpl();
  return instance;
}

} // namespace
#endif // copy from device_tracer.cc

class VEDeviceTracer : public profiler::ProfilerInterface {
  public:
    VEDeviceTracer();

    ~VEDeviceTracer() override { VLOG(2) << "~VEDeviceTracer"; }

    Status Start() override;
    Status Stop() override;
    Status Collect(StepStatsCollector* collector);
    Status CollectData(RunMetadata* run_metadata) override;

  private:
    struct KernelRecords {
      int nodeid;
      std::string name;
      uint64_t t0;
      uint64_t t1;
    };

    struct MemcpyRecords {
      int nodeid;
      std::string name;
      uint64_t start_timestamp;
      uint64_t end_timestamp;
    };

    int64 start_;
    uint64_t ve_start_timestamp_;
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
    std::string str_type[] = {"MEMCPYHtoD", "MEMCPYDtoH"};
#if 0
    const char* tls_annotation = tls_current_annotation.get();
    const char* annotation = tls_annotation ? tls_annotation : "unknown";
#else
    const char* annotation = "unknown";
#endif
    std::string name = strings::StrCat(annotation, ":", str_type[tmp->type]);

    memcpy_records_.push_back(MemcpyRecords{nodeid, name, tmp->start, tmp->end});
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

Status VEDeviceTracer::CollectData(RunMetadata* run_metadata) {
  StepStatsCollector collector(run_metadata->mutable_step_stats());
  TF_RETURN_IF_ERROR(Collect(&collector));
  collector.Finalize();
  return Status::OK();
}

Status VEDeviceTracer::Collect(StepStatsCollector *collector) {
  VLOG(2) << "VEDeviceTracer::Collect: kernel_records_.size=" << kernel_records_.size();

  mutex_lock guard(lock_);

  const string prefix = "";
  double mhz = ve_resolution_ / 1e6;

#if 0
  VLOG(2) << "VEDeviceTracer::Collect:"
    << " start_=" << start_
    << " ve_start_timestamp_=" << ve_start_timestamp_
    << " ve_resolution_=" << ve_resolution_;
#endif

  for (auto s : kernel_records_) {
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

  kernel_records_.clear();
  //VLOG(2) << "VEDeviceTracer::Collect: after clear " << kernel_records_.size();

  for (auto s : memcpy_records_) {
    auto elapsed_us = s.end_timestamp - s.start_timestamp;
    NodeExecStats *ns = new NodeExecStats;
    ns->set_all_start_micros(s.start_timestamp);
    ns->set_op_end_rel_micros(elapsed_us);
    ns->set_all_end_rel_micros(elapsed_us);
    ns->set_node_name(s.name);
    const string stream_device =
      strings::StrCat(prefix, "/device:VE:", s.nodeid, "/memcpy:");
    collector->Save(strings::StrCat(stream_device, "all"), ns);
  }

  return Status::OK();
}


std::unique_ptr<profiler::ProfilerInterface> CreateDeviceTracer() {
  if (const char* tmp = getenv("VE_NODE_NUMBER")) {
    if (atoi(tmp) < 0) {
      return nullptr;
    }
  }
  
  VLOG(2) << "CreateDeviceTracer(VE)";
  std::unique_ptr<profiler::ProfilerInterface> tracer(new VEDeviceTracer());
  return tracer;
}

auto register_device_tracer_factory = [] {
  RegisterProfilerFactory(&CreateDeviceTracer);
#if 0
  bool enable;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_OSS_GPU_PROFILER", true, &enable));
  if (enable) {
    RegisterProfilerFactory(&CreateDeviceTracer);
  }
#endif
  return 0;
}();

} // tensorflow

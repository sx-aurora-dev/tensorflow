#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"
//#include "tensorflow/core/common_runtime/visitable_allocator.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

#include "tensorflow/core/common_runtime/ve/ve_device.h"

#include "ve_offload.h"
#include <sys/types.h>
#include <sys/syscall.h>

#define VEO_ASYNC

#define TF_VE_EXECUTOR
#ifdef TF_VE_EXECUTOR
#include "tensorflow/core/lib/strings/str_util.h"
#endif

#define USE_DMA
#ifdef USE_DMA
#include <sys/ipc.h>
#include <sys/shm.h>
#endif

namespace tensorflow {

namespace {

class VEO {
  public:
    struct Args {
      struct veo_args* args; // FIXME: private
      Args() { args = veo_args_alloc(); }
      Args(const void* p, size_t len) {
        args = veo_args_alloc();
        set(p, len);
      }
      Args(const void* pIn, size_t lenIn, void* pOut, size_t lenOut) {
        args = veo_args_alloc();
        set(pIn, lenIn, pOut, lenOut);
      }
      ~Args() { veo_args_free(args); }
      void set(const void* p, size_t len) {
        veo_args_set_stack(args, VEO_INTENT_IN, 0, (char*)p, len);
        veo_args_set_i64(args, 1, len);
      }
      void set(const void* pIn, size_t lenIn, void* pOut, size_t lenOut) {
        veo_args_set_stack(args, VEO_INTENT_IN, 0, (char*)pIn, lenIn);
        veo_args_set_i64(args, 1, lenIn);
        veo_args_set_stack(args, VEO_INTENT_OUT, 2, (char*)pOut, lenOut);
        veo_args_set_i64(args, 3, lenOut);
      }
    };

    VEO() : cb_(nullptr) {}
    virtual ~VEO();

    typedef void (*cb_t)(int nodeid, int kind, const void* buf, void* data);

    bool isTracerEnabled() const { return cb_ != nullptr; }

    void setTraceCallback(cb_t cb, void* data) {
      VLOG(2) << "VEO::setTraceCallback: cb="
        << reinterpret_cast<void*>(cb) << " data=" << data;
      cb_ = cb;
      cb_data_ = data;
    }

    virtual Status get_timestamp(uint64_t* ts, double* resolution) {
      struct {
        uint64_t ts;
        double resolution;
      } tmp;
      size_t len = sizeof(tmp);

      Args a;
      veo_args_set_stack(a.args, VEO_INTENT_OUT, 0, (char*)&tmp, len);
      veo_args_set_i64(a.args, 1, len);

      Status s = call_and_wait(sym_get_timestamp_, a);
      if (s.ok()) {
        *ts = tmp.ts;
        *resolution = tmp.resolution;
      }

      return s;
    }

    virtual uint64_t alloc_mem(size_t size) {
#if 0
      VLOG(2) << "VEO::alloc_mem: tid=" << syscall(SYS_gettid);
#endif
      VLOG(2) << "VEO::alloc_mem: proc_=" << proc_ << " size=" << size;
      uint64_t addr;

      int ret;
      {
        ret = veo_alloc_mem(proc_, &addr, size);
      }

      VLOG(2) << "VEO::alloc_mem: ret=" << ret << " addr=" << std::hex << addr;
      return addr;
    }

    virtual int free_mem(uint64_t addr) {
      return veo_free_mem(proc_, addr);
    }

    virtual Status write_mem(uint64_t ve_addr, const void* vh_buff, size_t len) {
      int rc;
      uint64_t start = 0, end = 0; // initialize to suppress warning
      //uint64_t start0 = Env::Default()->NowMicros();

      //bool useDMA = false;
#ifdef USE_DMA
      if (dma_.available && len <= dma_.bufsize && len >= dma_.threshold) {
        mutex_lock guard(dma_.lock);
        //VLOG(2) << "VEO::write_mem: after dma lock";
        if (isTracerEnabled())
          start = Env::Default()->NowMicros();
        //start0 = Env::Default()->NowMicros(); useDMA = true;

        assert(len < 256 * 1024 * 1024);
        struct {
          uint64_t size;
          uint64_t vevma;
        } tmp;
        tmp.size = len;
        tmp.vevma = ve_addr;
        Args a(&tmp, sizeof(tmp));

        memcpy(dma_.shmptr, vh_buff, len);
        Status s = call_and_wait(dma_.sym_dma_read, a);
        if (!s.ok())
          return s;
        rc = 0;
      } else {
        if (isTracerEnabled())
          start = Env::Default()->NowMicros();
        rc = veo_write_mem(proc_, ve_addr, vh_buff, len);
      }

      if (isTracerEnabled()) {
        end = Env::Default()->NowMicros();
        callbackTracer(start, end, 0); // 0: HtoD
      }
#else
      if (isTracerEnabled()) {
        uint64_t start = Env::Default()->NowMicros();
        rc = veo_write_mem(proc_, ve_addr, vh_buff, len);
        uint64_t end = Env::Default()->NowMicros();
        callbackTracer(start, end, 0); // 0: HtoD
      } else
        rc = veo_write_mem(proc_, ve_addr, vh_buff, len);
#endif

      //uint64_t end0 = Env::Default()->NowMicros();
      //VLOG(0) << "VEO::write_mem: len=" << len << " time(us)=" << (end0 - start0) << " useDMA=" << useDMA;

      if (rc == 0)
        return Status::OK();
      else
        return errors::Internal("write_mem failed. rc=", rc);
    }

    virtual Status read_mem(void* vh_buff, uint64_t ve_addr, size_t len) {
      int rc;
      if (isTracerEnabled()) {
        uint64_t start = Env::Default()->NowMicros();
        rc = veo_read_mem(proc_, vh_buff, ve_addr, len);
        uint64_t end = Env::Default()->NowMicros();
        callbackTracer(start, end, 1); // 1: DtoH
      } else
        rc = veo_read_mem(proc_, vh_buff, ve_addr, len);

      if (rc == 0)
        return Status::OK();
      else
        return errors::Internal("write_mem failed. rc=", rc);
    }

    virtual Status compute(const std::string& name, const void* arg, size_t len,
                           const OpKernel* op) {
      VLOG(2) << "VEO::compute: name=" << name << " arg=" << arg << " len=" << len;
      uint64_t sym = find_kernel_sym(name);

      Args a;
      veo_args_set_stack(a.args, VEO_INTENT_IN, 0, (char*)arg, len);
      veo_args_set_i64(a.args, 1, len);
      return call_and_wait(sym, a);
    }

    virtual Status init(int nodeid);
    virtual Status sync() { return Status::OK(); }

  protected:
    uint64_t find_kernel_sym(std::string const& name) {
      auto it = kernel_map_.find(name);
      if (it == kernel_map_.end())
        return 0;
      return it->second;
    }

    std::string find_kernel_name(uint64_t sym) {
      for (auto p : kernel_map_) {
        if (p.second == sym)
          return p.first;
      }
      return "(unknown)";
    }

    virtual uint64_t get_sym(uint64_t lib_id, const char* name) {
      return veo_get_sym(proc_, lib_id, name);
    }

    virtual uint64_t call(uint64_t sym, const Args& a) {
      uint64_t req_id = veo_call_async(ctx_, sym, a.args);
      VLOG(2) << "VEO::call: return from veo_call_async. req_id=" << req_id;
      return req_id;
    }

    virtual Status wait(uint64_t req_id, uint64_t *pRetval = NULL) {
      VLOG(2) << "VEO::wait: call veo_wait_result for req_id=" << req_id;
      uint64_t retval;
      int ret = veo_call_wait_result(ctx_, req_id, &retval);
      VLOG(2) << "VEO::wait: return from veo_wait_result."
        << " req_id=" << req_id << " ret=" << ret << " retval=" << retval;
      if (pRetval)
        *pRetval = retval;
      if (ret != 0)
        return errors::Internal("Failed to wait kernel result");
      if (retval != 0)
        return errors::Internal("Failed in the kernel: retval=", retval);

      return Status::OK();
    }

    virtual Status call_and_wait(uint64_t sym, const Args& a, 
                                 uint64_t *pRetval = NULL) {
      uint64_t req_id = call(sym, a);
      if (req_id == VEO_REQUEST_ID_INVALID)
        return errors::Internal("Failed to call kernel");
      return wait(req_id, pRetval);
    }

    void callbackTracer(const std::vector<std::string>& kernel_names,
                        const void* buf)
    {
      VLOG(2) << "VEO::callbackTracer: cb_=" << reinterpret_cast<void*>(cb_);
      if (cb_) {
        struct {
          const std::vector<std::string>* kernel_names;
          const void* buf;
        } tmp;
        tmp.kernel_names = &kernel_names;
        tmp.buf = buf;
        cb_(0, 0, &tmp, cb_data_);
      }
    }

    void callbackTracer(uint64_t start, uint64_t end, int type) {
      if (cb_) {
        struct {
          uint64_t start;
          uint64_t end;
          uint64_t type;
        } tmp;
        tmp.start = start;
        tmp.end = end;
        tmp.type = type;
        cb_(0, 1, &tmp, cb_data_);
      }
    }

  private:
#if 0
    pid_t proc_pid_;
#endif
    struct veo_proc_handle* proc_;
    struct veo_thr_ctxt *ctx_;

    std::map<std::string, uint64_t> kernel_map_;
    uint64_t sym_get_timestamp_;
    cb_t cb_;
    void* cb_data_;

#ifdef USE_DMA
    struct {
      bool available;
      size_t bufsize;
      uint64_t threshold;
      mutex lock;
      uint64_t sym_dma_read;
      void* shmptr;
    } dma_;

    Status init_dma(veo_proc_handle* proc, uint64_t lib_id);
#endif
};

class KernelStack
{
  public:
    KernelStack(size_t size) : buf_size_(size), num_kernels_(0) {
      buf_ = reinterpret_cast<void*>(new char[buf_size_]);
      curr_ = reinterpret_cast<uintptr_t>(buf_)
        + sizeof(int32_t); // reserve int32_t to store num_kernels
      top_ = curr_ + buf_size_;
    }

    int push(uint64_t sym, const void* arg, size_t len, 
             const std::string* annotation) {
#if 0
      VLOG(2) << "KernelStack::push: num_kernels=" << num_kernels_
        << " len=" << len << " size=" << size();
#endif

      size_t sz = sizeof(uint64_t) + sizeof(size_t) + len;
      if (curr_ + sz >= top_) {
        VLOG(2) << "KernelStack::push: overflow";
        return 1;
      }

      ++num_kernels_;

      // copy to buf
      *reinterpret_cast<uint64_t*>(curr_) = sym;
      curr_ += sizeof(uint64_t);
      *reinterpret_cast<size_t*>(curr_) = len;
      curr_ += sizeof(size_t);
      memcpy(reinterpret_cast<void*>(curr_), arg, len);
      curr_ += len;

      if (annotation)
        annotations_.push_back(*annotation);

      return 0;
    }

    uint64_t find_sym(int idx) const {
      //VLOG(2) << "KernelStack::find_sym: idx=" << idx << " num_kernels_=" << num_kernels_;
      if (num_kernels_ <= idx)
        return 0;
      uintptr_t curr = reinterpret_cast<uintptr_t>(buf_);
      curr += sizeof(int32_t); // skip num_kernels
      uint64_t sym = 0;
      for (int i = 0; i < idx + 1; ++i) {
        sym = *reinterpret_cast<const uint64_t*>(curr);
        size_t len = *reinterpret_cast<const size_t*>(curr + sizeof(uint64_t));
        curr += len + sizeof(uint64_t) + sizeof(size_t);
        //VLOG(2) << "KernelStack::find_sym: i=" << i << " sym=" << sym;
      }
      return sym;
    }

    int32_t num_kernels() const { return num_kernels_; }
    void* buf() { return buf_; }
    size_t size() const { return curr_ - reinterpret_cast<uintptr_t>(buf_); }
    void clear() {
      curr_ = reinterpret_cast<uintptr_t>(buf_) + sizeof(int32_t);
      num_kernels_ = 0;
      annotations_.clear();
    }
    const std::vector<std::string>& annotations() const { return annotations_; }

  private:
    size_t buf_size_;
    int32_t num_kernels_;
    void* buf_;
    uintptr_t top_;
    uintptr_t curr_;
    std::vector<std::string> annotations_;
};

#ifdef VEO_ASYNC
class VEOAsync : public VEO
{
  public:
    VEOAsync()  {
      stack_size_ = 10 * 1024 * 1024;
      int stack_pool_size = 10;
        for (int i = 0; i < stack_pool_size - 1; ++i) {
          stack_pool_.push_back(new KernelStack(stack_size_));
        }
        currStack_ = new KernelStack(stack_size_);
      }

#ifdef TF_VE_EXECUTOR
    ~VEOAsync() {
      thread_done_ = true;
    }
#endif

    Status init(int nodeid) override {
      Status s = VEO::init(nodeid);
      if (!s.ok())
        return s;
      sym_prof_ = get_sym(0, "vetfkl_entry_prof");
      sym_noprof_ = get_sym(0, "vetfkl_entry");

      VLOG(2) << "VEOAsync: sym_prof=" << std::hex << sym_prof_;
      VLOG(2) << "VEOAsync: sym_noprof=" << std::hex << sym_noprof_;
      if (sym_prof_ == 0 || sym_noprof_ == 0)
        return errors::Internal("Failed to get symbol for vetfkl_entry");

#ifdef TF_VE_EXECUTOR
      if (char const* tmp = getenv("TF_VE_EXECUTOR")) {
          ve_executor_enabled_ = true;
          ve_executor_threshold_ = std::atoi(tmp);
          VLOG(2) << "VEOAsync: StartThread: "
            << " threshold=" << ve_executor_threshold_;
          thread_.reset(tensorflow::Env::Default()->StartThread(
            tensorflow::ThreadOptions(), "ve_sync_thread",
            std::bind(&VEOAsync::Run, this)));
      }
#endif
      return Status::OK();
    }

    virtual Status write_mem(uint64_t ve_addr, const void* vh_buff, size_t len) override {
      //VLOG(2) << "VEOAsync::write_mem";
      Status s = sync();
      if (!s.ok())
        return s;
      return VEO::write_mem(ve_addr, vh_buff, len);
    }

    virtual Status read_mem(void* vh_buff, uint64_t ve_addr, size_t len) override {
      Status s = sync();
      if (!s.ok())
        return s;
      return VEO::read_mem(vh_buff, ve_addr, len);
    }

    virtual Status compute(const std::string& name, const void* arg, size_t len,
                           const OpKernel* op) override {
      mutex_lock guard(lock_stack_);

      VLOG(2) << "VEOAsync::compute:"
        << " name=" << name
        << " num_kernels_in_stack=" << currStack_->num_kernels();
      uint64_t sym = find_kernel_sym(name);
      if (sym == 0)
        return errors::Internal("VEOAsync: VE kernel not found for ", name);

      int ret;
      if (isTracerEnabled()) {
        std::string annotation;
        if (op)
          annotation = strings::StrCat(op->name(), ":", op->type_string());
        else
          annotation = name;
        ret = currStack_->push(sym, arg, len, &annotation); 
      } else {
        ret = currStack_->push(sym, arg, len, nullptr); 
      }

      if (ret != 0)
        return errors::Internal("VEOAsync: Failed to push kernel");

#ifdef TF_VE_EXECUTOR
      if (ve_executor_enabled_
          && currStack_->num_kernels() >= ve_executor_threshold_) {
        VLOG(2) << "VEOAsync::compute: notify executor";
        executor_cond_.notify_all();
      }
#endif

      return Status::OK();
    }

    virtual Status sync() override {
      // Only one thread can sync at once.
      mutex_lock guard_sync(lock_sync_);

      if (currStack_->num_kernels() == 0)
        return Status::OK();

      KernelStack* stack = currStack_;
      KernelStack* nextStack;
      if (stack_pool_.size() > 0) {
          nextStack = stack_pool_.back();
          stack_pool_.pop_back();
      } else {
        nextStack = new KernelStack(stack_size_);
      }

      {
        mutex_lock guard_stack(lock_stack_);
        currStack_ = nextStack;
      }

      // here, curren thread is only one holder of the stack

      // Set n again because new kernel might be pushed to the stack.
      int32_t n = stack->num_kernels();
      VLOG(2) << "VEOAsync::sync: num_kernels=" << n;

      size_t len = stack->size();
      void* buf = stack->buf();
      *reinterpret_cast<int32_t*>(buf) = n;
      Status s;

      uint64_t retval = 0;
      if (isTracerEnabled()) {
        size_t len_out = sizeof(double) + sizeof(uint64_t) * n * 2;
        std::vector<char> buf_out(len_out);
        Args args(buf, len, buf_out.data(), len_out);
        s = call_and_wait(sym_prof_, args, &retval);

        if (s.ok())
          callbackTracer(stack->annotations(), buf_out.data());
      } else {
        Args args(buf, len);
        s = call_and_wait(sym_noprof_, args, &retval);
      }

      if (!s.ok()) {
        int i = retval >> 32;
        int rc = retval & 0xffffffff;
        uint64_t sym = stack->find_sym(i);
        std::string name = find_kernel_name(sym);
        VLOG(2) << "VEOAsync::sync: retval=" << retval
          << " i=" << i
          << " rc=" << rc
          << " sym=" << reinterpret_cast<void*>(sym)
          << " name=" << name;
        return errors::Internal("Failed in ", name, " Kernel on VE. rc=", rc);
      }

      stack->clear();
      stack_pool_.push_back(stack);

      VLOG(2) << "VEOAsync::sync: done num_kernels=" << n;
      return s;
    }

  private:
    mutex lock_stack_;
    mutex lock_sync_;

    std::vector<KernelStack*> stack_pool_;
    KernelStack* currStack_;
    size_t stack_size_;

    uint64_t sym_prof_;
    uint64_t sym_noprof_;

#ifdef TF_VE_EXECUTOR
    bool ve_executor_enabled_ = false;
    std::unique_ptr<Thread> thread_;
    int ve_executor_threshold_ = 1;
    bool thread_done_ = false;
    condition_variable executor_cond_;
    mutex executor_mutex_;

    Status Run() {
      VLOG(2) << "VEExecturo: begin";
      while (!thread_done_) {
        if (currStack_->num_kernels() >= ve_executor_threshold_) {
          VLOG(2) << "VEExecutor: sync";
          sync();
        } else {
          VLOG(2) << "VEExecutor: sleep";
          mutex_lock l(executor_mutex_);
          executor_cond_.wait(l);
          VLOG(2) << "VEExecutor: woke";
        }
      }
      return Status::OK();
    }
#endif
};
#endif

Status veo_sym_call(struct veo_proc_handle* proc,
                    struct veo_thr_ctxt* ctx,
                    uint64_t lib_id,
                    const char* name,
                    uint64_t* retval)
{
  uint64_t sym = veo_get_sym(proc, lib_id, name);
  if (!sym)
    return errors::Internal("Failed to get symbol for ", name);

  VEO::Args args;

  if (!args.args)
    return errors::Internal("Failed to allocate arguments");

  uint64_t req_id = veo_call_async(ctx, sym, args.args);
  //VLOG(2) << "VEO::load_kernel_syms: VEO request ID = " << req_id;
  if (req_id == VEO_REQUEST_ID_INVALID) {
    return errors::Internal("Failed to call VE");
  }

  int ret = veo_call_wait_result(ctx, req_id, retval);
  if (ret != 0) {
    return errors::Internal("Failed to call wait result");
  }

  return Status::OK();
}

Status load_kernel_syms(struct veo_proc_handle* proc,
                        struct veo_thr_ctxt* ctx,
                        uint64_t lib_id,
                        std::map<std::string, uint64_t>& map)
{
  Status s;

  uint64_t num_kernels;
  s = veo_sym_call(proc, ctx, lib_id, "get_num_kernels", &num_kernels);
  if (!s.ok())
    return s;
  VLOG(2) << "VEO::load_kernel_syms: num_kernels=" << num_kernels;

  uint64_t addr;
  s = veo_sym_call(proc, ctx, lib_id, "get_kernel_table_addr", &addr);
  if (!s.ok())
    return s;

  struct kernel {
    char name[256];
    char func[256];
  } table[num_kernels];

  int ret = veo_read_mem(proc, table, addr, num_kernels * sizeof(kernel));
  if (ret != 0)
    return errors::Internal("Failed to read mem");

  for (uint64_t i = 0; i < num_kernels; ++i) {
    uint64_t sym = veo_get_sym(proc, lib_id, table[i].func);
    VLOG(2) << "VEO::load_kernel_syms:"
      << " name=" << table[i].name
      << " func=" << table[i].func
      << " sym=" << (void*)sym;
    if (!sym)
      return errors::Internal("Failed to get symbol for ", table[i].func);
    map[table[i].name] = sym;
  }

  return Status::OK();
}

#ifdef USE_DMA
Status VEO::init_dma(veo_proc_handle* proc, uint64_t lib_id)
{
  size_t size = 256 * 1024 * 1024;
  if (const char* tmp = getenv("TF_DMA_BUF_SIZE")) {
    size = atoi(tmp);
  }
  dma_.bufsize = size;

  VLOG(2) << "VEO::init: buffer size for DMA is " << size
    << " bytes. Can be changed by TF_DMA_BUF_SIZE";

  if (size == 0) {
    LOG(WARNING) << "VE: DMA is disabled because TF_DMA_BUF_SIZE is 0 byte";
    dma_.available = false;
    return Status::OK();
  }

  // VE DMA uses hugetable.
  // To enable hugetable, `echo 1024 > /proc/sys/vm/nr_hugepages`
  int shmid = shmget(IPC_PRIVATE, size, SHM_HUGETLB | IPC_CREAT | IPC_EXCL | 0600);
  VLOG(2) << "VEO::init: shmid=" << shmid;
  if (shmid == -1) {
    // When hugetable is not available, DMA is disabled.
    LOG(WARNING) << "VE: DMA is disable because shmget with SHM_HUGETLB was failed";
    dma_.available = false;
    return Status::OK();
  }
  dma_.available = true;

  dma_.shmptr = shmat(shmid, NULL, 0);
  if (dma_.shmptr == (void*)-1) {
    return errors::Internal("shmat failed");
  }

  if (shmctl(shmid, IPC_RMID, NULL) != 0) {
    return errors::Internal("shmctl failed");
  }

  dma_.threshold = 256 * 1024;
  if (const char* tmp = getenv("TF_DMA_THRESHOLD")) {
    dma_.threshold = atoi(tmp);
  }

  VLOG(2) << "VEO::init: dma_threshold is " << dma_.threshold 
    << " bytes. Can be changed by TF_DMA_THRESHOLD"; 

  dma_.sym_dma_read = veo_get_sym(proc, lib_id, "vetfkl_dma_read");

  // call init_dma on VE
  uint64_t sym_dma_init = veo_get_sym(proc, lib_id, "vetfkl_dma_init");

  VLOG(2) << "VEO:init:: sym_dma_read=" << dma_.sym_dma_read;
  VLOG(2) << "VEO:init:: sym_dma_init=" << sym_dma_init;

  struct {
    int32_t shmid;
    uint64_t size;
  } tmp;

  tmp.shmid = shmid;
  tmp.size = size;

  Args a(&tmp, sizeof(tmp));
  return call_and_wait(sym_dma_init, a);
}
#endif // USE_DMA

Status VEO::init(int nodeid) {
#if 0
  VLOG(2) << "VEO::init: pid=" << getpid() << " tid=" << syscall(SYS_gettid);
#endif
  const char* filename = NULL ;

  if (const char* tmp = getenv("VEO_KERNEL")) {
    filename = tmp;
  }
  VLOG(2) << "VEO::init: filename=" << filename;
  VLOG(2) << "VEO::init: nodeid=" << nodeid;

  proc_ = veo_proc_create(nodeid);
  VLOG(2) << "VEO::init: proc_=" << proc_;
  if (!proc_)
    return errors::Internal("Failed to create VEO proc");

#if 0
  proc_pid_ = getpid();
  VLOG(2) << "VEO::init: pid=" << proc_pid_ << " tid=" << syscall(SYS_gettid);
#endif

  uint64_t lib_id = 0UL;
  if( filename != NULL ) {
    lib_id = veo_load_library(proc_, filename);
    VLOG(2) << "VEO::init: lib_id=" << lib_id;
    if (!lib_id)
      return errors::Internal("Failed to load library: ", filename);
  }

  ctx_ = veo_context_open(proc_);
  VLOG(2) << "VEO::init: ctx_=" << ctx_;
  if (!ctx_)
    return errors::Internal("Failed to open VEO context");

  sym_get_timestamp_ = veo_get_sym(proc_, lib_id, "vetfkl_get_timestamp");
  if (sym_get_timestamp_ == 0)
    return errors::Internal("Failed to veo_get_sym for vetfkl_get_timestamp");

#ifdef USE_DMA
  {
    Status s = init_dma(proc_, lib_id);
    if (!s.ok())
      return s;
  }
#endif

  return load_kernel_syms(proc_, ctx_, lib_id, kernel_map_);
}

VEO::~VEO() {
  VLOG(2) << "VEO::~VEO";
  veo_context_close(ctx_);
  veo_proc_destroy(proc_);
#ifdef USE_DMA
  shmdt(dma_.shmptr);
#endif
}

class VEMemAllocator : public SubAllocator {
  public:
    VEMemAllocator(VEO* veo) : SubAllocator({}, {}), veo_(veo) {}
    ~VEMemAllocator();
    void* Alloc(size_t alignments, size_t num_bytes);
    void Free(void* ptr, size_t num_bytes);

  private:
    VEO* veo_;
};

VEMemAllocator::~VEMemAllocator() {}

void* VEMemAllocator::Alloc(size_t alignments, size_t num_bytes) {
  VLOG(2) << "VEMemAllocator::Alloc: alignments=" << alignments << " num_bytes=" << num_bytes;
#if 0
  return veo_->alloc_mem(num_bytes);
#else
#if 0
  void* p = veo_->alloc_mem(num_bytes);
  if (reinterpret_cast<uintptr_t>(p) % alignments != 0) {
    VLOG(2) << "VEMemAllocator::Alloc: alignment error. addr=" << p;
    return NULL;
  }
#endif

  size_t n = num_bytes + alignments + sizeof(uintptr_t);
  uint64_t addr = veo_->alloc_mem(n);
  uint64_t addr0 = (addr + sizeof(uint64_t) + alignments) & ~(alignments - 1);
  VLOG(2) << "VEMemAllocator::Alloc addr=" << std::hex << addr
    << " addr0=" << std::hex << addr0;
  *reinterpret_cast<uint64_t*>(addr0 - sizeof(uint64_t)) = addr;
  return reinterpret_cast<void*>(addr0);
#endif
}

void VEMemAllocator::Free(void* ptr, size_t num_bytes) {
  VLOG(2) << "VEMemAllocator::Free: ptr=" << ptr;

  uint64_t addr = reinterpret_cast<uintptr_t>(ptr) - sizeof(uint64_t);

  VLOG(2) << "VEMemAllocator::Free: addr=" << std::hex << addr;

  veo_->free_mem(addr);
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

class VEDeviceContextImpl : public VEDeviceContext {
  public:
    VEDeviceContextImpl(VEO* veo) : veo_(veo) {}

    virtual void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                                       Tensor* device_tensor, StatusCallback done,
                                       bool sync_dst_compute = true) const override;

    virtual void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                                       Device* device, Tensor* cpu_tensor,
                                       StatusCallback done) override;

    virtual void CopyTensorInSameDevice(const Tensor* input_tensor,
					Device* device,
					Tensor* output_tensor,
					StatusCallback done) const override;

    virtual Status Compute(const std::string& name, const void* arg, size_t len,
                           const OpKernel* op);

  private:
    VEO* veo_;
};

class VEDevice : public LocalDevice {
  public:
    VEDevice(const SessionOptions& options, const string name,
             Allocator* ve_allocator,
             Allocator* cpu_allocator) :
      LocalDevice(options,
                  Device::BuildDeviceAttributes(name, "VE",
                                                Bytes(256 << 20),
                                                DeviceLocality())),
      ve_allocator_(ve_allocator),
      cpu_allocator_(cpu_allocator) {}

    ~VEDevice() override;

    Status Init(const SessionOptions& options, VEO* veo);
    Status Sync() override;

    Allocator* GetAllocator(AllocatorAttributes attr) override {
      if (attr.on_host())
        return cpu_allocator_;
      else
        return ve_allocator_;
    }

    Status MakeTensorFromProto(const TensorProto& tensor_proto,
                               const AllocatorAttributes alloc_attrs,
                               Tensor* tensor) override;

  protected:
    Allocator* ve_allocator_;
    Allocator* cpu_allocator_;

  private:
    GpuDeviceInfo* gpu_device_info_;
    std::vector<VEDeviceContextImpl*> device_contexts_;
};

VEDevice::~VEDevice() {
  delete gpu_device_info_;
  for (auto ctx : device_contexts_) ctx->Unref();
}

Status VEDevice::Init(const SessionOptions& options, VEO* veo) {
  VLOG(2) << "VEDevice::Init";
  device_contexts_.push_back(new VEDeviceContextImpl(veo));

  VLOG(2) << "VEDevice::Init DeviceContext=" << device_contexts_.back();

  gpu_device_info_ = new GpuDeviceInfo;
  gpu_device_info_->default_context = device_contexts_[0];
  set_tensorflow_gpu_device_info(gpu_device_info_);

  return Status::OK();
}

Status VEDevice::MakeTensorFromProto(const TensorProto& tensor_proto,
                                     const AllocatorAttributes alloc_attrs,
                                     Tensor* tensor) {
  VLOG(2) << "VEDeviceContextImpl::MakeTensorFromProto";
  AllocatorAttributes attr;
  attr.set_on_host(true);
  Allocator* host_alloc = GetAllocator(attr);

  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(host_alloc, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }
  Status status;
  if (alloc_attrs.on_host()) {
    *tensor = parsed;
  } else {
    Tensor copy(GetAllocator(alloc_attrs), parsed.dtype(), parsed.shape());

    // If the tensor is not initialized, we likely ran out of memory.
    if (!copy.IsInitialized()) {
      return errors::ResourceExhausted(
          "OOM when allocating tensor of shape ", parsed.shape().DebugString(),
          " and type ", DataTypeString(parsed.dtype()));
    }

    device_contexts_[0]->CopyCPUTensorToDevice(
        &parsed, this, &copy, [&status](const Status& s) { status = s; });
    *tensor = copy;
  }
  return status;
}

class VEOFactory {
  public:
    Status GetOrCreate(VEO** pveo, int nodeid) {
      mutex_lock guard(lock_);

      if (!veo_) {
#if defined(VEO_ASYNC)
        if (getenv("TF_VE_SYNC")) {
          VLOG(2) << "use VEO (not VEOAsync) because TF_VEO_SYNC is set";
          veo_ = new VEO;
        } else {
          veo_ = new VEOAsync;
        }
#else
        veo_ = new VEO;
#endif
        Status s = veo_->init(nodeid);
        if (!s.ok())
          return s;
      }

      *pveo = veo_;
      return Status::OK();
    }

    static VEOFactory* Global() {
      static VEOFactory* instance = new VEOFactory;
      return instance;
    }

  private:
    mutex lock_;
    VEO* veo_ = NULL;

    VEOFactory() = default;
    TF_DISALLOW_COPY_AND_ASSIGN(VEOFactory);
};

Status VEDevice::Sync() {
  VLOG(2) << "VEDevice::Sync";
  VEO* veo = NULL;
  int nodeid = 0; // FIXME
  Status s = VEOFactory::Global()->GetOrCreate(&veo, nodeid);
  if (!s.ok())
    return s;

  return veo->sync();
}

class VEDeviceFactory : public DeviceFactory {
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override {
    const string device_name = strings::StrCat(name_prefix, "/device:VE:0");
    VLOG(2) << "VEDeviceFactory::CreateDevices: " << device_name;

    int nodeid = 0;
    if (const char* tmp = getenv("VE_NODE_NUMBER")) {
      nodeid = atoi(tmp);
    }

    if (nodeid < 0) // user disables VE
      return Status::OK();

    VEO* veo = NULL;
    Status s = VEOFactory::Global()->GetOrCreate(&veo, nodeid);
    if (!s.ok())
      return s;

    size_t total_memory = 20UL*1024*1024*1024;
    Allocator* ve_allocator = new VEBFCAllocator(total_memory, true, "VE_0_bfc", veo);

    int numa_node = 0;

    std::unique_ptr<VEDevice> device 
      = absl::make_unique<VEDevice>(options, device_name, ve_allocator,
                                    ProcessState::singleton()->GetCPUAllocator(numa_node));
    TF_RETURN_IF_ERROR(device->Init(options, veo));
    devices->push_back(std::move(device));
    return Status::OK();
  }

  Status ListPhysicalDevices(std::vector<string>* devices) {
    int nodeid = 0;
    if (const char* tmp = getenv("VE_NODE_NUMBER")) {
      nodeid = atoi(tmp);
    }
    if (nodeid > 0) {
      const string device_name = strings::StrCat("/physical_device:VE:", nodeid);
      devices->push_back(device_name);
    }
    return Status::OK();
  }
};

} // namespace

REGISTER_LOCAL_DEVICE_FACTORY("VE", VEDeviceFactory, 220);

void VEDeviceContextImpl::CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                                                Tensor* device_tensor,
                                                StatusCallback done,
                                                bool sync_dst_compute) const {
  VLOG(2) << "VEDeviceContextImpl::CopyCPUTensorToDevice";

  const void* in = DMAHelper::base(cpu_tensor);
  void* out = DMAHelper::base(device_tensor);

  VLOG(2) << "VEDeviceContextImpl::CopyCPUTensorToDevice: in=" << in
    << " out=" << out << " size=" << cpu_tensor->TotalBytes();
#if 0
  VLOG(2) << "VEDeviceContextImpl::CopyCPUTensorToDevice: proc_pid_=" << getpid() << " tid=" << syscall(SYS_gettid);
#endif

  Status s = veo_->write_mem((uint64_t)out, in, cpu_tensor->TotalBytes());
  done(s);
  VLOG(2) << "VEDeviceContextImpl::CopyCPUTensorToDevice: done";
}

void VEDeviceContextImpl::CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                                                Device* device, Tensor* cpu_tensor,
                                                StatusCallback done) {
  VLOG(2) << "VEDeviceContextImpl::CopyDeviceTensorToCPU";

  const void* in = DMAHelper::base(device_tensor);
  void* out = DMAHelper::base(cpu_tensor);

  VLOG(2) << "VEDeviceContextImpl::CopyDeviceTensorToCPU: in=" << in
    << " out=" << out << " size=" << device_tensor->TotalBytes();

  Status s = veo_->read_mem(out, (uint64_t)in, device_tensor->TotalBytes());
  done(s);
}

void VEDeviceContextImpl::CopyTensorInSameDevice(const Tensor* input_tensor,
                                                 Device* device,
                                                 Tensor* output_tensor,
                                                 StatusCallback done) const {
  VLOG(2) << "VEDeviceContextImpl::CopyTensorInSameDevice";

  struct {
    uint64_t dst, src ;
    size_t size ;
  } args ;

  args.dst = (uint64_t) DMAHelper::base(output_tensor) ;
  args.src = (uint64_t) DMAHelper::base(input_tensor) ;

  args.size = input_tensor->TotalBytes() ;

  Status s = veo_->compute("Snapshot", (void*)&args, sizeof(args), nullptr);
  done(s) ;
}

Status VEDeviceContextImpl::Compute(const std::string& name, const void* arg, size_t len,
                                    const OpKernel* op)
{
  VLOG(2) << "VEDeviceContextImpl::Compute: name=" << name;
  return veo_->compute(name, arg, len, op);
}

Status ve_get_timestamp(int nodeid, uint64_t* ts, double* resolution)
{
  VEO* veo = NULL;
  Status s = VEOFactory::Global()->GetOrCreate(&veo, nodeid);
  if (!s.ok())
    return s;

  return veo->get_timestamp(ts, resolution);
}

Status ve_set_trace_callback(int nodeid, VEO::cb_t cb, void* data)
{
  VLOG(2) << __FUNCTION__ << ": cb=" << reinterpret_cast<void*>(cb)
    << " data=" << data;
  VEO* veo = NULL;
  Status s = VEOFactory::Global()->GetOrCreate(&veo, nodeid);
  if (s.ok())
    veo->setTraceCallback(cb, data);

  return s;
}

} // namespace tensorflow


// runtime/runtime.hpp - runtime interfaces
//
// The runtime is responsible for:
//   - selecting a device
//   - allocating memory
//   - launching compiled kernels
//   - managing streams / events
//   - caching compiled kernels by (graph hash, shape, dtype, layout, hardware)
//   - dispatching specialized kernels with a fallback path
//
// The optimizer NEVER calls into the runtime. The runtime is the user-facing
// entry point for executing a compiled module.
#pragma once

#include "cg/backend/backend.hpp"
#include "cg/codegen/codegen_ir.hpp"
#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"
#include "cg/schedule/schedule.hpp"

#include <any>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cg {

// A Tensor is a runtime-owned buffer with shape, dtype, layout, device.
class Tensor {
public:
    Shape shape;
    DType dtype = DType::F32;
    LayoutPtr layout;
    DeviceId device;
    void* data = nullptr;
    u64   num_bytes = 0;
    bool  owns_memory = false;
};

class Allocator {
public:
    virtual ~Allocator() = default;
    virtual void* allocate(u64 bytes, u64 alignment = 256) = 0;
    virtual void  deallocate(void* p) = 0;
};

class Stream {
public:
    virtual ~Stream() = default;
    virtual void sync() = 0;
    virtual DeviceId device() const = 0;
};

class Event {
public:
    virtual ~Event() = default;
    virtual void record(Stream& s) = 0;
    virtual void wait() = 0;
    virtual double elapsed_sec_since(Event& earlier) = 0;
};

class Device {
public:
    virtual ~Device() = default;
    virtual DeviceId id() const = 0;
    virtual Allocator& allocator() = 0;
    virtual std::unique_ptr<Stream> create_stream() = 0;
    virtual std::unique_ptr<Event>  create_event()  = 0;
};

// A compiled executable bound to a specific device. Launching it is a
// matter of passing input tensors + an output buffer + a stream.
class BoundExecutable {
public:
    virtual ~BoundExecutable() = default;
    virtual void launch(Span<Tensor*> inputs,
                        Span<Tensor*> outputs,
                        Stream& stream) = 0;
};

// Persistent kernel cache. Keyed by (graph hash, shape, dtype, layout,
// hardware fingerprint). Stored on disk under a configured cache dir.
class KernelCache {
public:
    explicit KernelCache(std::string dir) : dir_(std::move(dir)) {}

    std::optional<std::shared_ptr<Executable>>
    lookup(u64 key);

    void insert(u64 key, std::shared_ptr<Executable> exe);

    // Compute a stable hash for (graph_text, shapes, dtypes, hardware_name).
    static u64 compute_key(const std::string& graph_text,
                           const std::string& shapes,
                           const std::string& dtypes,
                           const std::string& hardware_name);

    const std::string& dir() const { return dir_; }
    usize size() const { return mem_cache_.size(); }

private:
    std::string dir_;
    std::unordered_map<u64, std::shared_ptr<Executable>> mem_cache_;
};

// Runtime context: owns devices and a kernel cache.
class Runtime {
public:
    Runtime();
    ~Runtime();

    // Register a device. The runtime takes ownership.
    void add_device(std::unique_ptr<Device> d);
    Device* get_device(DeviceId id) const;
    usize num_devices() const { return devices_.size(); }

    KernelCache& kernel_cache() { return cache_; }

    // Bind an Executable to a specific device. Returns null on failure.
    std::unique_ptr<BoundExecutable>
    bind(std::shared_ptr<Executable> exe, DeviceId device);

    // Compile + cache + bind in one call. If the kernel is already in the
    // cache (keyed by `cache_key`), the cached Executable is used. Otherwise
    // `backend.compile(cgm)` is called, the result is cached, and then bound.
    // This is the main end-to-end entry point for executing a compiled module.
    struct CompileResult {
        std::shared_ptr<Executable> executable;
        bool cache_hit = false;
    };
    CompileResult compile_and_cache(u64 cache_key,
                                     const CGModule& cgm,
                                     MachineBackend& backend);

    // Autotune + cache: run the autotuner over a ScheduleSpace, compile the
    // best schedule via `backend`, cache the result under `cache_key`, and
    // return the Executable. The `benchmark` function is called by the
    // autotuner to measure each candidate schedule's runtime.
    struct AutotuneResult {
        std::shared_ptr<Executable> executable;
        double best_runtime = 0.0;
        usize total_benchmarks = 0;
        bool cache_hit = false;
    };
    AutotuneResult autotune_and_cache(
        u64 cache_key,
        const ScheduleSpace& space,
        const std::function<double(const Schedule&)>& benchmark,
        MachineBackend& backend);

private:
    std::vector<std::unique_ptr<Device>> devices_;
    KernelCache cache_;
};

} // namespace cg

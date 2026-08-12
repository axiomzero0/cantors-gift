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
#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"

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

    const std::string& dir() const { return dir_; }

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

private:
    std::vector<std::unique_ptr<Device>> devices_;
    KernelCache cache_;
};

} // namespace cg

// runtime/runtime.cpp - basic runtime implementation with autotuner integration
#include "cg/runtime/runtime.hpp"
#include "cg/autotuner/bayesian_optimizer.hpp"

#include <fstream>
#include <filesystem>
#include <sstream>

namespace cg {

namespace {

// A simple host allocator used when no device-specific allocator is provided.
class HostAllocator : public Allocator {
public:
    void* allocate(u64 bytes, u64 alignment) override {
        if (alignment <= alignof(std::max_align_t)) {
            return std::malloc(bytes);
        }
#if defined(__cpp_aligned_new) && (__cpp_aligned_new >= 201606L)
        return ::operator new(bytes, std::align_val_t{alignment});
#else
        void* p = nullptr;
        if (posix_memalign(&p, alignment, bytes) != 0) return nullptr;
        return p;
#endif
    }
    void deallocate(void* p) override {
        std::free(p);
    }
};

class HostStream : public Stream {
public:
    explicit HostStream(DeviceId d) : d_(d) {}
    void sync() override {}
    DeviceId device() const override { return d_; }
private:
    DeviceId d_;
};

class HostEvent : public Event {
public:
    void record(Stream&) override {}
    void wait() override {}
    double elapsed_sec_since(Event&) override { return 0.0; }
};

class HostDevice : public Device {
public:
    explicit HostDevice(u32 idx) : id_(DeviceId::cpu(idx)), alloc_() {}
    DeviceId id() const override { return id_; }
    Allocator& allocator() override { return alloc_; }
    std::unique_ptr<Stream> create_stream() override { return std::make_unique<HostStream>(id_); }
    std::unique_ptr<Event>  create_event()  override { return std::make_unique<HostEvent>(); }
private:
    DeviceId id_;
    HostAllocator alloc_;
};

} // namespace

Runtime::Runtime() : cache_(".cantors_cache") {}
Runtime::~Runtime() = default;

void Runtime::add_device(std::unique_ptr<Device> d) {
    devices_.push_back(std::move(d));
}

Device* Runtime::get_device(DeviceId id) const {
    for (auto& d : devices_) if (d->id() == id) return d.get();
    return nullptr;
}

std::unique_ptr<BoundExecutable>
Runtime::bind(std::shared_ptr<Executable>, DeviceId) {
    // Real implementation dispatches to a backend-specific launcher.
    // For the foundational commit we return null: the optimizer + backend
    // + runtime integration is wired up once a concrete emitter exists.
    return nullptr;
}

u64 KernelCache::compute_key(const std::string& graph_text,
                              const std::string& shapes,
                              const std::string& dtypes,
                              const std::string& hardware_name) {
    u64 h = 0xcbf29ce484222325ULL; // FNV offset basis
    auto hash_str = [&](const std::string& s) {
        for (char c : s) {
            h ^= static_cast<u64>(static_cast<u8>(c));
            h *= 0x100000001b3ULL; // FNV prime
        }
    };
    hash_str(graph_text);
    hash_str(shapes);
    hash_str(dtypes);
    hash_str(hardware_name);
    return h;
}

std::optional<std::shared_ptr<Executable>>
KernelCache::lookup(u64 key) {
    auto it = mem_cache_.find(key);
    if (it != mem_cache_.end()) return it->second;

    // Try disk.
    namespace fs = std::filesystem;
    auto path = fs::path(dir_) / (std::to_string(key) + ".bin");
    if (!fs::exists(path)) return std::nullopt;
    // Disk deserialization requires a stable binary schema; for now we only
    // use the in-memory cache. A future commit will add binary serialization.
    return std::nullopt;
}

void KernelCache::insert(u64 key, std::shared_ptr<Executable> exe) {
    mem_cache_[key] = std::move(exe);
}

Runtime::CompileResult Runtime::compile_and_cache(u64 cache_key,
                                                    const CGModule& cgm,
                                                    MachineBackend& backend) {
    CompileResult result;
    // Check cache first.
    auto cached = cache_.lookup(cache_key);
    if (cached) {
        result.executable = *cached;
        result.cache_hit = true;
        return result;
    }
    // Compile.
    result.executable = backend.compile(cgm);
    result.cache_hit = false;
    cache_.insert(cache_key, result.executable);
    return result;
}

Runtime::AutotuneResult Runtime::autotune_and_cache(
    u64 cache_key,
    const ScheduleSpace& space,
    const std::function<double(const Schedule&)>& benchmark,
    MachineBackend& backend) {
    AutotuneResult result;
    // Check cache first.
    auto cached = cache_.lookup(cache_key);
    if (cached) {
        result.executable = *cached;
        result.cache_hit = true;
        return result;
    }
    // Run the autotuner.
    auto tune_result = bayesian_autotune(space, benchmark);
    result.best_runtime = tune_result.best_runtime;
    result.total_benchmarks = tune_result.total_benchmarks;
    // The autotuner found the best schedule; the caller is responsible for
    // compiling it via `backend`. For the end-to-end path, we compile a
    // placeholder CGModule here — in a real workflow, the caller would lower
    // the Tensor IR using the best schedule before calling this method.
    // For now, we return the best schedule info and let the caller compile.
    (void)backend;
    result.cache_hit = false;
    return result;
}

} // namespace cg

// runtime/runtime.cpp - basic runtime implementation
#include "cg/runtime/runtime.hpp"

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

std::optional<std::shared_ptr<Executable>>
KernelCache::lookup(u64 key) {
    auto it = mem_cache_.find(key);
    if (it != mem_cache_.end()) return it->second;

    // Try disk.
    namespace fs = std::filesystem;
    auto path = fs::path(dir_) / (std::to_string(key) + ".bin");
    if (!fs::exists(path)) return std::nullopt;
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    // We do not attempt to deserialize the executable here; that requires
    // a stable binary schema that will be defined alongside the first
    // concrete backend.
    return std::nullopt;
}

void KernelCache::insert(u64 key, std::shared_ptr<Executable> exe) {
    mem_cache_[key] = std::move(exe);
}

} // namespace cg

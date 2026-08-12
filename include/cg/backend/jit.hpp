// backend/jit.hpp - JIT execution for x86-64 machine code
//
// Takes raw machine code bytes, makes them executable via mmap, and provides
// a callable function pointer. This is the final step: the compiler produces
// bytes, the JIT makes them runnable, and the user can call the function.
//
// On Linux: mmap with PROT_READ|PROT_WRITE|PROT_EXEC.
// On macOS: same.
// On Windows: VirtualProtect (not implemented in this foundational version).
#pragma once

#include "cg/core/util.hpp"

#include <cstddef>
#include <vector>

namespace cg {

class JITMemory {
public:
    JITMemory() = default;
    ~JITMemory();

    JITMemory(const JITMemory&) = delete;
    JITMemory& operator=(const JITMemory&) = delete;
    JITMemory(JITMemory&&) noexcept;
    JITMemory& operator=(JITMemory&&) noexcept;

    // Allocate `size` bytes of executable memory and copy `code` into it.
    // Returns a callable function pointer.
    void* allocate(const std::vector<u8>& code);

    // The function pointer (null if not allocated).
    void* entry() const { return entry_; }
    usize size() const { return size_; }

    // True if the JIT memory is valid (allocated and executable).
    bool valid() const { return entry_ != nullptr; }

private:
    void* base_ = nullptr;
    void* entry_ = nullptr;
    usize size_ = 0;
};

// A typed wrapper around JITMemory for calling the compiled function.
// The signature depends on the kernel; for a simple kernel that takes
// pointer arguments and returns void:
//   using KernelFn = void (*)(void* arg1, void* arg2, ...);
template <typename Signature>
class JITFunction {
public:
    JITFunction() = default;

    explicit JITFunction(std::vector<u8> code) {
        memory_.allocate(code);
    }

    Signature get() const {
        return reinterpret_cast<Signature>(memory_.entry());
    }

    bool valid() const { return memory_.valid(); }

    // Call the function with the given arguments.
    template <typename... Args>
    auto operator()(Args&&... args) const {
        auto fn = get();
        return fn(std::forward<Args>(args)...);
    }

private:
    JITMemory memory_;
};

} // namespace cg

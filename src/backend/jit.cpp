// backend/jit.cpp - JIT execution via mmap
#include "cg/backend/jit.hpp"

#include <cstring>
#include <stdexcept>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#else
// Windows would use VirtualProtect; not implemented in this version.
#endif

namespace cg {

JITMemory::~JITMemory() {
    if (base_) {
#ifdef __linux__
        munmap(base_, size_);
#elif defined(__APPLE__)
        munmap(base_, size_);
#endif
    }
}

JITMemory::JITMemory(JITMemory&& other) noexcept
    : base_(other.base_), entry_(other.entry_), size_(other.size_) {
    other.base_ = nullptr;
    other.entry_ = nullptr;
    other.size_ = 0;
}

JITMemory& JITMemory::operator=(JITMemory&& other) noexcept {
    if (this != &other) {
        if (base_) {
#ifdef __linux__
            munmap(base_, size_);
#elif defined(__APPLE__)
            munmap(base_, size_);
#endif
        }
        base_ = other.base_;
        entry_ = other.entry_;
        size_ = other.size_;
        other.base_ = nullptr;
        other.entry_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void* JITMemory::allocate(const std::vector<u8>& code) {
    if (code.empty()) return nullptr;

    size_ = code.size();

    // Round up to page size.
    usize page_size = 4096;
#ifdef __linux__
    page_size = static_cast<usize>(sysconf(_SC_PAGESIZE));
#elif defined(__APPLE__)
    page_size = static_cast<usize>(sysconf(_SC_PAGESIZE));
#endif
    usize rounded = ((size_ + page_size - 1) / page_size) * page_size;

#ifdef __linux__
    // Allocate executable memory.
    base_ = mmap(nullptr, rounded,
                 PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        return nullptr;
    }
#elif defined(__APPLE__)
    base_ = mmap(nullptr, rounded,
                 PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        return nullptr;
    }
#else
    // Fallback: use aligned_alloc (NOT executable; for testing only).
    base_ = std::aligned_alloc(4096, rounded);
    if (!base_) return nullptr;
#endif

    // Copy the code.
    std::memcpy(base_, code.data(), size_);
    entry_ = base_;
    return entry_;
}

} // namespace cg

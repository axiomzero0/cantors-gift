// core/util.hpp - core utility primitives for cantors-gift
//
// Small, header-only primitives shared across the compiler. Nothing in here
// pulls in <iostream> at scale; we use <cstdio> only where actually needed.
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace cg {

// ---------------------------------------------------------------------------
// Integer types
// ---------------------------------------------------------------------------
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using usize = std::size_t;
using isize = std::ptrdiff_t;

// ---------------------------------------------------------------------------
// StringRef - non-owning string view alias
// ---------------------------------------------------------------------------
using StringRef = std::string_view;

// ---------------------------------------------------------------------------
// Span - alias for std::span, kept for naming consistency
// ---------------------------------------------------------------------------
template <typename T>
using Span = std::span<T>;

template <typename T>
Span<T> make_span(T* data, usize n) {
    return Span<T>(data, n);
}

template <typename T, usize N>
Span<T> make_span(T (&arr)[N]) {
    return Span<T>(arr, N);
}

template <typename T>
Span<T> make_span(std::vector<T>& v) {
    return Span<T>(v.data(), v.size());
}

template <typename T>
Span<const T> make_span(const std::vector<T>& v) {
    return Span<const T>(v.data(), v.size());
}

// SmallVector overloads defined after SmallVector below.

// ---------------------------------------------------------------------------
// SmallVector - stack-allocated small buffer optimized vector
//
// Designed to be a drop-in subset of std::vector for the operations the
// compiler actually performs. We deliberately avoid exception machinery:
// all operations are noexcept-by-convention and use checked growth.
// ---------------------------------------------------------------------------
template <typename T, usize N = 4>
class SmallVector {
public:
    using value_type      = T;
    using size_type       = usize;
    using difference_type = isize;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;
    using iterator        = T*;
    using const_iterator  = const T*;

    SmallVector() noexcept = default;

    SmallVector(std::initializer_list<T> il) {
        reserve(il.size());
        for (const auto& v : il) emplace_back(v);
    }

    SmallVector(usize count, const T& value) {
        reserve(count);
        for (usize i = 0; i < count; ++i) emplace_back(value);
    }

    SmallVector(const SmallVector& other) {
        reserve(other.size_);
        for (usize i = 0; i < other.size_; ++i) emplace_back(other[i]);
    }

    SmallVector(SmallVector&& other) noexcept {
        steal_from(std::move(other));
    }

    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            clear();
            reserve(other.size_);
            for (usize i = 0; i < other.size_; ++i) emplace_back(other[i]);
        }
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            destroy_all();
            deallocate();
            steal_from(std::move(other));
        }
        return *this;
    }

    ~SmallVector() {
        destroy_all();
        deallocate();
    }

    void push_back(const T& v) { emplace_back(v); }
    void push_back(T&& v)      { emplace_back(std::move(v)); }

    template <typename... Args>
    reference emplace_back(Args&&... args) {
        if (size_ == capacity_) grow(capacity_ == 0 ? (N == 0 ? 2 : N) : capacity_ * 2);
        new (slot(size_)) T(std::forward<Args>(args)...);
        ++size_;
        return back();
    }

    void pop_back() {
        if (size_ == 0) return;
        --size_;
        slot(size_)->~T();
    }

    void clear() {
        destroy_all();
        size_ = 0;
    }

    void resize(usize n) {
        if (n < size_) {
            for (usize i = n; i < size_; ++i) slot(i)->~T();
            size_ = n;
        } else if (n > size_) {
            reserve(n);
            for (usize i = size_; i < n; ++i) new (slot(i)) T();
            size_ = n;
        }
    }

    void reserve(usize n) {
        if (n <= capacity_) return;
        grow(n);
    }

    void assign(usize n, const T& v) {
        clear();
        reserve(n);
        for (usize i = 0; i < n; ++i) emplace_back(v);
    }

    template <typename It>
    void assign(It first, It last) {
        clear();
        usize n = static_cast<usize>(std::distance(first, last));
        reserve(n);
        for (auto it = first; it != last; ++it) emplace_back(*it);
    }

    iterator insert(iterator pos, const T& v) {
        usize idx = static_cast<usize>(pos - data());
        if (size_ == capacity_) grow(capacity_ == 0 ? (N == 0 ? 2 : N) : capacity_ * 2);
        // shift right
        for (usize i = size_; i > idx; --i) {
            new (slot(i)) T(std::move(*slot(i - 1)));
            slot(i - 1)->~T();
        }
        new (slot(idx)) T(v);
        ++size_;
        return data() + idx;
    }

    iterator erase(iterator pos) {
        return erase(pos, pos + 1);
    }

    iterator erase(iterator first, iterator last) {
        usize begin = static_cast<usize>(first - data());
        usize end   = static_cast<usize>(last  - data());
        if (end <= begin) return first;
        // Destroy the erased range
        for (usize i = begin; i < end; ++i) slot(i)->~T();
        // Shift the tail
        usize tail = size_ - end;
        for (usize i = 0; i < tail; ++i) {
            new (slot(begin + i)) T(std::move(*slot(end + i)));
            slot(end + i)->~T();
        }
        size_ -= (end - begin);
        return data() + begin;
    }

    reference       operator[](usize i)       { return *slot(i); }
    const_reference operator[](usize i) const { return *slot(i); }

    reference       front()       { return *slot(0); }
    const_reference front() const { return *slot(0); }
    reference       back()        { return *slot(size_ - 1); }
    const_reference back()  const { return *slot(size_ - 1); }

    pointer       data()       { return slot(0); }
    const_pointer data() const { return slot(0); }

    iterator       begin()        { return data(); }
    iterator       end()          { return data() + size_; }
    const_iterator begin()  const { return data(); }
    const_iterator end()    const { return data() + size_; }

    usize    size()     const { return size_; }
    usize    capacity() const { return capacity_; }
    bool     empty()    const { return size_ == 0; }

    bool operator==(const SmallVector& o) const {
        if (size_ != o.size_) return false;
        for (usize i = 0; i < size_; ++i)
            if (!((*this)[i] == o[i])) return false;
        return true;
    }
    bool operator!=(const SmallVector& o) const { return !(*this == o); }

private:
    T* slot(usize i) {
        if (is_inline()) return reinterpret_cast<T*>(&inline_[0]) + i;
        return heap_ + i;
    }
    const T* slot(usize i) const {
        if (is_inline()) return reinterpret_cast<const T*>(&inline_[0]) + i;
        return heap_ + i;
    }

    bool is_inline() const { return capacity_ <= N; }

    void destroy_all() {
        for (usize i = 0; i < size_; ++i) slot(i)->~T();
    }

    void deallocate() {
        if (!is_inline() && heap_) {
            ::operator delete(heap_, capacity_ * sizeof(T));
            heap_ = nullptr;
        }
    }

    void grow(usize new_cap) {
        if (new_cap <= capacity_) return;
        T* new_buf = static_cast<T*>(::operator new(new_cap * sizeof(T)));
        for (usize i = 0; i < size_; ++i) {
            new (new_buf + i) T(std::move(*slot(i)));
            slot(i)->~T();
        }
        deallocate();
        heap_ = new_buf;
        capacity_ = new_cap;
    }

    void steal_from(SmallVector&& other) {
        if (other.is_inline()) {
            for (usize i = 0; i < other.size_; ++i)
                new (reinterpret_cast<T*>(&inline_[0]) + i) T(std::move(other[i]));
            size_ = other.size_;
            capacity_ = N;
            other.destroy_all();
        } else {
            heap_ = other.heap_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.heap_ = nullptr;
            other.size_ = 0;
            other.capacity_ = N;
        }
    }

    alignas(T) std::byte inline_[N * sizeof(T)];
    T*     heap_     = nullptr;
    usize  size_     = 0;
    usize  capacity_ = N;
};

template <typename T, usize N>
Span<T> make_span(SmallVector<T, N>& v) {
    return Span<T>(v.data(), v.size());
}

template <typename T, usize N>
Span<const T> make_span(const SmallVector<T, N>& v) {
    return Span<const T>(v.data(), v.size());
}

// ---------------------------------------------------------------------------
// Result<T, E> - tagged union for fallible operations
// ---------------------------------------------------------------------------
template <typename T, typename E = std::string>
class Result {
public:
    Result(T v)  : ok_(true)  { new (&storage_) T(std::move(v)); }
    Result(E e, bool) : ok_(false) { new (&storage_) E(std::move(e)); }

    static Result ok(T v)   { return Result(std::move(v)); }
    static Result err(E e)  { return Result(std::move(e), false); }

    Result(const Result& o) : ok_(o.ok_) {
        if (ok_) new (&storage_) T(o.as_ok());
        else     new (&storage_) E(o.as_err());
    }
    Result(Result&& o) noexcept : ok_(o.ok_) {
        if (ok_) new (&storage_) T(std::move(o.as_ok()));
        else     new (&storage_) E(std::move(o.as_err()));
    }

    Result& operator=(const Result&) = delete;
    Result& operator=(Result&&)      = delete;

    ~Result() {
        if (ok_) as_ok().~T();
        else     as_err().~E();
    }

    bool is_ok()  const { return ok_; }
    bool is_err() const { return !ok_; }

    T&       value()       { return as_ok(); }
    const T& value() const { return as_ok(); }
    E&       error()       { return as_err(); }
    const E& error() const { return as_err(); }

private:
    T& as_ok() { return *reinterpret_cast<T*>(&storage_); }
    E& as_err() { return *reinterpret_cast<E*>(&storage_); }
    const T& as_ok()  const { return *reinterpret_cast<const T*>(&storage_); }
    const E& as_err() const { return *reinterpret_cast<const E*>(&storage_); }

    bool ok_;
    alignas(alignof(T) > alignof(E) ? alignof(T) : alignof(E))
        std::byte storage_[sizeof(T) > sizeof(E) ? sizeof(T) : sizeof(E)];
};

// ---------------------------------------------------------------------------
// Hashing combiner
// ---------------------------------------------------------------------------
inline void hash_combine(u64& seed, u64 v) {
    // FNV-style mix
    seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

template <typename T>
u64 hash_value(const T& v) {
    return std::hash<T>{}(v);
}

// ---------------------------------------------------------------------------
// Aligned allocation
// ---------------------------------------------------------------------------
inline void* aligned_alloc(usize alignment, usize size) {
    // Use C17 aligned_alloc when available; fall back to operator new
    // with manual over-alignment otherwise.
#if defined(__cpp_aligned_new) && (__cpp_aligned_new >= 201606L)
    return ::operator new(size, std::align_val_t{alignment});
#else
    return std::aligned_alloc(alignment, size);
#endif
}

inline void aligned_free(void* p, usize alignment) {
#if defined(__cpp_aligned_new) && (__cpp_aligned_new >= 201606L)
    ::operator delete(p, std::align_val_t{alignment});
#else
    std::free(p);
#endif
}

// ---------------------------------------------------------------------------
// Power-of-two helpers
// ---------------------------------------------------------------------------
inline bool is_power_of_two(u64 x) { return x != 0 && (x & (x - 1)) == 0; }

inline u64 next_power_of_two(u64 x) {
    if (x <= 1) return 1;
    return u64(1) << (64 - std::countl_zero(x - 1));
}

inline u64 log2_exact(u64 x) {
    return std::countr_zero(x);
}

// ---------------------------------------------------------------------------
// Ceil / floor division for signed and unsigned
// ---------------------------------------------------------------------------
inline u64 ceildiv(u64 a, u64 b) {
    return (a + b - 1) / b;
}

inline i64 ceildiv_signed(i64 a, i64 b) {
    // Careful with signs; matches Python's math.ceil(a/b) for the quotient.
    if ((a >= 0) == (b > 0)) {
        // Same sign -> positive quotient
        i64 aa = a < 0 ? -a : a;
        i64 bb = b < 0 ? -b : b;
        return (aa + bb - 1) / bb;
    } else {
        i64 aa = a < 0 ? -a : a;
        i64 bb = b < 0 ? -b : b;
        return -static_cast<i64>(aa / bb);
    }
}

} // namespace cg

// ir/attributes.hpp - compile-time constant attributes attached to ops
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cg {

class Attribute;
using AttributePtr = std::shared_ptr<const Attribute>;

enum class AttrKind : u8 {
    Integer,
    Float,
    Bool,
    String,
    IntegerArray,
    BoolArray,
    DType,
    DTypeArray,
};

class Attribute {
public:
    AttrKind kind;

    // Direct values
    i64    integer = 0;
    double real    = 0.0;
    bool   flag    = false;
    std::string str;
    DType  dtype   = DType::F32;

    // Array values
    std::vector<i64>  ints;
    std::vector<bool> bools;
    std::vector<DType> dtypes;

    // ---- Factories --------------------------------------------------------
    static AttributePtr make_integer(i64 v) {
        auto a = std::make_shared<Attribute>();
        a->kind = AttrKind::Integer;
        a->integer = v;
        return a;
    }
    static AttributePtr make_float(double v) {
        auto a = std::make_shared<Attribute>();
        a->kind = AttrKind::Float;
        a->real = v;
        return a;
    }
    static AttributePtr make_bool(bool v) {
        auto a = std::make_shared<Attribute>();
        a->kind = AttrKind::Bool;
        a->flag = v;
        return a;
    }
    static AttributePtr make_string(std::string v) {
        auto a = std::make_shared<Attribute>();
        a->kind = AttrKind::String;
        a->str = std::move(v);
        return a;
    }
    static AttributePtr make_int_array(std::vector<i64> v) {
        auto a = std::make_shared<Attribute>();
        a->kind = AttrKind::IntegerArray;
        a->ints = std::move(v);
        return a;
    }
    static AttributePtr make_bool_array(std::vector<bool> v) {
        auto a = std::make_shared<Attribute>();
        a->kind = AttrKind::BoolArray;
        a->bools = std::move(v);
        return a;
    }
    static AttributePtr make_dtype(DType v) {
        auto a = std::make_shared<Attribute>();
        a->kind = AttrKind::DType;
        a->dtype = v;
        return a;
    }
    static AttributePtr make_dtype_array(std::vector<DType> v) {
        auto a = std::make_shared<Attribute>();
        a->kind = AttrKind::DTypeArray;
        a->dtypes = std::move(v);
        return a;
    }

    bool structurally_equal(const Attribute& other) const;
    std::string to_string() const;
};

// An attribute dictionary keyed by name. Ordering is stable (insertion order).
class AttributeDict {
public:
    bool has(StringRef name) const {
        for (auto& [k, _] : entries_) if (k == name) return true;
        return false;
    }

    AttributePtr get(StringRef name) const {
        for (auto& [k, v] : entries_) if (k == name) return v;
        return nullptr;
    }

    void set(StringRef name, AttributePtr v) {
        for (auto& [k, _] : entries_) {
            if (k == name) {
                // update in place
                // (entries_ is a vector of pairs; we re-find by index)
                for (auto& [k2, v2] : entries_) {
                    if (k2 == name) { v2 = std::move(v); return; }
                }
            }
        }
        entries_.emplace_back(std::string(name), std::move(v));
    }

    void remove(StringRef name) {
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->first == name) { entries_.erase(it); return; }
        }
    }

    auto begin() { return entries_.begin(); }
    auto end()   { return entries_.end(); }
    auto begin() const { return entries_.begin(); }
    auto end()   const { return entries_.end(); }

    bool empty() const { return entries_.empty(); }
    usize size() const { return entries_.size(); }

private:
    std::vector<std::pair<std::string, AttributePtr>> entries_;
};

} // namespace cg

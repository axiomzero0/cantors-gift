// vendor/dispatch.hpp - vendor library dispatch
//
// Vendor libraries (cuBLAS, cuDNN, rocBLAS, oneDNN, CUTLASS) are treated as
// *alternative implementations* of tensor operations. The optimizer compares
// a generated kernel against the vendor library and selects whichever is
// faster.
//
// The dispatch interface is backend-agnostic: a VendorKernel describes what
// op it implements and what shapes/dtypes it supports. The runtime queries
// the vendor library at load time and registers available kernels.
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"
#include "cg/ir/operation.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cg {

enum class VendorKind : u8 {
    cuBLAS,
    cuDNN,
    rocBLAS,
    MIOpen,
    oneDNN,
    CUTLASS,
    ComposableKernel,
};

struct VendorOpSupport {
    Opcode opcode;
    std::vector<DType> supported_dtypes;
    // Minimum and maximum rank supported.
    u8 min_rank = 0;
    u8 max_rank = UINT8_MAX;
    // If non-empty, only these specific shapes are supported (e.g. tensor
    // core tiles).
    std::vector<std::vector<i64>> supported_shapes;
};

class VendorKernel {
public:
    virtual ~VendorKernel() = default;
    virtual VendorKind vendor() const = 0;
    virtual std::string name() const = 0;
    virtual Opcode opcode() const = 0;

    // True iff this kernel can implement `op` with the given shapes/dtypes.
    virtual bool supports(const Operation& op) const = 0;

    // Estimated runtime in seconds (for cost comparison). Returns nullopt
    // if the vendor library hasn't been queried yet.
    virtual std::optional<double> estimated_runtime() const { return std::nullopt; }

    // Actually execute the op (requires the vendor library to be loaded).
    // Returns true on success.
    virtual bool execute(const Operation& op, void* stream) { return false; }
};

class VendorDispatcher {
public:
    static VendorDispatcher& instance();

    void register_kernel(std::unique_ptr<VendorKernel> kernel);

    // Find the best vendor kernel for `op`, or nullptr if none support it.
    VendorKernel* find_best(const Operation& op) const;

    // List all registered kernels.
    const std::vector<std::unique_ptr<VendorKernel>>& kernels() const {
        return kernels_;
    }

    // Check if a specific vendor library is available (loaded).
    bool has_vendor(VendorKind kind) const;

private:
    VendorDispatcher();
    std::vector<std::unique_ptr<VendorKernel>> kernels_;
};

// Vendor library names for diagnostics.
inline std::string_view vendor_name(VendorKind k) {
    switch (k) {
        case VendorKind::cuBLAS:           return "cuBLAS";
        case VendorKind::cuDNN:            return "cuDNN";
        case VendorKind::rocBLAS:          return "rocBLAS";
        case VendorKind::MIOpen:           return "MIOpen";
        case VendorKind::oneDNN:           return "oneDNN";
        case VendorKind::CUTLASS:          return "CUTLASS";
        case VendorKind::ComposableKernel: return "ComposableKernel";
    }
    return "?";
}

} // namespace cg

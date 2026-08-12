// vendor/dispatch.cpp - vendor library dispatch implementation
#include "cg/vendor/dispatch.hpp"
#include "cg/ir/ops.hpp"

namespace cg {

VendorDispatcher& VendorDispatcher::instance() {
    static VendorDispatcher d;
    return d;
}

VendorDispatcher::VendorDispatcher() = default;

void VendorDispatcher::register_kernel(std::unique_ptr<VendorKernel> kernel) {
    kernels_.push_back(std::move(kernel));
}

VendorKernel* VendorDispatcher::find_best(const Operation& op) const {
    VendorKernel* best = nullptr;
    double best_time = std::numeric_limits<double>::infinity();
    for (auto& k : kernels_) {
        if (!k->supports(op)) continue;
        auto rt = k->estimated_runtime();
        if (rt && *rt < best_time) {
            best_time = *rt;
            best = k.get();
        }
    }
    // If no kernel has a runtime estimate, return the first that supports.
    if (!best) {
        for (auto& k : kernels_) {
            if (k->supports(op)) { best = k.get(); break; }
        }
    }
    return best;
}

bool VendorDispatcher::has_vendor(VendorKind kind) const {
    for (auto& k : kernels_) {
        if (k->vendor() == kind) return true;
    }
    return false;
}

// ---- Built-in vendor kernel stubs (always available) ----
// These describe what the vendor libraries *would* support if loaded. The
// runtime checks at load time whether the actual library is present and
// updates the estimated_runtime / execute methods accordingly.

namespace {

class cuBLASMatmul : public VendorKernel {
public:
    VendorKind vendor() const override { return VendorKind::cuBLAS; }
    std::string name() const override { return "cublas_matmul"; }
    Opcode opcode() const override { return OP_MATMUL; }

    bool supports(const Operation& op) const override {
        if (op.opcode != OP_MATMUL) return false;
        if (op.operands.size() != 2) return false;
        auto a = op.operands[0].as_tensor();
        auto b = op.operands[1].as_tensor();
        if (!a || !b) return false;
        // cuBLAS supports F32, F16, BF16, F64, I8.
        return a->dtype == DType::F32 || a->dtype == DType::F16 ||
               a->dtype == DType::BF16 || a->dtype == DType::F64 ||
               a->dtype == DType::I8;
    }

    std::optional<double> estimated_runtime() const override {
        // cuBLAS is generally the fastest matmul on NVIDIA GPUs.
        return 1e-5;
    }
};

class cuDNNConv2D : public VendorKernel {
public:
    VendorKind vendor() const override { return VendorKind::cuDNN; }
    std::string name() const override { return "cudnn_conv2d"; }
    Opcode opcode() const override { return OP_CONV2D; }

    bool supports(const Operation& op) const override {
        if (op.opcode != OP_CONV2D) return false;
        if (op.operands.size() != 2) return false;
        auto in = op.operands[0].as_tensor();
        if (!in) return false;
        return in->dtype == DType::F32 || in->dtype == DType::F16 ||
               in->dtype == DType::BF16;
    }

    std::optional<double> estimated_runtime() const override {
        return 5e-5;
    }
};

class rocBLASMatmul : public VendorKernel {
public:
    VendorKind vendor() const override { return VendorKind::rocBLAS; }
    std::string name() const override { return "rocblas_matmul"; }
    Opcode opcode() const override { return OP_MATMUL; }

    bool supports(const Operation& op) const override {
        if (op.opcode != OP_MATMUL) return false;
        if (op.operands.size() != 2) return false;
        auto a = op.operands[0].as_tensor();
        if (!a) return false;
        return a->dtype == DType::F32 || a->dtype == DType::F16 ||
               a->dtype == DType::BF16;
    }

    std::optional<double> estimated_runtime() const override {
        return 1.2e-5;
    }
};

class oneDNNMatmul : public VendorKernel {
public:
    VendorKind vendor() const override { return VendorKind::oneDNN; }
    std::string name() const override { return "onednn_matmul"; }
    Opcode opcode() const override { return OP_MATMUL; }

    bool supports(const Operation& op) const override {
        if (op.opcode != OP_MATMUL) return false;
        if (op.operands.size() != 2) return false;
        auto a = op.operands[0].as_tensor();
        if (!a) return false;
        return a->dtype == DType::F32;
    }

    std::optional<double> estimated_runtime() const override {
        return 2e-5;
    }
};

} // namespace

// Register built-in vendor kernel descriptors at process start.
namespace {
struct VendorInit {
    VendorInit() {
        auto& d = VendorDispatcher::instance();
        d.register_kernel(std::make_unique<cuBLASMatmul>());
        d.register_kernel(std::make_unique<cuDNNConv2D>());
        d.register_kernel(std::make_unique<rocBLASMatmul>());
        d.register_kernel(std::make_unique<oneDNNMatmul>());
    }
};
static VendorInit g_vendor_init;
} // namespace

} // namespace cg

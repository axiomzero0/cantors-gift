// backend/nvidia_backend.hpp - NVIDIA GPU backend (PTX-based)
//
// Compiles a CGModule to an Executable containing PTX text. The NVIDIA
// driver (loaded at runtime) compiles PTX -> SASS. We own everything up to
// PTX.
#pragma once

#include "cg/backend/backend.hpp"
#include "cg/backend/ptx/ptx_emitter.hpp"

namespace cg {

class NvidiaBackend : public MachineBackend {
public:
    explicit NvidiaBackend(std::string sm_target = "sm_80")
        : sm_target_(std::move(sm_target)) {
        target_info_.name = "nvidia_" + sm_target_;
        target_info_.device = DeviceId::cuda();
        target_info_.hardware = HardwareModel::generic_nvidia_gpu();
    }

    std::string name() const override { return "nvidia"; }
    const TargetInfo& target_info() const override { return target_info_; }

    std::unique_ptr<Executable> compile(const CGModule& module) override;
    std::optional<std::string> emit_text(const CGModule& module) override;

private:
    std::string sm_target_;
    TargetInfo target_info_;
};

} // namespace cg

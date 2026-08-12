// backend/amd_backend.hpp - AMD GPU backend (GCN ISA text emitter)
//
// Emits AMD GCN (Graphics Core Next) ISA text from Codegen IR. Like the PTX
// emitter, this produces text that the AMD toolchain (ROCm assembler) can
// assemble into a code object. We own everything up to the GCN ISA; the
// AMD assembler handles the final encoding.
//
// The emitter supports:
//   - kernel declarations with amdgpu_kernel
//   - register declarations (s, v registers)
//   - global loads/stores (buffer_load / buffer_store)
//   - shared memory (lds_load / lds_store)
//   - FMA (v_fma_f32)
//   - vector ops (v_add_f32, v_mul_f32)
//   - barriers (s_barrier)
//   - thread index intrinsics
#pragma once

#include "cg/backend/backend.hpp"
#include "cg/codegen/codegen_ir.hpp"

#include <string>

namespace cg {

class GCNEmitter {
public:
    GCNEmitter() = default;

    std::string emit_kernel(const CGFunction& fn,
                            const std::string& kernel_name,
                            const std::string& gcn_target = "gfx908");

    std::string emit_body(const CGFunction& fn);

private:
    std::string reg_name(const CGValue& v) const;
    std::string gcn_type_name(DType dt) const;
    std::string emit_instruction(const CGInstruction& inst) const;
};

class AmdBackend : public MachineBackend {
public:
    explicit AmdBackend(std::string gcn_target = "gfx908")
        : gcn_target_(std::move(gcn_target)) {
        target_info_.name = "amd_" + gcn_target_;
        target_info_.device = DeviceId::rocm();
        // Approximate CDNA2 (MI210) specs.
        target_info_.hardware = HardwareModel::generic_nvidia_gpu();
        target_info_.hardware.name = "amd_" + gcn_target_;
        target_info_.hardware.device = DeviceId::rocm();
    }

    std::string name() const override { return "amd"; }
    const TargetInfo& target_info() const override { return target_info_; }

    std::unique_ptr<Executable> compile(const CGModule& module) override;
    std::optional<std::string> emit_text(const CGModule& module) override;

private:
    std::string gcn_target_;
    TargetInfo target_info_;
};

} // namespace cg

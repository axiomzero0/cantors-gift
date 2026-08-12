// backend/cpu_backend.hpp - CPU backend (x86-64 via X86Emitter)
//
// Compiles a CGModule to an Executable containing raw x86-64 machine code
// bytes. The bytes can be cast to a function pointer and called directly
// (after making the memory executable via mmap or VirtualProtect).
#pragma once

#include "cg/backend/backend.hpp"
#include "cg/backend/x86/x86_emitter.hpp"

namespace cg {

class CpuBackend : public MachineBackend {
public:
    CpuBackend() {
        target_info_.name = "cpu_x86_64";
        target_info_.device = DeviceId::cpu();
        target_info_.hardware = HardwareModel::generic_cpu();
    }

    std::string name() const override { return "cpu"; }
    const TargetInfo& target_info() const override { return target_info_; }

    std::unique_ptr<Executable> compile(const CGModule& module) override;
    std::optional<std::string> emit_text(const CGModule& module) override;

private:
    TargetInfo target_info_;
};

// A simple x86 disassembler for debugging (produces a human-readable string
// of the emitted bytes). Not a full disassembler; just enough to verify the
// emitter is producing sensible output.
std::string disassemble_x86(Span<const u8> bytes);

} // namespace cg

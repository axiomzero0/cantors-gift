// backend/nvidia_backend.cpp
#include "cg/backend/nvidia_backend.hpp"
#include "cg/backend/lowering.hpp"

namespace cg {

std::unique_ptr<Executable> NvidiaBackend::compile(const CGModule& module) {
    auto exe = std::make_unique<Executable>();
    exe->target_device = DeviceId::cuda();

    for (const auto& fn : module.functions) {
        std::string ptx = lower_to_ptx(fn, fn.name, sm_target_);
        exe->ptx_text += ptx + "\n";
        exe->entrypoints.push_back({fn.name, 0});
    }

    return exe;
}

std::optional<std::string> NvidiaBackend::emit_text(const CGModule& module) {
    std::string out;
    for (const auto& fn : module.functions) {
        out += lower_to_ptx(fn, fn.name, sm_target_);
        out += "\n";
    }
    return out;
}

} // namespace cg

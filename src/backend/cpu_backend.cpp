// backend/cpu_backend.cpp
#include "cg/backend/cpu_backend.hpp"
#include "cg/backend/lowering.hpp"

#include <iomanip>
#include <sstream>

namespace cg {

std::unique_ptr<Executable> CpuBackend::compile(const CGModule& module) {
    auto exe = std::make_unique<Executable>();
    exe->target_device = DeviceId::cpu();

    // Concatenate all function bodies into one blob. Each function starts
    // at the current end of the blob.
    for (const auto& fn : module.functions) {
        u64 offset = exe->machine_code.size();
        auto bytes = lower_to_x86(fn);
        exe->machine_code.insert(exe->machine_code.end(),
                                  bytes.begin(), bytes.end());
        exe->entrypoints.push_back({fn.name, offset});
    }

    // Generate a simple disassembly for debugging.
    exe->disassembly = disassemble_x86(make_span(exe->machine_code));

    return exe;
}

std::optional<std::string> CpuBackend::emit_text(const CGModule& module) {
    std::ostringstream os;
    for (const auto& fn : module.functions) {
        auto bytes = lower_to_x86(fn);
        os << "; function " << fn.name << " (" << bytes.size() << " bytes)\n";
        os << disassemble_x86(make_span(bytes)) << "\n";
    }
    return os.str();
}

std::string disassemble_x86(Span<const u8> bytes) {
    std::ostringstream os;
    // Extremely simple: just dump bytes in hex with offsets.
    for (usize i = 0; i < bytes.size(); ++i) {
        if (i % 16 == 0) {
            if (i) os << "\n";
            os << std::hex << std::setfill('0') << std::setw(4) << i << ": ";
        }
        os << std::hex << std::setfill('0') << std::setw(2)
           << static_cast<int>(bytes[i]) << " ";
    }
    os << std::dec << "\n";
    return os.str();
}

} // namespace cg

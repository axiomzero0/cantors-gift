// backend/backend.hpp - backend interface
//
// A MachineBackend takes a CodegenModule and produces an Executable. We
// deliberately keep the interface narrow: the backend does NOT see Tensor IR,
// does NOT participate in scheduling, and does NOT see analyses. The
// optimizer hands it a finalized CodegenModule.
//
// Concrete backends (X86Backend via Xbyak/AsmJit, NvidiaBackend via PTX,
// AmdBackend via AMD assembler) live in their own subdirectories and are
// compiled in only when their respective CMake options are enabled.
#pragma once

#include "cg/codegen/codegen_ir.hpp"
#include "cg/cost/hardware_model.hpp"
#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cg {

// A blob of machine code (or PTX/ISA text) plus metadata.
class Executable {
public:
    std::string name;
    DeviceId    target_device;
    // Either raw bytes (CPU) or PTX text (NVIDIA).
    std::vector<u8>  machine_code;
    std::string      ptx_text;
    // Symbol table: entrypoint name -> offset into machine_code.
    std::vector<std::pair<std::string, u64>> entrypoints;
    // Total shared-memory bytes the kernel needs.
    u64 shared_mem_bytes = 0;
    // Total constant-memory bytes the kernel needs.
    u64 constant_mem_bytes = 0;
    // Required number of threads per block (GPU) or 0 for CPU.
    u32 threads_per_block = 0;
    // Optional human-readable disassembly for debugging.
    std::string disassembly;
};

// TargetInfo describes the static capabilities of a backend. It is exposed
// to the optimizer so it can make target-aware decisions without depending
// on a concrete backend class.
class TargetInfo {
public:
    std::string name;
    DeviceId    device;
    HardwareModel hardware;

    // Supported dtypes for vector operations.
    std::vector<DType> supported_vector_dtypes;
    u32 max_vector_width_bytes = 16;

    // Supported tile shapes for tensor cores (M, N, K).
    std::vector<std::tuple<u32, u32, u32>> tensor_core_tiles;

    bool supports_tensor_core(DType dt) const {
        // Tensor cores are reported via hardware.tensor_core_flops_per_sec.
        return hardware.tensor_core_flops_per_sec.count(static_cast<u8>(dt)) > 0;
    }
};

// An emitter produces machine code (or PTX) for a single CGFunction.
// Backends implement this.
class MachineEmitter {
public:
    virtual ~MachineEmitter() = default;
    virtual void emit(const CGFunction& fn) = 0;
    virtual std::vector<u8> take_bytes() = 0;
    virtual std::string take_text() { return {}; }
};

class MachineBackend {
public:
    virtual ~MachineBackend() = default;
    virtual std::string name() const = 0;
    virtual const TargetInfo& target_info() const = 0;

    // Lower a CodegenModule into an Executable.
    virtual std::unique_ptr<Executable>
    compile(const CGModule& module) = 0;

    // Optional: emit text (PTX or assembly) for debugging.
    virtual std::optional<std::string>
    emit_text(const CGModule& module) { return std::nullopt; }
};

} // namespace cg

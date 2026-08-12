// backend/ptx_emitter.hpp - NVIDIA PTX text emitter
//
// Emits valid PTX (Parallel Thread Execution) text from Codegen IR functions.
// PTX is NVIDIA's textual GPU assembly; the NVIDIA driver compiles PTX into
// SASS (the actual GPU machine code). We own everything up to PTX; NVIDIA
// owns the final encoding.
//
// The emitter supports:
//   - kernel declarations with .param pointers
//   - register declarations (.reg .f32, .reg .u64, etc.)
//   - shared memory declarations (.shared)
//   - global loads/stores (ld.global / st.global)
//   - shared loads/stores (ld.shared / st.shared)
//   - FMA (fma.rn.f32)
//   - vector loads/stores (ld.global.v4.f32)
//   - barriers (bar.sync)
//   - thread index intrinsics (tid, ntid, ctaid)
//   - arithmetic (add, sub, mul, mad)
//   - predicated execution
//
// The output is a PTX string that can be fed to cuModuleLoadData or written
// to a .ptx file for offline compilation with ptxas.
#pragma once

#include "cg/codegen/codegen_ir.hpp"
#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"

#include <sstream>
#include <string>
#include <unordered_map>

namespace cg {

class PTXEmitter {
public:
    PTXEmitter() = default;

    // Emit a complete PTX module for the given function.
    // `kernel_name` becomes the .entry name.
    // `sm_target` selects the .target (e.g. "sm_80", "sm_90").
    std::string emit_kernel(const CGFunction& fn,
                            const std::string& kernel_name,
                            const std::string& sm_target = "sm_80");

    // Lower-level: emit just the body (instructions) of a function.
    std::string emit_body(const CGFunction& fn);

    // Register a kernel argument (name + dtype + is_pointer).
    struct Arg {
        std::string name;
        DType dtype;
        bool is_pointer;
    };

    void add_arg(Arg a) { args_.push_back(std::move(a)); }

private:
    std::vector<Arg> args_;

    std::string ptx_type_name(DType dt) const;
    std::string ptx_type_name_vec(DType dt, u8 width) const;
    std::string reg_name(const CGValue& v) const;
    std::string emit_instruction(const CGInstruction& inst) const;
};

} // namespace cg

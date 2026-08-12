// backend/lowering.hpp - lower Codegen IR to machine-specific code
//
// This is the bridge between the backend-agnostic Codegen IR and the
// concrete emitters (PTX, x86). It walks a CGFunction and emits the
// corresponding machine code via the chosen emitter.
#pragma once

#include "cg/backend/ptx/ptx_emitter.hpp"
#include "cg/backend/x86/x86_emitter.hpp"
#include "cg/codegen/codegen_ir.hpp"

#include <string>
#include <vector>

namespace cg {

// Lower a CGFunction to PTX text.
std::string lower_to_ptx(const CGFunction& fn,
                         const std::string& kernel_name,
                         const std::string& sm_target = "sm_80");

// Lower a CGFunction to x86-64 machine code bytes.
// Produces a complete function: prologue, body, epilogue.
// The returned bytes can be cast to a function pointer and called (after
// making the memory executable).
std::vector<u8> lower_to_x86(const CGFunction& fn);

} // namespace cg

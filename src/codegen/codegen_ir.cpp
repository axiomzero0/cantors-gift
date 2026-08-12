// codegen/codegen_ir.cpp - helpers
#include "cg/codegen/codegen_ir.hpp"

namespace cg {

CGInstruction make_load(CGValue dst, CGValue ptr, CGValue offset, MemorySpace ms) {
    CGInstruction i;
    i.opcode = CGOpcode::Load;
    i.operands = {ptr, offset};
    i.results = {dst};
    i.mem_space = ms;
    return i;
}

CGInstruction make_store(CGValue ptr, CGValue offset, CGValue val, MemorySpace ms) {
    CGInstruction i;
    i.opcode = CGOpcode::Store;
    i.operands = {ptr, offset, val};
    i.mem_space = ms;
    return i;
}

CGInstruction make_fma(CGValue acc, CGValue a, CGValue b) {
    CGInstruction i;
    i.opcode = CGOpcode::FMA;
    i.operands = {acc, a, b};
    i.results = {acc};
    return i;
}

CGInstruction make_vector_load(CGValue dst, CGValue ptr, CGValue offset,
                               u8 width, MemorySpace ms) {
    CGInstruction i;
    i.opcode = CGOpcode::VectorLoad;
    i.operands = {ptr, offset};
    i.results = {dst};
    i.mem_space = ms;
    i.attributes.set("width", Attribute::make_integer(width));
    return i;
}

CGInstruction make_vector_store(CGValue ptr, CGValue offset, CGValue val,
                                MemorySpace ms) {
    CGInstruction i;
    i.opcode = CGOpcode::VectorStore;
    i.operands = {ptr, offset, val};
    i.mem_space = ms;
    return i;
}

CGInstruction make_barrier() {
    CGInstruction i;
    i.opcode = CGOpcode::Barrier;
    return i;
}

} // namespace cg

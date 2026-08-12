// codegen/codegen_ir.hpp - low-level IR for codegen
//
// The Codegen IR is the final representation before emission. It contains
// primitive operations (loads, stores, FMAs, vector ops, barriers, async
// copies) but stops short of being a full machine IR. The backend emitter
// (Xbyak/PTX/AMD) consumes Codegen IR and produces machine code.
//
// We deliberately do NOT model physical registers here. The scheduler
// estimated register pressure earlier; the emitter handles spilling.
#pragma once

#include "cg/core/dtype.hpp"
#include "cg/core/util.hpp"
#include "cg/ir/attributes.hpp"
#include "cg/layout/layout.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cg {

enum class CGOpcode : u8 {
    Load,
    Store,
    VectorLoad,
    VectorStore,
    FMA,
    Add,
    Mul,
    Reduce,
    Barrier,
    AsyncCopy,
    Prefetch,
    Broadcast,
    Shuffle,
    Cmp,
    Select,
    Cast,
    Const,
};

class CGValue {
public:
    u32 id = 0;
    DType dtype = DType::F32;
    u8  width = 1;       // vector width in elements
    std::string name;    // optional debug name

    bool operator==(const CGValue& o) const { return id == o.id; }
};

class CGInstruction {
public:
    CGOpcode opcode;
    SmallVector<CGValue, 3> operands;
    SmallVector<CGValue, 2> results;
    AttributeDict attributes;
    MemorySpace mem_space = MemorySpace::Generic;
    std::string comment;
};

class CGFunction {
public:
    std::string name;
    std::vector<CGInstruction> instructions;
    std::vector<CGValue>       args;
    std::vector<CGValue>       returns;

    CGValue allocate(DType dt, u8 width = 1) {
        CGValue v;
        v.id = next_id_++;
        v.dtype = dt;
        v.width = width;
        return v;
    }

    void emit(CGInstruction inst) {
        instructions.push_back(std::move(inst));
    }

private:
    u32 next_id_ = 1;
};

class CGModule {
public:
    std::vector<CGFunction> functions;

    CGFunction& create_function(std::string name) {
        functions.emplace_back();
        functions.back().name = std::move(name);
        return functions.back();
    }
};

// Helpers
CGInstruction make_load(CGValue dst, CGValue ptr, CGValue offset,
                        MemorySpace ms = MemorySpace::Generic);
CGInstruction make_store(CGValue ptr, CGValue offset, CGValue val,
                         MemorySpace ms = MemorySpace::Generic);
CGInstruction make_fma(CGValue acc, CGValue a, CGValue b);
CGInstruction make_vector_load(CGValue dst, CGValue ptr, CGValue offset,
                               u8 width, MemorySpace ms = MemorySpace::Generic);
CGInstruction make_vector_store(CGValue ptr, CGValue offset, CGValue val,
                                MemorySpace ms = MemorySpace::Generic);
CGInstruction make_barrier();

} // namespace cg

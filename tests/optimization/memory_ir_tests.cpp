// tests/memory_ir_tests.cpp - Memory IR alloc/free insertion tests
#include "cg/analysis/analysis.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"
#include "cg/optimization/memory/memory_planning.hpp"
#include "cg/optimization/dce/dce.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

TEST(MemoryIR, InsertsAllocOps) {
    Module m;
    auto f = m.create_function("mem",
        {make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b(f);
    auto r1 = b.relu(f->args()[0]);
    auto r2 = b.exp(r1);
    b.output_tensor(r2);

    AnalysisManager am(m);
    MemoryPlanningPass p;
    p.run(m, am);

    // There should be OP_ALLOC ops in the block.
    usize allocs = 0;
    for (auto& op : *f->entry()) {
        if (op.opcode == OP_ALLOC) ++allocs;
    }
    EXPECT_GT(allocs, 0u);
}

TEST(MemoryIR, InsertsFreeOps) {
    Module m;
    auto f = m.create_function("mem",
        {make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b(f);
    auto r1 = b.relu(f->args()[0]);
    auto r2 = b.exp(r1);
    b.output_tensor(r2);

    AnalysisManager am(m);
    MemoryPlanningPass p;
    p.run(m, am);

    usize frees = 0;
    for (auto& op : *f->entry()) {
        if (op.opcode == OP_FREE) ++frees;
    }
    EXPECT_GT(frees, 0u);
}

TEST(MemoryIR, AllocAndFreeCountsMatch) {
    Module m;
    auto f = m.create_function("mem",
        {make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b(f);
    auto r1 = b.relu(f->args()[0]);
    auto r2 = b.exp(r1);
    b.output_tensor(r2);

    AnalysisManager am(m);
    MemoryPlanningPass p;
    p.run(m, am);

    usize allocs = 0, frees = 0;
    for (auto& op : *f->entry()) {
        if (op.opcode == OP_ALLOC) ++allocs;
        if (op.opcode == OP_FREE) ++frees;
    }
    EXPECT_EQ(allocs, frees);
}

TEST(MemoryIR, BufferIdAnnotationPresent) {
    Module m;
    auto f = m.create_function("mem",
        {make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b(f);
    auto r1 = b.relu(f->args()[0]);
    auto r2 = b.exp(r1);
    b.output_tensor(r2);

    AnalysisManager am(m);
    MemoryPlanningPass p;
    p.run(m, am);

    bool found_buffer_id = false;
    for (auto& op : *f->entry()) {
        if (op.attributes.get("buffer_id")) found_buffer_id = true;
    }
    EXPECT_TRUE(found_buffer_id);
}

TEST(MemoryIR, AllocHasBytesAttribute) {
    Module m;
    auto f = m.create_function("mem",
        {make_tensor_type({64, 64}, DType::F32)},
        {make_tensor_type({64, 64}, DType::F32)});
    Builder b(f);
    auto r1 = b.relu(f->args()[0]);
    auto r2 = b.exp(r1);
    b.output_tensor(r2);

    AnalysisManager am(m);
    MemoryPlanningPass p;
    p.run(m, am);

    for (auto& op : *f->entry()) {
        if (op.opcode != OP_ALLOC) continue;
        auto bytes_attr = op.attributes.get("bytes");
        EXPECT_NE(bytes_attr, nullptr);
        if (bytes_attr) {
            EXPECT_GT(bytes_attr->integer, 0);
        }
    }
}

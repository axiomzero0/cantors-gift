// tests/optimization_tests.cpp - pass tests
#include "cg/analysis/analysis.hpp"
#include "cg/analysis/shape_analysis.hpp"
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"
#include "cg/optimization/canonicalize/canonicalize.hpp"
#include "cg/optimization/cse/cse.hpp"
#include "cg/optimization/const_fold/const_fold.hpp"
#include "cg/optimization/dce/dce.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

namespace {

Module build_with_dup() {
    Module m;
    auto f = m.create_function("dup",
        {make_tensor_type({4, 4}, DType::F32),
         make_tensor_type({4, 4}, DType::F32)},
        {make_tensor_type({4, 4}, DType::F32)});
    Builder b(f);
    auto a = f->args()[0];
    auto c = f->args()[1];
    auto x = b.add(a, c);
    auto y = b.add(a, c);
    auto z = b.mul(x, y);
    b.output_tensor(z);
    return m;
}

} // namespace

TEST(Optimization, CSE) {
    auto m = build_with_dup();
    AnalysisManager am(m);
    CSEPass cse;
    cse.run(m, am);

    // Count add ops; should be exactly 1 now.
    usize adds = 0;
    for (auto& f : m.functions()) {
        for (auto& op : *f->entry()) {
            if (op.opcode == OP_ADD) ++adds;
        }
    }
    EXPECT_EQ(adds, 1u);
}

TEST(Optimization, DCE) {
    Module m;
    auto f = m.create_function("dce",
        {make_tensor_type({4, 4}, DType::F32)},
        {make_tensor_type({4, 4}, DType::F32)});
    Builder b(f);
    auto a = f->args()[0];
    auto unused = b.neg(a);              // no users
    auto used    = b.relu(a);
    (void)unused;
    b.output_tensor(used);

    AnalysisManager am(m);
    DCEPass dce;
    dce.run(m, am);

    // The neg op should be removed.
    bool has_neg = false;
    for (auto& op : *f->entry()) {
        if (op.opcode == OP_NEG) has_neg = true;
    }
    EXPECT_FALSE(has_neg);
}

TEST(Optimization, CanonicalizeAddZero) {
    Module m;
    auto f = m.create_function("canon",
        {make_tensor_type({4, 4}, DType::F32)},
        {make_tensor_type({4, 4}, DType::F32)});
    Builder b(f);
    auto a = f->args()[0];
    // Build constant 0 of shape [4,4] f32.
    std::string zero_bytes(4 * 4 * 4, '\0');
    AttributeDict attrs;
    attrs.set("shape", Attribute::make_int_array({4, 4}));
    attrs.set("dtype", Attribute::make_dtype(DType::F32));
    attrs.set("bytes", Attribute::make_string(std::move(zero_bytes)));
    auto* zero_op = b.create(OP_CONSTANT, {}, attrs);
    auto sum = b.add(a, zero_op->results[0]);
    b.output_tensor(sum);

    AnalysisManager am(m);
    CanonicalizePass canon;
    canon.run(m, am);
    DCEPass dce;
    dce.run(m, am);

    // After canonicalization + DCE, x + 0 should be gone entirely.
    bool has_add_with_zero = false;
    for (auto& op : *f->entry()) {
        if (op.opcode == OP_ADD) has_add_with_zero = true;
    }
    EXPECT_FALSE(has_add_with_zero);
}

TEST(Optimization, ConstantFoldAdd) {
    Module m;
    auto f = m.create_function("cf",
        {},
        {make_tensor_type({2}, DType::F32)});
    Builder b(f);
    std::string a_bytes(2 * 4, '\0');
    {
        float v = 1.0f; std::memcpy(a_bytes.data(), &v, 4);
        v = 2.0f; std::memcpy(a_bytes.data() + 4, &v, 4);
    }
    std::string bb_bytes(2 * 4, '\0');
    {
        float v = 3.0f; std::memcpy(bb_bytes.data(), &v, 4);
        v = 4.0f; std::memcpy(bb_bytes.data() + 4, &v, 4);
    }
    AttributeDict a1; a1.set("shape", Attribute::make_int_array({2}));
    a1.set("dtype", Attribute::make_dtype(DType::F32));
    a1.set("bytes", Attribute::make_string(std::move(a_bytes)));
    auto* ca = b.create(OP_CONSTANT, {}, a1);

    AttributeDict a2; a2.set("shape", Attribute::make_int_array({2}));
    a2.set("dtype", Attribute::make_dtype(DType::F32));
    a2.set("bytes", Attribute::make_string(std::move(bb_bytes)));
    auto* cb = b.create(OP_CONSTANT, {}, a2);

    auto sum = b.add(ca->results[0], cb->results[0]);
    b.output_tensor(sum);

    AnalysisManager am(m);
    ConstantFoldingPass cf;
    cf.run(m, am);

    // There should now be a constant op holding [4, 6].
    bool found_folded = false;
    for (auto& op : *f->entry()) {
        if (op.opcode != OP_CONSTANT) continue;
        auto bytes_attr = op.attributes.get("bytes");
        if (!bytes_attr || bytes_attr->kind != AttrKind::String) continue;
        if (bytes_attr->str.size() != 8) continue;
        float v0, v1;
        std::memcpy(&v0, bytes_attr->str.data(), 4);
        std::memcpy(&v1, bytes_attr->str.data() + 4, 4);
        if (v0 == 4.0f && v1 == 6.0f) found_folded = true;
    }
    EXPECT_TRUE(found_folded);
}

TEST(Optimization, PassManagerInvalidatesAnalyses) {
    auto m = build_with_dup();
    AnalysisManager am(m);

    // Get a fresh analysis; CSE doesn't preserve it.
    // (We use ShapeAnalysis as a stand-in; it's invalidated on every
    // non-preserving pass.)
    auto& sa = am.get<ShapeAnalysis>();
    (void)sa;

    PassManager pm;
    pm.add(std::make_unique<CSEPass>());
    pm.run(m, am);

    // The ShapeAnalysis cache should have been dropped.
    EXPECT_FALSE(am.has<ShapeAnalysis>());
}

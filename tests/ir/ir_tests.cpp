// tests/ir_tests.cpp - IR unit tests
#include "cg/ir/builder.hpp"
#include "cg/ir/ops.hpp"
#include "cg/ir/printer.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

namespace {

Module make_simple_module() {
    Module m;
    auto f = m.create_function("test",
        {make_tensor_type({16, 32}, DType::F32),
         make_tensor_type({32, 64}, DType::F32)},
        {make_tensor_type({16, 64}, DType::F32)});
    Builder b(f);
    auto a = f->args()[0];
    auto bb = f->args()[1];
    auto c = b.matmul(a, bb);
    auto d = b.relu(c);
    b.output_tensor(d);
    return m;
}

} // namespace

TEST(IR, BuildAndPrint) {
    auto m = make_simple_module();
    auto s = to_string(m);
    EXPECT_NE(s.find("matmul"), std::string::npos);
    EXPECT_NE(s.find("relu"), std::string::npos);
    EXPECT_NE(s.find("output"), std::string::npos);
}

TEST(IR, TypeInferenceMatmul) {
    auto m = make_simple_module();
    auto f = m.functions()[0].get();
    ASSERT_NE(f, nullptr);
    // Walk and find matmul op.
    Operation* mm = nullptr;
    for (auto& op : *f->entry()) {
        if (op.opcode == OP_MATMUL) { mm = &op; break; }
    }
    ASSERT_NE(mm, nullptr);
    auto t = mm->results[0].as_tensor();
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape.rank(), 2u);
    EXPECT_EQ(t->shape[0]->value, 16);
    EXPECT_EQ(t->shape[1]->value, 64);
}

TEST(IR, OpRegistryLookup) {
    auto* info = OpRegistry::instance().lookup(OP_ADD);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->name, "add");
    EXPECT_TRUE(info->traits.has(OpTrait::Commutative));
    EXPECT_TRUE(info->traits.has(OpTrait::Pure));

    auto* mm = OpRegistry::instance().lookup(OP_MATMUL);
    ASSERT_NE(mm, nullptr);
    EXPECT_TRUE(mm->traits.has(OpTrait::Reduction));
    EXPECT_TRUE(mm->traits.has(OpTrait::TensorCore));
}

TEST(IR, ReplaceAllUses) {
    Module m;
    auto f = m.create_function("x",
        {make_tensor_type({4, 4}, DType::F32),
         make_tensor_type({4, 4}, DType::F32)},
        {make_tensor_type({4, 4}, DType::F32)});
    Builder b(f);
    auto a = f->args()[0];
    auto c = f->args()[1];
    auto sum = b.add(a, c);
    auto neg = b.neg(sum);
    b.output_tensor(neg);

    // Replace all uses of `sum` with `a`. After replacement, the neg
    // should depend on `a` instead of `sum`.
    m.replace_all_uses(sum, a);
    bool any_uses_sum = false;
    for (auto& op : *f->entry()) {
        for (auto& v : op.operands) {
            if (v == sum) any_uses_sum = true;
        }
    }
    EXPECT_FALSE(any_uses_sum);
}

TEST(IR, Effects) {
    auto* out_info = OpRegistry::instance().lookup(OP_OUTPUT);
    ASSERT_NE(out_info, nullptr);
    EXPECT_FALSE(out_info->effects.is_pure());

    auto* add_info = OpRegistry::instance().lookup(OP_ADD);
    ASSERT_NE(add_info, nullptr);
    EXPECT_TRUE(add_info->effects.is_pure());
}

TEST(IR, BuilderCast) {
    Module m;
    auto f = m.create_function("cast_test",
        {make_tensor_type({4, 4}, DType::F32)},
        {make_tensor_type({4, 4}, DType::F16)});
    Builder b(f);
    auto v = b.cast(f->args()[0], DType::F16);
    b.output_tensor(v);

    auto t = v.as_tensor();
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->dtype, DType::F16);
}

TEST(IR, BuilderReduce) {
    Module m;
    auto f = m.create_function("red_test",
        {make_tensor_type({8, 16, 32}, DType::F32)},
        {make_tensor_type({8, 32}, DType::F32)});
    Builder b(f);
    auto v = b.reduce_sum(f->args()[0], {1}, false);
    b.output_tensor(v);

    auto t = v.as_tensor();
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape.rank(), 2u);
    EXPECT_EQ(t->shape[0]->value, 8);
    EXPECT_EQ(t->shape[1]->value, 32);
}

TEST(IR, BuilderTranspose) {
    Module m;
    auto f = m.create_function("t_test",
        {make_tensor_type({8, 16}, DType::F32)},
        {make_tensor_type({16, 8}, DType::F32)});
    Builder b(f);
    auto v = b.transpose(f->args()[0], {1, 0});
    b.output_tensor(v);

    auto t = v.as_tensor();
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape.rank(), 2u);
    EXPECT_EQ(t->shape[0]->value, 16);
    EXPECT_EQ(t->shape[1]->value, 8);
}

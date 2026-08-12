// tests/layout_tests.cpp - layout tests
#include "cg/layout/layout.hpp"

#include "cg/test/gtest_compat.hpp"

#include <vector>

using namespace cg;

TEST(Layout, RowMajorStrides) {
    Shape s{DimExpr::make_constant(4), DimExpr::make_constant(8),
            DimExpr::make_constant(16)};
    auto l = Layout::make_row_major(s);
    ASSERT_NE(l, nullptr);
    ASSERT_EQ(l->strides.size(), 3u);
    EXPECT_EQ(l->strides[0], 128);
    EXPECT_EQ(l->strides[1], 16);
    EXPECT_EQ(l->strides[2], 1);
    EXPECT_TRUE(l->is_row_major_contiguous());
}

TEST(Layout, ColMajorStrides) {
    Shape s{DimExpr::make_constant(4), DimExpr::make_constant(8)};
    auto l = Layout::make_col_major(s);
    ASSERT_NE(l, nullptr);
    ASSERT_EQ(l->strides.size(), 2u);
    EXPECT_EQ(l->strides[0], 1);
    EXPECT_EQ(l->strides[1], 4);
    EXPECT_FALSE(l->is_row_major_contiguous());
}

TEST(Layout, ByteOffset) {
    Shape s{DimExpr::make_constant(4), DimExpr::make_constant(8)};
    auto l = Layout::make_row_major(s);
    auto off = l->byte_offset(std::vector<i64>{2, 3}, DType::F32);
    ASSERT_TRUE(off.has_value());
    // 2*8 + 3 = 19 elements * 4 bytes = 76
    EXPECT_EQ(*off, 76u);
}

TEST(Layout, TransposeByteOffset) {
    Shape s{DimExpr::make_constant(4), DimExpr::make_constant(8)};
    auto base = Layout::make_row_major(s);
    auto t = Layout::make_transpose(base, {1, 0});
    ASSERT_NE(t, nullptr);
    ASSERT_EQ(t->shape.rank(), 2u);
    EXPECT_EQ(t->shape[0]->value, 8);
    EXPECT_EQ(t->shape[1]->value, 4);
    // After transpose, output (i,j) -> base (j,i).
    // base(j,i) = j*8 + i.
    // For output (3, 2): base (2, 3) = 2*8 + 3 = 19 elements * 4 bytes = 76.
    auto off = t->byte_offset(std::vector<i64>{3, 2}, DType::F32);
    ASSERT_TRUE(off.has_value());
    EXPECT_EQ(*off, (2u * 8u + 3u) * 4u);
}

TEST(Layout, StructurallyEqual) {
    Shape s{DimExpr::make_constant(4), DimExpr::make_constant(8)};
    auto l1 = Layout::make_row_major(s);
    auto l2 = Layout::make_row_major(s);
    EXPECT_TRUE(l1->structurally_equal(*l2));
}

TEST(Layout, Slice) {
    Shape s{DimExpr::make_constant(8), DimExpr::make_constant(8)};
    auto base = Layout::make_row_major(s);
    SmallVector<std::pair<i64, i64>> ranges;
    ranges.push_back({2, 4});
    ranges.push_back({1, 3});
    auto sl = Layout::make_slice(base, ranges);
    ASSERT_EQ(sl->shape.rank(), 2u);
    EXPECT_EQ(sl->shape[0]->value, 4);
    EXPECT_EQ(sl->shape[1]->value, 3);
    // Offsets: index (0,0) -> base (2,1).
    auto off = sl->byte_offset(std::vector<i64>{0, 0}, DType::F32);
    ASSERT_TRUE(off.has_value());
    EXPECT_EQ(*off, (2u * 8u + 1u) * 4u);
}

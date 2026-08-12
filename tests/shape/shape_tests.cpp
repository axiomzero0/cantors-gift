// tests/shape_tests.cpp - shape system unit tests
#include "cg/shape/simplifier.hpp"
#include "cg/shape/solver.hpp"
#include "cg/shape/inference.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

TEST(Shape, SimplifyIdentity) {
    auto N = DimExpr::make_symbol(1, "N");
    auto e = simplify_dim(DimExpr::make_add(N, DimExpr::make_constant(0)));
    EXPECT_TRUE(e->is_symbol());
    EXPECT_EQ(e->symbol.id, 1u);
}

TEST(Shape, SimplifyConstants) {
    auto e = simplify_dim(DimExpr::make_add(
        DimExpr::make_constant(7), DimExpr::make_constant(5)));
    ASSERT_TRUE(e->is_constant());
    EXPECT_EQ(e->value, 12);
}

TEST(Shape, SimplifyMulByOne) {
    auto N = DimExpr::make_symbol(1, "N");
    auto e = simplify_dim(DimExpr::make_mul(N, DimExpr::make_constant(1)));
    EXPECT_TRUE(e->is_symbol());
}

TEST(Shape, SimplifyMulByZero) {
    auto N = DimExpr::make_symbol(1, "N");
    auto e = simplify_dim(DimExpr::make_mul(N, DimExpr::make_constant(0)));
    ASSERT_TRUE(e->is_constant(0));
}

TEST(Shape, CeilDivSimplification) {
    auto e = simplify_dim(DimExpr::make_ceil_div(
        DimExpr::make_constant(33), DimExpr::make_constant(8)));
    ASSERT_TRUE(e->is_constant());
    EXPECT_EQ(e->value, 5);
}

TEST(Shape, CeilDivDivisible) {
    auto e = simplify_dim(DimExpr::make_ceil_div(
        DimExpr::make_constant(32), DimExpr::make_constant(8)));
    ASSERT_TRUE(e->is_constant());
    EXPECT_EQ(e->value, 4);
}

TEST(Shape, ModSimplification) {
    auto e = simplify_dim(DimExpr::make_mod(
        DimExpr::make_constant(17), DimExpr::make_constant(5)));
    ASSERT_TRUE(e->is_constant());
    EXPECT_EQ(e->value, 2);
}

TEST(Shape, ConstraintSetEqualities) {
    auto M = DimExpr::make_symbol(1, "M");
    auto N = DimExpr::make_symbol(2, "N");
    auto K = DimExpr::make_symbol(3, "K");

    ConstraintSet cs;
    cs.add_eq(M, N);
    cs.add_mod_eq(K, 16, 0);

    Solver s(std::move(cs));
    EXPECT_EQ(s.prove_equal(M, N), SolverResult::ProvedTrue);
    EXPECT_EQ(s.prove_divisible(K, 16), SolverResult::ProvedTrue);
    EXPECT_EQ(s.prove_divisible(K, 32), SolverResult::Unknown);
}

TEST(Shape, RangeFromConstants) {
    auto e = DimExpr::make_add(DimExpr::make_constant(3),
                               DimExpr::make_constant(5));
    auto r = Solver(ConstraintSet{}).range(e);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->lo, 8);
    EXPECT_EQ(r->hi, 8);
}

TEST(Shape, InferMatmul) {
    Shape A{DimExpr::make_constant(8), DimExpr::make_constant(16)};
    Shape B{DimExpr::make_constant(16), DimExpr::make_constant(32)};
    auto r = infer_matmul(A, B);
    ASSERT_TRUE(r.ok);
    ASSERT_EQ(r.shape.rank(), 2u);
    EXPECT_EQ(r.shape[0]->value, 8);
    EXPECT_EQ(r.shape[1]->value, 32);
}

TEST(Shape, InferMatmulBatched) {
    Shape A{DimExpr::make_constant(2),
            DimExpr::make_constant(8),
            DimExpr::make_constant(16)};
    Shape B{DimExpr::make_constant(2),
            DimExpr::make_constant(16),
            DimExpr::make_constant(32)};
    auto r = infer_matmul(A, B);
    ASSERT_TRUE(r.ok);
    ASSERT_EQ(r.shape.rank(), 3u);
    EXPECT_EQ(r.shape[0]->value, 2);
    EXPECT_EQ(r.shape[1]->value, 8);
    EXPECT_EQ(r.shape[2]->value, 32);
}

TEST(Shape, InferReduction) {
    Shape A{DimExpr::make_constant(8), DimExpr::make_constant(16)};
    std::vector<i32> axes = {1};
    auto r = infer_reduction(A, make_span(axes), false);
    ASSERT_TRUE(r.ok);
    ASSERT_EQ(r.shape.rank(), 1u);
    EXPECT_EQ(r.shape[0]->value, 8);
}

TEST(Shape, InferReductionKeepDims) {
    Shape A{DimExpr::make_constant(8), DimExpr::make_constant(16)};
    std::vector<i32> axes = {1};
    auto r = infer_reduction(A, make_span(axes), true);
    ASSERT_TRUE(r.ok);
    ASSERT_EQ(r.shape.rank(), 2u);
    EXPECT_EQ(r.shape[0]->value, 8);
    EXPECT_EQ(r.shape[1]->value, 1);
}

TEST(Shape, InferTranspose) {
    Shape A{DimExpr::make_constant(8), DimExpr::make_constant(16),
            DimExpr::make_constant(32)};
    std::vector<i32> perm = {2, 0, 1};
    auto r = infer_transpose(A, make_span(perm));
    ASSERT_TRUE(r.ok);
    ASSERT_EQ(r.shape.rank(), 3u);
    EXPECT_EQ(r.shape[0]->value, 32);
    EXPECT_EQ(r.shape[1]->value, 8);
    EXPECT_EQ(r.shape[2]->value, 16);
}

TEST(Shape, InferReshapeInferDyn) {
    Shape A{DimExpr::make_constant(8), DimExpr::make_constant(16)};
    Shape target{DimExpr::make_constant(4),
                 DimExpr::make_constant(-1),
                 DimExpr::make_constant(4)};
    auto r = infer_reshape(A, target);
    ASSERT_TRUE(r.ok);
    ASSERT_EQ(r.shape.rank(), 3u);
    EXPECT_EQ(r.shape[0]->value, 4);
    EXPECT_EQ(r.shape[1]->value, 8);
    EXPECT_EQ(r.shape[2]->value, 4);
}

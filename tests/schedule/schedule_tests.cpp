// tests/schedule_tests.cpp - schedule IR tests
#include "cg/schedule/domain.hpp"
#include "cg/schedule/schedule.hpp"

#include "cg/test/gtest_compat.hpp"

using namespace cg;

TEST(Schedule, IterationDomainConstantExtent) {
    IterationDomain d(DimExpr::make_constant(0),
                      DimExpr::make_constant(64));
    i64 ext;
    ASSERT_TRUE(d.is_constant_extent(&ext));
    EXPECT_EQ(ext, 64);
}

TEST(Schedule, IterationDomainWithStep) {
    IterationDomain d(DimExpr::make_constant(0),
                      DimExpr::make_constant(100),
                      DimExpr::make_constant(8));
    i64 ext;
    ASSERT_TRUE(d.is_constant_extent(&ext));
    EXPECT_EQ(ext, 13); // ceil(100/8) = 13
}

TEST(Schedule, IterationDomainSymbolicExtent) {
    auto N = DimExpr::make_symbol(1, "N");
    IterationDomain d(DimExpr::make_constant(0), N);
    i64 ext;
    EXPECT_FALSE(d.is_constant_extent(&ext));
}

TEST(Schedule, TransformString) {
    Transform t{TransformKind::Tile, "m", 64, 0, "", MemorySpace::Generic};
    EXPECT_EQ(t.to_string().substr(0, 4), "tile");
}

TEST(Schedule, HashAndEquality) {
    Schedule a, b;
    a.add({TransformKind::Tile, "m", 64, 0, "", MemorySpace::Generic});
    b.add({TransformKind::Tile, "m", 64, 0, "", MemorySpace::Generic});
    EXPECT_EQ(a.hash(), b.hash());
    EXPECT_TRUE(a == b);

    Schedule c;
    c.add({TransformKind::Tile, "m", 128, 0, "", MemorySpace::Generic});
    EXPECT_NE(a.hash(), c.hash());
    EXPECT_FALSE(a == c);
}

// Minimal GoogleTest-compatible test framework.
//
// We can't always FetchContent GoogleTest at configure time (no network in
// CI sandboxes, air-gapped builds, etc.). This header provides a minimal but
// faithful subset of the GoogleTest API so that the project's own tests can
// build and run without an external dependency.
//
// Tests written against this header are *also* compatible with real
// GoogleTest: if -DCG_USE_SYSTEM_GTEST=ON is set and GoogleTest is found via
// find_package, we link against it instead and skip this header.
#pragma once

#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cg_test {

struct TestCase {
    const char* suite;
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* suite, const char* name, void(*fn)()) {
        registry().push_back({suite, name, fn});
    }
};

inline int run_all() {
    int passed = 0, failed = 0;
    for (auto& tc : registry()) {
        try {
            tc.fn();
            std::cout << "[  OK  ] " << tc.suite << "." << tc.name << "\n";
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "[ FAIL ] " << tc.suite << "." << tc.name
                      << ": " << e.what() << "\n";
            ++failed;
        } catch (...) {
            std::cout << "[ FAIL ] " << tc.suite << "." << tc.name
                      << ": unknown exception\n";
            ++failed;
        }
    }
    std::cout << "\n" << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}

struct Failure : std::runtime_error {
    explicit Failure(const std::string& msg) : std::runtime_error(msg) {}
};

template <typename A, typename B>
void assert_eq(const char* file, int line, const char* a_expr, const char* b_expr,
               const A& a, const B& b) {
    if (!(a == b)) {
        std::ostringstream os;
        os << file << ":" << line << ": EXPECT_EQ(" << a_expr << ", " << b_expr
           << ") failed";
        throw Failure(os.str());
    }
}

template <typename A, typename B>
void assert_ne(const char* file, int line, const char* a_expr, const char* b_expr,
               const A& a, const B& b) {
    if (a == b) {
        std::ostringstream os;
        os << file << ":" << line << ": EXPECT_NE(" << a_expr << ", " << b_expr
           << ") failed";
        throw Failure(os.str());
    }
}

inline void assert_true(const char* file, int line, const char* expr, bool v) {
    if (!v) {
        std::ostringstream os;
        os << file << ":" << line << ": EXPECT_TRUE(" << expr << ") failed";
        throw Failure(os.str());
    }
}

inline void assert_false(const char* file, int line, const char* expr, bool v) {
    if (v) {
        std::ostringstream os;
        os << file << ":" << line << ": EXPECT_FALSE(" << expr << ") failed";
        throw Failure(os.str());
    }
}

template <typename A, typename B>
void assert_lt(const char* file, int line, const char* a_expr, const char* b_expr,
               const A& a, const B& b) {
    if (!(a < b)) {
        std::ostringstream os;
        os << file << ":" << line << ": EXPECT_LT(" << a_expr << ", " << b_expr
           << ") failed";
        throw Failure(os.str());
    }
}

template <typename A, typename B>
void assert_le(const char* file, int line, const char* a_expr, const char* b_expr,
               const A& a, const B& b) {
    if (!(a <= b)) {
        std::ostringstream os;
        os << file << ":" << line << ": EXPECT_LE(" << a_expr << ", " << b_expr
           << ") failed";
        throw Failure(os.str());
    }
}

template <typename A, typename B>
void assert_gt(const char* file, int line, const char* a_expr, const char* b_expr,
               const A& a, const B& b) {
    if (!(a > b)) {
        std::ostringstream os;
        os << file << ":" << line << ": EXPECT_GT(" << a_expr << ", " << b_expr
           << ") failed";
        throw Failure(os.str());
    }
}

template <typename A, typename B>
void assert_ge(const char* file, int line, const char* a_expr, const char* b_expr,
               const A& a, const B& b) {
    if (!(a >= b)) {
        std::ostringstream os;
        os << file << ":" << line << ": EXPECT_GE(" << a_expr << ", " << b_expr
           << ") failed";
        throw Failure(os.str());
    }
}

} // namespace cg_test

#define CG_TEST_STRINGIFY_INNER(x) #x
#define CG_TEST_STRINGIFY(x) CG_TEST_STRINGIFY_INNER(x)

#define TEST(suite, name)                                                    \
    static void suite##_##name##_fn();                                       \
    static ::cg_test::Registrar suite##_##name##_reg(                        \
        #suite, #name, &suite##_##name##_fn);                                \
    static void suite##_##name##_fn()

#define EXPECT_EQ(a, b) ::cg_test::assert_eq(__FILE__, __LINE__, #a, #b, (a), (b))
#define EXPECT_NE(a, b) ::cg_test::assert_ne(__FILE__, __LINE__, #a, #b, (a), (b))
#define EXPECT_TRUE(x)  ::cg_test::assert_true(__FILE__, __LINE__, #x, (x))
#define EXPECT_FALSE(x) ::cg_test::assert_false(__FILE__, __LINE__, #x, (x))
#define EXPECT_LT(a, b) ::cg_test::assert_lt(__FILE__, __LINE__, #a, #b, (a), (b))
#define EXPECT_LE(a, b) ::cg_test::assert_le(__FILE__, __LINE__, #a, #b, (a), (b))
#define EXPECT_GT(a, b) ::cg_test::assert_gt(__FILE__, __LINE__, #a, #b, (a), (b))
#define EXPECT_GE(a, b) ::cg_test::assert_ge(__FILE__, __LINE__, #a, #b, (a), (b))

#define ASSERT_EQ(a, b) EXPECT_EQ(a, b)
#define ASSERT_NE(a, b) EXPECT_NE(a, b)
#define ASSERT_TRUE(x)  EXPECT_TRUE(x)
#define ASSERT_FALSE(x) EXPECT_FALSE(x)
#define ASSERT_LT(a, b) EXPECT_LT(a, b)
#define ASSERT_LE(a, b) EXPECT_LE(a, b)
#define ASSERT_GT(a, b) EXPECT_GT(a, b)
#define ASSERT_GE(a, b) EXPECT_GE(a, b)

#define CG_TEST_MAIN()                                                       \
    int main(int argc, char** argv) {                                        \
        (void)argc; (void)argv;                                              \
        return ::cg_test::run_all();                                         \
    }

cmake_minimum_required(VERSION 3.20)

project(cantors_gift
    VERSION 0.1.0
    DESCRIPTION "Multi-level SSA-based tensor compiler"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CG_BUILD_TESTS        "Build unit and integration tests"             ON)
option(CG_BUILD_BENCHMARKS   "Build benchmarks"                             OFF)
option(CG_BUILD_TOOLS        "Build CLI tools (compiler/profiler/tuner)"    OFF)
option(CG_ENABLE_LTO         "Enable link-time optimization"                OFF)
option(CG_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"            OFF)
option(CG_FETCH_GOOGLETEST   "Fetch GoogleTest via FetchContent (needs net)" OFF)
option(CG_USE_SYSTEM_GTEST   "Use system GoogleTest via find_package"       OFF)

# ---------------------------------------------------------------------------
# Compiler / linker flags
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Position-independent code everywhere (so we can be embedded into JITs).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CG_WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Wredundant-move
    -Wno-unused-parameter
    -Wno-sign-compare)

if(CG_WARNINGS_AS_ERRORS)
    list(APPEND CG_WARNING_FLAGS -Werror)
endif()

add_library(cg_warnings INTERFACE)
target_compile_options(cg_warnings INTERFACE ${CG_WARNING_FLAGS})
target_include_directories(cg_warnings INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(CG_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Disallow in-source builds.
# ---------------------------------------------------------------------------
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR
        "In-source builds are not allowed. Please create a separate build directory.")
endif()

# ---------------------------------------------------------------------------
# Export compile commands for tooling.
# ---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# The main library.
# ---------------------------------------------------------------------------
add_library(cantors_gift STATIC)

file(GLOB_RECURSE CG_HEADERS
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    "${CMAKE_CURRENT_SOURCE_DIR}/include/cg/*.hpp")

file(GLOB_RECURSE CG_SOURCES
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES false
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

target_sources(cantors_gift
    PRIVATE ${CG_SOURCES}
    PUBLIC FILE_SET HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
    FILES ${CG_HEADERS})

target_include_directories(cantors_gift
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_features(cantors_gift PUBLIC cxx_std_20)
target_link_libraries(cantors_gift PUBLIC cg_warnings)

# Threads.
find_package(Threads REQUIRED)
target_link_libraries(cantors_gift PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CG_BUILD_TESTS)
    enable_testing()

    # Test library: a minimal GoogleTest-compatible framework so that tests
    # build & run without any external network or system dependency. If
    # CG_FETCH_GOOGLETEST or CG_USE_SYSTEM_GTEST is ON, real GoogleTest is
    # used instead.
    set(CG_USING_COMPAT_GTEST OFF)
    if(CG_FETCH_GOOGLETEST)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
            URL_HASH SHA256=8ad598c73ad796e0d828070088ece745c62c3f153f96adc9b89bf6d3b62ffd4d
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    elseif(CG_USE_SYSTEM_GTEST)
        find_package(GTest REQUIRED)
    else()
        set(CG_USING_COMPAT_GTEST ON)
    endif()

    file(GLOB_RECURSE CG_TEST_SOURCES
        CONFIGURE_DEPENDS
        LIST_DIRECTORIES false
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

    # The compat test framework's main() lives in tests/test_main.cpp.
    # We compile it into an OBJECT library so every test executable can link
    # against it without re-compiling.
    if(CG_USING_COMPAT_GTEST)
        add_library(cg_test_main OBJECT tests/test_main.cpp)
        target_link_libraries(cg_test_main PUBLIC cantors_gift)
    endif()

    foreach(test_source ${CG_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        if(test_name STREQUAL "test_main")
            continue()
        endif()
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE cantors_gift)
        if(CG_USING_COMPAT_GTEST)
            target_link_libraries(${test_name} PRIVATE cg_test_main)
        else()
            target_link_libraries(${test_name} PRIVATE GTest::gtest_main)
        endif()
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
if(CG_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS cantors_gift cg_warnings
    EXPORT cantors_giftTargets
    FILE_SET HEADERS
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(EXPORT cantors_giftTargets
    FILE cantors_giftTargets.cmake
    NAMESPACE cg::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cantors_gift)

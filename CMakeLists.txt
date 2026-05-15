cmake_minimum_required(VERSION 3.18)
project(vsim LANGUAGES CXX VERSION ${SKBUILD_PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate Python (scikit-build-core pre-finds it; fall back to find_package for local builds)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind cmake config via the Python module
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Ensure stim sources exist locally. They are intentionally not tracked in git.
set(STIM_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor")
if(NOT EXISTS "${STIM_VENDOR_DIR}/stim.h" OR NOT EXISTS "${STIM_VENDOR_DIR}/stim")
    message(STATUS "Stim sources missing in vendor/. Fetching from PyPI...")
    execute_process(
        COMMAND
            "${Python_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/fetch_stim_header.py"
            "--version" "1.15.0"
            "--output-dir" "${STIM_VENDOR_DIR}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE STIM_FETCH_RESULT
    )
    if(NOT STIM_FETCH_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to fetch stim sources (exit ${STIM_FETCH_RESULT})")
    endif()
endif()

# ---------------------------------------------------------------------------
# Collect stim source files (exclude tests, perfs, pybind, top-level main.cc,
# and the march-dispatcher pybind translation unit).
# ---------------------------------------------------------------------------
file(GLOB_RECURSE STIM_SRCS "${STIM_VENDOR_DIR}/stim/*.cc")
list(FILTER STIM_SRCS EXCLUDE REGEX ".*\\.test\\.cc$")
list(FILTER STIM_SRCS EXCLUDE REGEX ".*\\.perf\\.cc$")
list(FILTER STIM_SRCS EXCLUDE REGEX ".*\\.pybind\\.cc$")
list(FILTER STIM_SRCS EXCLUDE REGEX ".*/main\\.cc$")
list(FILTER STIM_SRCS EXCLUDE REGEX ".*/march\\..*\\.cc$")

# Build stim as a static library so it can be linked into the extension.
add_library(stim_lib STATIC ${STIM_SRCS})
target_include_directories(stim_lib PUBLIC "${STIM_VENDOR_DIR}")
target_compile_options(stim_lib PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -fno-strict-aliasing
        -O3
        -fPIC
    >
)

# ---------------------------------------------------------------------------
# Build the nanobind Python extension
# ---------------------------------------------------------------------------
nanobind_add_module(_fast src/vsim/_fast.cpp)
target_link_libraries(_fast PRIVATE stim_lib)
target_include_directories(_fast PRIVATE "${STIM_VENDOR_DIR}")
target_compile_options(_fast PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -fno-strict-aliasing
        -O3
    >
)

# scikit-build-core installs to the package directory inferred from the
# project name; explicitly install under vsim/ to match the Python import.
install(TARGETS _fast LIBRARY DESTINATION vsim)

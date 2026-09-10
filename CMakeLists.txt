cmake_minimum_required(VERSION 3.24)
project(helix LANGUAGES C CXX)

if(APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Host-only by default: both build command-line executables, which cannot be
# launched on an iOS device and would only fail at link time when cross
# compiling. The library itself is what an app links.
if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(HELIX_HOST_TOOLS_DEFAULT OFF)
else()
    set(HELIX_HOST_TOOLS_DEFAULT ON)
endif()

option(HELIX_BUILD_TESTS "Build the HELIX test suite" ${HELIX_HOST_TOOLS_DEFAULT})
option(HELIX_BUILD_BENCH "Build the HELIX benchmarks"  ${HELIX_HOST_TOOLS_DEFAULT})

add_compile_options($<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)

# ---------------------------------------------------------------------------
# helix_ref -- CPU reference. No Metal dependency, so it builds and tests
# anywhere and can never be blamed for a GPU problem.
# ---------------------------------------------------------------------------
add_library(helix_ref STATIC
    src/reference/scan_ref.cpp
    src/reference/scan_chunked_ref.cpp
    src/runtime/desc.cpp
)
target_include_directories(helix_ref PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

include(cmake/HelixMetal.cmake)
include(cmake/HelixEmbed.cmake)

# ---------------------------------------------------------------------------
# Embed both metallibs into the static library, so a build tree is not part of
# the deliverable. On by default; HELIX_EMBED_METALLIB=OFF falls back to loading
# from disk, which is convenient when iterating on shaders.
# ---------------------------------------------------------------------------
option(HELIX_EMBED_METALLIB "Embed metallibs into libhelix.a" ON)

set(HELIX_EMBED_SOURCES "")
set(HELIX_EMBED_TARGETS "")
if(HELIX_METAL_AVAILABLE AND HELIX_METALLIB AND HELIX_EMBED_METALLIB)
    set(HELIX_EMBED_DIR "${CMAKE_BINARY_DIR}/embed")
    file(MAKE_DIRECTORY "${HELIX_EMBED_DIR}")

    helix_embed_metallib(helix_embed_base "${HELIX_METALLIB}"
                         helix_metallib_data "${HELIX_EMBED_DIR}/helix_metallib_data.c")
    list(APPEND HELIX_EMBED_SOURCES "${HELIX_EMBED_DIR}/helix_metallib_data.c")
    list(APPEND HELIX_EMBED_TARGETS helix_embed_base)

    if(HELIX_MPP_AVAILABLE)
        helix_embed_metallib(helix_embed_mpp "${HELIX_MPP_METALLIB}"
                             helix_mpp_metallib_data "${HELIX_EMBED_DIR}/helix_mpp_metallib_data.c")
        list(APPEND HELIX_EMBED_SOURCES "${HELIX_EMBED_DIR}/helix_mpp_metallib_data.c")
        list(APPEND HELIX_EMBED_TARGETS helix_embed_mpp)
    endif()
endif()

# ---------------------------------------------------------------------------
# helix -- the Metal runtime. Only built when the Metal toolchain is present;
# the reference library and its tests never depend on it.
# ---------------------------------------------------------------------------
if(HELIX_METAL_AVAILABLE AND HELIX_METALLIB)
    add_library(helix STATIC
        src/runtime/device.mm
        src/runtime/library.mm
        src/runtime/caps.mm
        src/runtime/dispatch.mm
        src/runtime/desc.cpp
        ${HELIX_EMBED_SOURCES}
    )
    if(HELIX_EMBED_SOURCES)
        target_compile_definitions(helix PRIVATE HELIX_HAVE_EMBEDDED_METALLIB)
        add_dependencies(helix ${HELIX_EMBED_TARGETS})
        if(HELIX_MPP_AVAILABLE)
            target_compile_definitions(helix PRIVATE HELIX_HAVE_EMBEDDED_MPP_METALLIB)
        endif()
    endif()
    target_include_directories(helix PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(helix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(helix PRIVATE HELIX_METALLIB_PATH="${HELIX_METALLIB}")
    if(HELIX_MPP_AVAILABLE)
        target_compile_definitions(helix PRIVATE HELIX_MPP_METALLIB_PATH="${HELIX_MPP_METALLIB}")
        add_dependencies(helix helix_mpp_metallib)
    endif()
    target_link_libraries(helix PUBLIC
        "-framework Metal" "-framework Foundation")
    add_dependencies(helix helix_metallib)
endif()

if(HELIX_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(HELIX_BUILD_BENCH AND TARGET helix)
    add_executable(bench_scan bench/bench_scan.mm)
    target_link_libraries(bench_scan PRIVATE helix helix_ref)
    target_include_directories(bench_scan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

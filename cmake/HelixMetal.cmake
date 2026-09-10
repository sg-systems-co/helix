# HelixMetal.cmake -- .metal -> .air -> helix.metallib
#
# Xcode 26 ships `metal` as a separately downloadable component. Probe for it
# at configure time by actually compiling a trivial shader; if it is missing,
# degrade loudly rather than failing the whole configure, so the CPU reference
# and its tests still build.

set(HELIX_METAL_AVAILABLE OFF)

if(NOT APPLE)
    return()
endif()

# ---------------------------------------------------------------------------
# Which SDK the shaders are compiled against.
#
# A metallib records the platform it was built for, so a macOS metallib will
# not load on an iOS device -- newLibraryWithData just fails at runtime, well
# after the build looked like it succeeded. Derive the SDK from the target
# platform rather than hardcoding `macosx`, and let a caller override it for
# the simulator (which needs `iphonesimulator`, a third, distinct platform).
# ---------------------------------------------------------------------------
if(NOT HELIX_METAL_SDK)
    if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
        # CMake sets CMAKE_OSX_SYSROOT to the SDK path or name it resolved; the
        # simulator and the device are different Metal platforms, so pick the
        # matching one instead of assuming a physical device.
        if(CMAKE_OSX_SYSROOT MATCHES "[Ss]imulator")
            set(HELIX_METAL_SDK "iphonesimulator")
        else()
            set(HELIX_METAL_SDK "iphoneos")
        endif()
    else()
        set(HELIX_METAL_SDK "macosx")
    endif()
endif()
message(STATUS "HELIX: Metal SDK = ${HELIX_METAL_SDK}")

set(HELIX_METAL_PROBE "${CMAKE_BINARY_DIR}/helix_metal_probe.metal")
file(WRITE "${HELIX_METAL_PROBE}"
     "#include <metal_stdlib>\nkernel void probe(uint t [[thread_position_in_grid]]) {}\n")

execute_process(
    COMMAND xcrun -sdk ${HELIX_METAL_SDK} metal -c "${HELIX_METAL_PROBE}"
                  -o "${CMAKE_BINARY_DIR}/helix_metal_probe.air"
    RESULT_VARIABLE HELIX_METAL_PROBE_RESULT
    OUTPUT_QUIET ERROR_VARIABLE HELIX_METAL_PROBE_ERR
)

if(NOT HELIX_METAL_PROBE_RESULT EQUAL 0)
    message(WARNING
        "Metal Toolchain not available -- skipping metallib build.\n"
        "  Install it with:  xcodebuild -downloadComponent MetalToolchain\n"
        "  metal said: ${HELIX_METAL_PROBE_ERR}")
    return()
endif()

# Two metallibs, deliberately.
#
# The MPP tensor-ops kernels need -std=metal4.0 and only run on Apple10 (M5).
# Keeping them in a separate library means a pre-M5 device never has to load
# them: the runtime opens helix_mpp.metallib only after checking the GPU family,
# and falls back to the portable library if it is missing or fails to load.
file(GLOB HELIX_METAL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.metal")
file(GLOB HELIX_MPP_SOURCES   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/mpp/*.metal")
if(NOT HELIX_METAL_SOURCES)
    message(STATUS "HELIX: Metal toolchain OK, no kernels yet.")
    set(HELIX_METAL_AVAILABLE ON)
    return()
endif()

set(HELIX_METALLIB "${CMAKE_BINARY_DIR}/helix.metallib")
set(HELIX_AIR_DIR  "${CMAKE_BINARY_DIR}/air")
file(MAKE_DIRECTORY "${HELIX_AIR_DIR}")

# -gline-tables-only + -frecord-sources let the Xcode shader profiler attribute
# cost per source line, which is the whole point of having a profiling setup.
set(HELIX_METAL_FLAGS -std=metal3.2 -O3 -ffast-math)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND HELIX_METAL_FLAGS -gline-tables-only -frecord-sources)
endif()

set(HELIX_AIR_FILES "")
foreach(metal_src ${HELIX_METAL_SOURCES})
    get_filename_component(metal_name "${metal_src}" NAME_WE)
    set(air_file "${HELIX_AIR_DIR}/${metal_name}.air")
    set(dep_file "${HELIX_AIR_DIR}/${metal_name}.d")
    # -MD/-MF plus DEPFILE is what makes an edit to helix_common.h or
    # helix_params.h actually rebuild the metallib. Listing only the .metal
    # source here produces a silently stale metallib on every header change --
    # a genuinely expensive way to lose an afternoon.
    add_custom_command(
        OUTPUT "${air_file}"
        COMMAND xcrun -sdk ${HELIX_METAL_SDK} metal ${HELIX_METAL_FLAGS}
                -I "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
                -MD -MF "${dep_file}"
                -c "${metal_src}" -o "${air_file}"
        DEPENDS "${metal_src}"
        DEPFILE "${dep_file}"
        COMMENT "Compiling Metal shader ${metal_name}.metal"
        VERBATIM)
    list(APPEND HELIX_AIR_FILES "${air_file}")
endforeach()

# Modern Xcode ships only the `metal` driver; it links .air -> .metallib based
# on the output extension.
add_custom_command(
    OUTPUT "${HELIX_METALLIB}"
    COMMAND xcrun -sdk ${HELIX_METAL_SDK} metal ${HELIX_AIR_FILES} -o "${HELIX_METALLIB}"
    DEPENDS ${HELIX_AIR_FILES}
    COMMENT "Linking helix.metallib"
    VERBATIM)

add_custom_target(helix_metallib ALL DEPENDS "${HELIX_METALLIB}")

set(HELIX_METAL_AVAILABLE ON)

# ---------------------------------------------------------------------------
# helix_mpp.metallib -- MetalPerformancePrimitives tensor ops, metal4.0, M5 only
# ---------------------------------------------------------------------------
if(HELIX_MPP_SOURCES)
    set(HELIX_MPP_FLAGS -std=metal4.0 -O3 -ffast-math)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND HELIX_MPP_FLAGS -gline-tables-only -frecord-sources)
    endif()

    # Probe metal4.0 + the MPP headers before committing to the target, so an
    # older toolchain degrades to the simdgroup_matrix path instead of failing
    # the whole configure.
    set(HELIX_MPP_PROBE "${CMAKE_BINARY_DIR}/helix_mpp_probe.metal")
    file(WRITE "${HELIX_MPP_PROBE}"
         "#include <metal_stdlib>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
         "kernel void probe(uint t [[thread_position_in_grid]]) {}\n")
    execute_process(
        COMMAND xcrun -sdk ${HELIX_METAL_SDK} metal -std=metal4.0 -c "${HELIX_MPP_PROBE}"
                      -o "${CMAKE_BINARY_DIR}/helix_mpp_probe.air"
        RESULT_VARIABLE HELIX_MPP_PROBE_RESULT OUTPUT_QUIET ERROR_QUIET)

    if(HELIX_MPP_PROBE_RESULT EQUAL 0)
        set(HELIX_MPP_METALLIB "${CMAKE_BINARY_DIR}/helix_mpp.metallib")
        set(HELIX_MPP_AIR_FILES "")
        foreach(mpp_src ${HELIX_MPP_SOURCES})
            get_filename_component(mpp_name "${mpp_src}" NAME_WE)
            set(mpp_air "${HELIX_AIR_DIR}/${mpp_name}.air")
            set(mpp_dep "${HELIX_AIR_DIR}/${mpp_name}.d")
            add_custom_command(
                OUTPUT "${mpp_air}"
                COMMAND xcrun -sdk ${HELIX_METAL_SDK} metal ${HELIX_MPP_FLAGS}
                        -I "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
                        -MD -MF "${mpp_dep}"
                        -c "${mpp_src}" -o "${mpp_air}"
                DEPENDS "${mpp_src}"
                DEPFILE "${mpp_dep}"
                COMMENT "Compiling MPP shader ${mpp_name}.metal (metal4.0)"
                VERBATIM)
            list(APPEND HELIX_MPP_AIR_FILES "${mpp_air}")
        endforeach()

        add_custom_command(
            OUTPUT "${HELIX_MPP_METALLIB}"
            COMMAND xcrun -sdk ${HELIX_METAL_SDK} metal ${HELIX_MPP_AIR_FILES} -o "${HELIX_MPP_METALLIB}"
            DEPENDS ${HELIX_MPP_AIR_FILES}
            COMMENT "Linking helix_mpp.metallib"
            VERBATIM)
        add_custom_target(helix_mpp_metallib ALL DEPENDS "${HELIX_MPP_METALLIB}")
        set(HELIX_MPP_AVAILABLE ON)
        message(STATUS "HELIX: MPP tensor-ops path enabled (metal4.0).")
    else()
        message(STATUS "HELIX: metal4.0 / MPP headers unavailable -- simdgroup_matrix only.")
    endif()
endif()


#!/usr/bin/env bash
#
# build-ios.sh -- cross-compile HELIX + the HELIX-accelerated llama.cpp for iOS.
#
# Produces a directory of static libraries and headers ready to be consumed by
# the local Swift package in ../helix-chat-ui:
#
#     dist/ios-<platform>/
#       lib/     libhelix.a  libllama.a  libggml*.a
#       include/ llama.h  ggml*.h  helix/helix.h
#
# Static, deliberately. A dylib built into a CMake build tree cannot ship inside
# an .app without being embedded and re-signed, and the macOS package's
# -rpath-to-build-directory trick has no iOS equivalent. Static libraries make
# the app bundle self-contained.
#
# Usage:
#   scripts/build-ios.sh                      # device, arm64, Release
#   scripts/build-ios.sh --platform simulator # simulator (see note below)
#   scripts/build-ios.sh --clean
#
set -euo pipefail

HELIX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_ROOT="${HELIX_ROOT}/third_party/llama.cpp"

PLATFORM="device"
BUILD_TYPE="Release"
DEPLOYMENT_TARGET="17.0"
CLEAN=0
JOBS="$(sysctl -n hw.ncpu)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)          PLATFORM="$2"; shift 2 ;;
        --build-type)        BUILD_TYPE="$2"; shift 2 ;;
        --deployment-target) DEPLOYMENT_TARGET="$2"; shift 2 ;;
        --jobs|-j)           JOBS="$2"; shift 2 ;;
        --clean)             CLEAN=1; shift ;;
        -h|--help)           sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$PLATFORM" in
    device)    SYSROOT="iphoneos" ;;
    simulator) SYSROOT="iphonesimulator" ;;
    *) echo "--platform must be 'device' or 'simulator'" >&2; exit 2 ;;
esac

HELIX_BUILD="${HELIX_ROOT}/build-ios-${PLATFORM}"
LLAMA_BUILD="${LLAMA_ROOT}/build-ios-${PLATFORM}"
DIST="${HELIX_ROOT}/dist/ios-${PLATFORM}"

if [[ "$CLEAN" == "1" ]]; then
    echo ">> removing ${HELIX_BUILD} ${LLAMA_BUILD} ${DIST}"
    rm -rf "$HELIX_BUILD" "$LLAMA_BUILD" "$DIST"
fi

# --- preflight ------------------------------------------------------------
# The Metal toolchain is a separately downloadable Xcode component, and its
# absence otherwise shows up as a metallib that silently never gets built.
if ! xcrun -sdk "$SYSROOT" metal --version >/dev/null 2>&1; then
    echo "error: no Metal toolchain for SDK '${SYSROOT}'." >&2
    echo "       install it with:  xcodebuild -downloadComponent MetalToolchain" >&2
    exit 1
fi
command -v cmake >/dev/null || { echo "error: cmake not found" >&2; exit 1; }

GEN=(-G "Unix Makefiles")
command -v ninja >/dev/null && GEN=(-G Ninja)

# Flags shared by both projects. GGML_NATIVE=OFF matters: the host is an M5 and
# -march=native would emit instructions the target does not have.
IOS_FLAGS=(
    -DCMAKE_SYSTEM_NAME=iOS
    -DCMAKE_OSX_ARCHITECTURES=arm64
    -DCMAKE_OSX_SYSROOT="$SYSROOT"
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DBUILD_SHARED_LIBS=OFF
)

echo "==> platform=${PLATFORM} sysroot=${SYSROOT} min=${DEPLOYMENT_TARGET} type=${BUILD_TYPE}"

# --- 1. libhelix.a --------------------------------------------------------
# Built first: the ggml interception hunk links ${HELIX_BUILD}/libhelix.a
# directly, so it has to exist before llama.cpp configures.
echo "==> configuring HELIX"
cmake -S "$HELIX_ROOT" -B "$HELIX_BUILD" "${GEN[@]}" "${IOS_FLAGS[@]}" \
    -DHELIX_BUILD_TESTS=OFF \
    -DHELIX_BUILD_BENCH=OFF

echo "==> building HELIX"
cmake --build "$HELIX_BUILD" -j "$JOBS"

[[ -f "${HELIX_BUILD}/libhelix.a" ]] || {
    echo "error: libhelix.a was not produced -- the Metal toolchain probe most" >&2
    echo "       likely failed, which downgrades the build to the CPU reference." >&2
    exit 1
}

# --- 2. llama.cpp + ggml, with the HELIX SSM_SCAN interception -------------
echo "==> configuring llama.cpp"
cmake -S "$LLAMA_ROOT" -B "$LLAMA_BUILD" "${GEN[@]}" "${IOS_FLAGS[@]}" \
    -DGGML_NATIVE=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=Apple \
    -DGGML_OPENMP=OFF \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_CURL=OFF \
    -DGGML_METAL_HELIX=ON \
    -DHELIX_ROOT="$HELIX_ROOT" \
    -DHELIX_BUILD="$HELIX_BUILD"

# Named targets rather than `all`: the unified `llama-app` binary is a host
# tool that pulls in the whole common/ layer, and it is not something an iOS
# app links against anyway.
echo "==> building llama.cpp"
cmake --build "$LLAMA_BUILD" -j "$JOBS" \
    --target llama ggml ggml-base ggml-cpu ggml-metal ggml-blas

# --- 3. stage into dist/ --------------------------------------------------
echo "==> staging ${DIST}"
rm -rf "$DIST"
mkdir -p "$DIST/lib" "$DIST/include/helix"

find "$LLAMA_BUILD" -name 'lib*.a' -exec cp {} "$DIST/lib/" \;
cp "${HELIX_BUILD}/libhelix.a" "$DIST/lib/"

cp "${LLAMA_ROOT}/include/llama.h"           "$DIST/include/"
cp "${LLAMA_ROOT}/ggml/include/"*.h          "$DIST/include/"
cp "${HELIX_ROOT}/include/helix/helix.h"     "$DIST/include/helix/"

# One merged archive alongside the individual ones.
#
# Static libraries are order-sensitive at link time and these are mutually
# recursive: libggml.a calls ggml_backend_metal_reg() in libggml-metal.a, which
# calls into libhelix.a, and all of them call back into libggml-base.a. The
# linker resolves to a fixpoint *within* a single archive but only makes one
# pass over a list of them, so any hand-picked order is a latent
# undefined-symbol bug. Merging removes the question entirely, and leaves the
# Swift package with a single -l flag.
libtool -static -no_warning_for_no_symbols \
    -o "$DIST/lib/libHelixLlama.a" "$DIST"/lib/lib*.a 2>/dev/null

# --- 4. verify what was actually produced ---------------------------------
# A cross-compile that silently produced host-architecture objects is the
# failure this catches: it links fine on the Mac and fails only in Xcode.
echo "==> verifying"

# LC_BUILD_VERSION platform codes. A library built for the wrong platform links
# on the Mac and fails only once Xcode tries to embed it, so check the value
# recorded in the object rather than trusting the flags we passed.
case "$PLATFORM" in
    device)    want_plat=2 ;;   # PLATFORM_IOS
    simulator) want_plat=7 ;;   # PLATFORM_IOSSIMULATOR
esac
plat_name() {
    case "$1" in
        1) echo "macOS" ;;  2) echo "iOS" ;;  7) echo "iOS-simulator" ;;
        "") echo "unknown" ;;  *) echo "code-$1" ;;
    esac
}

fail=0
for lib in "$DIST"/lib/*.a; do
    [[ "$(basename "$lib")" == "libHelixLlama.a" ]] && continue
    # No `exit` inside awk: closing the pipe early sends SIGPIPE to otool, and
    # under `set -o pipefail` that aborts the whole script -- which only shows
    # up on archives big enough that otool is still writing, so the small
    # libraries pass and libllama.a silently kills the loop.
    plat="$(otool -l "$lib" 2>/dev/null |
            awk '/LC_BUILD_VERSION/{f=1} f && /platform/{if (!p) p=$2; f=0} END{print p}')"
    arch="$(lipo -archs "$lib" 2>/dev/null || echo '?')"
    printf '    %-24s arch=%-8s platform=%s\n' "$(basename "$lib")" "$arch" "$(plat_name "$plat")"
    [[ "$arch" == *arm64* ]] || { echo "      ^ expected arm64" >&2; fail=1; }
    [[ "$plat" == "$want_plat" ]] || { echo "      ^ expected $(plat_name "$want_plat")" >&2; fail=1; }
done

# The metallib is embedded in libhelix.a, so confirm the archive really carries
# the shader bytes -- an empty embed is otherwise invisible until first launch.
if ! nm "$DIST/lib/libhelix.a" 2>/dev/null | grep -q helix_metallib_data; then
    echo "error: libhelix.a has no embedded metallib symbol" >&2
    fail=1
fi

[[ "$fail" == "0" ]] || { echo "==> FAILED verification" >&2; exit 1; }

echo
echo "==> done: $DIST"
echo "    point the Swift package at it with:"
echo "        export HELIX_IOS_DIST=$DIST"

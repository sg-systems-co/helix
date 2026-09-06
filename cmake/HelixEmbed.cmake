# HelixEmbed.cmake -- turn a .metallib into a C array so it ships inside the
# static library.
#
# Without this, libhelix.a carries an absolute path to the build tree and the
# library cannot be distributed. ggml solves the same problem with
# GGML_METAL_EMBED_LIBRARY; this is the equivalent, done at the metallib level
# rather than the source level because HELIX ships pre-compiled AIR.
#
# The generated header is written at build time, not configure time, so it
# tracks the metallib it came from.

# Captured at include time. Inside a function body CMAKE_CURRENT_LIST_DIR
# resolves against the caller, not this file.
set(HELIX_EMBED_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/HelixEmbedRun.cmake")

function(helix_embed_metallib TARGET_NAME METALLIB_PATH SYMBOL OUT_HEADER)
    add_custom_command(
        OUTPUT "${OUT_HEADER}"
        COMMAND ${CMAKE_COMMAND}
                -DHELIX_EMBED_IN=${METALLIB_PATH}
                -DHELIX_EMBED_OUT=${OUT_HEADER}
                -DHELIX_EMBED_SYM=${SYMBOL}
                -P "${HELIX_EMBED_SCRIPT}"
        DEPENDS "${METALLIB_PATH}"
        COMMENT "Embedding ${SYMBOL}"
        VERBATIM)
    add_custom_target(${TARGET_NAME} DEPENDS "${OUT_HEADER}")
endfunction()

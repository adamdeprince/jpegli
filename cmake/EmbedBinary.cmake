if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT OR NOT DEFINED SYMBOL)
  message(FATAL_ERROR "INPUT, OUTPUT and SYMBOL are required")
endif()
if(NOT DEFINED GUARD)
  set(GUARD JPEGLI_AMD_VULKAN_PROGRESSIVE_SPV_H_)
endif()

file(READ "${INPUT}" binary_hex HEX)
string(REGEX REPLACE "([0-9a-fA-F][0-9a-fA-F])" "0x\\1," binary_bytes
                     "${binary_hex}")
file(WRITE "${OUTPUT}"
     "// Generated from ${INPUT}; do not edit.\n"
     "#ifndef ${GUARD}\n"
     "#define ${GUARD}\n"
     "#include <cstddef>\n"
     "alignas(4) static constexpr unsigned char ${SYMBOL}[] = {\n"
     "${binary_bytes}\n};\n"
     "static constexpr size_t ${SYMBOL}Size = sizeof(${SYMBOL});\n"
     "#endif\n")

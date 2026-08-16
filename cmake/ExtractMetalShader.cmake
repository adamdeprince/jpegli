# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

if(NOT DEFINED INPUT OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "ExtractMetalShader.cmake requires INPUT and OUTPUT")
endif()

file(READ "${INPUT}" JPEGLI_APPLE_METAL_IMPLEMENTATION)
set(JPEGLI_APPLE_METAL_BEGIN
    "const char kMetalSource[] = R\"metal(\n")
set(JPEGLI_APPLE_METAL_END "\n)metal\";")

string(FIND "${JPEGLI_APPLE_METAL_IMPLEMENTATION}"
       "${JPEGLI_APPLE_METAL_BEGIN}" JPEGLI_APPLE_METAL_BEGIN_OFFSET)
if(JPEGLI_APPLE_METAL_BEGIN_OFFSET EQUAL -1)
  message(FATAL_ERROR "Metal shader begin marker was not found in ${INPUT}")
endif()
string(LENGTH "${JPEGLI_APPLE_METAL_BEGIN}"
       JPEGLI_APPLE_METAL_BEGIN_LENGTH)
math(EXPR JPEGLI_APPLE_METAL_SOURCE_OFFSET
     "${JPEGLI_APPLE_METAL_BEGIN_OFFSET} + ${JPEGLI_APPLE_METAL_BEGIN_LENGTH}")
string(SUBSTRING "${JPEGLI_APPLE_METAL_IMPLEMENTATION}"
       ${JPEGLI_APPLE_METAL_SOURCE_OFFSET} -1 JPEGLI_APPLE_METAL_TAIL)
string(FIND "${JPEGLI_APPLE_METAL_TAIL}" "${JPEGLI_APPLE_METAL_END}"
       JPEGLI_APPLE_METAL_SOURCE_LENGTH)
if(JPEGLI_APPLE_METAL_SOURCE_LENGTH EQUAL -1)
  message(FATAL_ERROR "Metal shader end marker was not found in ${INPUT}")
endif()

string(SUBSTRING "${JPEGLI_APPLE_METAL_TAIL}" 0
       ${JPEGLI_APPLE_METAL_SOURCE_LENGTH} JPEGLI_APPLE_METAL_SOURCE)
file(WRITE "${OUTPUT}" "${JPEGLI_APPLE_METAL_SOURCE}\n")

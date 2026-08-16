# Copyright (c) the JPEG XL Project Authors.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

include(jpegli_lists.cmake)

set(JPEGLI_INTERNAL_LIBS
  hwy
  Threads::Threads
  ${ATOMICS_LIBRARIES}
)

# JPEGLIB setup
set(BITS_IN_JSAMPLE 8)
set(MEM_SRCDST_SUPPORTED 1)

if(JPEGLI_LIBJPEG_LIBRARY_SOVERSION STREQUAL "62")
  set(JPEG_LIB_VERSION 62)
elseif(JPEGLI_LIBJPEG_LIBRARY_SOVERSION STREQUAL "7")
  set(JPEG_LIB_VERSION 70)
elseif(JPEGLI_LIBJPEG_LIBRARY_SOVERSION STREQUAL "8")
  set(JPEG_LIB_VERSION 80)
endif()

configure_file(
  ../third_party/libjpeg-turbo/jconfig.h.in include/jpegli/jconfig.h)
configure_file(
  ../third_party/libjpeg-turbo/jpeglib.h include/jpegli/jpeglib.h COPYONLY)
configure_file(
  ../third_party/libjpeg-turbo/jmorecfg.h include/jpegli/jmorecfg.h COPYONLY)
foreach(JPEGLI_PUBLIC_HEADER apple_metal.h common.h decode.h encode.h types.h)
  configure_file(
    jpegli/${JPEGLI_PUBLIC_HEADER}
    include/jpegli/${JPEGLI_PUBLIC_HEADER} COPYONLY)
endforeach()

if(JPEGLI_ENABLE_APPLE_METAL)
  find_library(JPEGLI_FOUNDATION_FRAMEWORK Foundation REQUIRED)
  find_library(JPEGLI_METAL_FRAMEWORK Metal REQUIRED)
  list(APPEND JPEGLI_INTERNAL_JPEGLI_SOURCES jpegli/apple_metal.mm)

  if(JPEGLI_APPLE_METAL_PRECOMPILE_SHADERS)
    set(JPEGLI_XCODE_TOOLCHAIN_HINTS)
    if(DEFINED ENV{DEVELOPER_DIR})
      list(APPEND JPEGLI_XCODE_TOOLCHAIN_HINTS
        "$ENV{DEVELOPER_DIR}/Toolchains/XcodeDefault.xctoolchain/usr/bin")
    endif()
    list(APPEND JPEGLI_XCODE_TOOLCHAIN_HINTS
      "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin")
    find_program(JPEGLI_METAL_COMPILER NAMES metal
      HINTS ${JPEGLI_XCODE_TOOLCHAIN_HINTS})
    find_program(JPEGLI_METALLIB_LINKER NAMES metallib
      HINTS ${JPEGLI_XCODE_TOOLCHAIN_HINTS})
    find_program(JPEGLI_XXD_EXECUTABLE NAMES xxd)

    if(JPEGLI_METAL_COMPILER AND JPEGLI_METALLIB_LINKER AND
       JPEGLI_XXD_EXECUTABLE)
      set(JPEGLI_APPLE_METAL_GENERATED_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/apple_metal")
      file(MAKE_DIRECTORY "${JPEGLI_APPLE_METAL_GENERATED_DIR}")
      set(JPEGLI_APPLE_METAL_SOURCE
        "${JPEGLI_APPLE_METAL_GENERATED_DIR}/jpegli_apple_metal.metal")
      set(JPEGLI_APPLE_METAL_AIR
        "${JPEGLI_APPLE_METAL_GENERATED_DIR}/jpegli_apple_metal.air")
      set(JPEGLI_APPLE_METAL_LIBRARY
        "${JPEGLI_APPLE_METAL_GENERATED_DIR}/jpegli_apple_metal.metallib")
      set(JPEGLI_APPLE_METAL_HEADER
        "${JPEGLI_APPLE_METAL_GENERATED_DIR}/jpegli_apple_metal_metallib.inc")

      add_custom_command(
        OUTPUT "${JPEGLI_APPLE_METAL_SOURCE}"
        COMMAND "${CMAKE_COMMAND}"
          "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/jpegli/apple_metal.mm"
          "-DOUTPUT=${JPEGLI_APPLE_METAL_SOURCE}"
          -P "${PROJECT_SOURCE_DIR}/cmake/ExtractMetalShader.cmake"
        DEPENDS
          "${CMAKE_CURRENT_SOURCE_DIR}/jpegli/apple_metal.mm"
          "${PROJECT_SOURCE_DIR}/cmake/ExtractMetalShader.cmake"
        VERBATIM)
      add_custom_command(
        OUTPUT "${JPEGLI_APPLE_METAL_AIR}"
        COMMAND "${JPEGLI_METAL_COMPILER}"
          -c -fmetal-math-mode=safe
          -fmetal-math-fp32-functions=precise
          "${JPEGLI_APPLE_METAL_SOURCE}"
          -o "${JPEGLI_APPLE_METAL_AIR}"
        DEPENDS "${JPEGLI_APPLE_METAL_SOURCE}"
        VERBATIM)
      add_custom_command(
        OUTPUT "${JPEGLI_APPLE_METAL_LIBRARY}"
        COMMAND "${JPEGLI_METALLIB_LINKER}"
          "${JPEGLI_APPLE_METAL_AIR}"
          -o "${JPEGLI_APPLE_METAL_LIBRARY}"
        DEPENDS "${JPEGLI_APPLE_METAL_AIR}"
        VERBATIM)
      add_custom_command(
        OUTPUT "${JPEGLI_APPLE_METAL_HEADER}"
        COMMAND "${JPEGLI_XXD_EXECUTABLE}" -i
          -n kJpegliAppleMetalLibraryBytes
          "${JPEGLI_APPLE_METAL_LIBRARY}"
          "${JPEGLI_APPLE_METAL_HEADER}"
        DEPENDS "${JPEGLI_APPLE_METAL_LIBRARY}"
        VERBATIM)
      add_custom_target(jpegli-apple-metal-shader
        DEPENDS "${JPEGLI_APPLE_METAL_HEADER}")
      message(STATUS "JPEGli: embedding precompiled Apple Metal shaders")
    else()
      message(STATUS
        "JPEGli: Xcode Metal tools unavailable; shaders will compile lazily")
    endif()
  endif()
endif()

add_library(jpegli-static STATIC "${JPEGLI_INTERNAL_JPEGLI_SOURCES}")
target_compile_options(jpegli-static PRIVATE "${JPEGLI_INTERNAL_FLAGS}")
target_compile_options(jpegli-static PUBLIC ${JPEGLI_COVERAGE_FLAGS})
set_property(TARGET jpegli-static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(jpegli-static PRIVATE
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
  "${JPEGLI_HWY_INCLUDE_DIRS}"
)
target_include_directories(jpegli-static PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/jpegli>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/jpegli>"
)
target_link_libraries(jpegli-static PUBLIC ${JPEGLI_INTERNAL_LIBS})
if(JPEGLI_ENABLE_APPLE_METAL)
  target_compile_definitions(jpegli-static PRIVATE
    JPEGLI_ENABLE_APPLE_METAL=1)
  if(TARGET jpegli-apple-metal-shader)
    add_dependencies(jpegli-static jpegli-apple-metal-shader)
    target_include_directories(jpegli-static PRIVATE
      "${JPEGLI_APPLE_METAL_GENERATED_DIR}")
    target_compile_definitions(jpegli-static PRIVATE
      JPEGLI_APPLE_METAL_PRECOMPILED_LIBRARY=1)
  endif()
  set_source_files_properties(jpegli/apple_metal.mm PROPERTIES
    COMPILE_FLAGS "-fobjc-arc")
  target_link_libraries(jpegli-static PUBLIC
    ${JPEGLI_FOUNDATION_FRAMEWORK}
    ${JPEGLI_METAL_FRAMEWORK})
endif()

install(TARGETS jpegli-static
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")

#
# Tests for jpegli-static
#

find_package(JPEG)
if(JPEG_FOUND AND BUILD_TESTING)
# TODO(eustas): merge into jpegli_tests.cmake?

add_library(jpegli_libjpeg_util-obj OBJECT
  ${JPEGLI_INTERNAL_JPEGLI_LIBJPEG_HELPER_FILES}
)
target_include_directories(jpegli_libjpeg_util-obj PRIVATE
  "${PROJECT_SOURCE_DIR}"
  "${JPEG_INCLUDE_DIRS}"
)
target_compile_options(jpegli_libjpeg_util-obj PRIVATE
  "${JPEGLI_INTERNAL_FLAGS}" "${JPEGLI_COVERAGE_FLAGS}")

# Individual test binaries:
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
foreach (TESTFILE IN LISTS JPEGLI_INTERNAL_JPEGLI_TESTS)
  # The TESTNAME is the name without the extension or directory.
  get_filename_component(TESTNAME ${TESTFILE} NAME_WE)
  add_executable(${TESTNAME} ${TESTFILE}
    $<TARGET_OBJECTS:jpegli_libjpeg_util-obj>
    ${JPEGLI_INTERNAL_JPEGLI_TESTLIB_FILES}
  )
  target_compile_options(${TESTNAME} PRIVATE
    ${JPEGLI_INTERNAL_FLAGS}
    # Add coverage flags to the test binary so code in the private headers of
    # the library is also instrumented when running tests that execute it.
    ${JPEGLI_COVERAGE_FLAGS}
  )
  target_compile_definitions(${TESTNAME} PRIVATE
    -DTEST_DATA_PATH="${JPEGLI_TEST_DATA_PATH}")
  target_include_directories(${TESTNAME} PRIVATE
    "${PROJECT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_BINARY_DIR}/include"
  )
  target_link_libraries(${TESTNAME}
    hwy
    jpegli-static
    gtest
    gtest_main
    ${JPEG_LIBRARIES}
  )
  set_target_properties(${TESTNAME} PROPERTIES LINK_FLAGS "${JPEGLI_COVERAGE_LINK_FLAGS}")
  # Output test targets in the test directory.
  set_target_properties(${TESTNAME} PROPERTIES PREFIX "tests/")
  if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_target_properties(${TESTNAME} PROPERTIES COMPILE_FLAGS "-Wno-error")
  endif ()
  # 240 seconds because some build types (e.g. coverage) can be quite slow.
  gtest_discover_tests(${TESTNAME} DISCOVERY_TIMEOUT 240)
endforeach ()
endif()

#
# Build libjpeg.so that links to libjpeg-static
#

if (JPEGLI_ENABLE_JPEGLI_LIBJPEG AND NOT APPLE AND NOT WIN32 AND NOT EMSCRIPTEN)
add_library(jpegli-libjpeg-obj OBJECT "${JPEGLI_INTERNAL_JPEGLI_WRAPPER_SOURCES}")
target_compile_options(jpegli-libjpeg-obj PRIVATE ${JPEGLI_INTERNAL_FLAGS})
target_compile_options(jpegli-libjpeg-obj PUBLIC ${JPEGLI_COVERAGE_FLAGS})
set_property(TARGET jpegli-libjpeg-obj PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(jpegli-libjpeg-obj PRIVATE
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/jpegli>"
)
target_compile_definitions(jpegli-libjpeg-obj PUBLIC
  ${JPEGLI_LIBJPEG_OBJ_COMPILE_DEFINITIONS}
)
set(JPEGLI_LIBJPEG_INTERNAL_OBJECTS $<TARGET_OBJECTS:jpegli-libjpeg-obj>)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/jpegli)
add_library(jpeg SHARED ${JPEGLI_LIBJPEG_INTERNAL_OBJECTS})
target_link_libraries(jpeg PUBLIC ${JPEGLI_COVERAGE_FLAGS})
target_link_libraries(jpeg PRIVATE jpegli-static)
set_target_properties(jpeg PROPERTIES
  VERSION ${JPEGLI_LIBJPEG_LIBRARY_VERSION}
  SOVERSION ${JPEGLI_LIBJPEG_LIBRARY_SOVERSION}
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/jpegli"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/jpegli")

# Add a jpeg.version file as a version script to tag symbols with the
# appropriate version number.
set_target_properties(jpeg PROPERTIES
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/jpegli/jpeg.version.${JPEGLI_LIBJPEG_LIBRARY_SOVERSION})
set_property(TARGET jpeg APPEND_STRING PROPERTY
  LINK_FLAGS " -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/jpegli/jpeg.version.${JPEGLI_LIBJPEG_LIBRARY_SOVERSION}")

if (JPEGLI_INSTALL_JPEGLI_LIBJPEG)
  install(TARGETS jpeg
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/jpegli/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

# This hides the default visibility symbols from static libraries bundled into
# the shared library. In particular this prevents exposing symbols from hwy
# in the shared library.
if(LINKER_SUPPORT_EXCLUDE_LIBS)
  set_property(TARGET jpeg APPEND_STRING PROPERTY
    LINK_FLAGS " ${LINKER_EXCLUDE_LIBS_FLAG}")
endif()
endif()

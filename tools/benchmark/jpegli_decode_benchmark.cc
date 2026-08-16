// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <turbojpeg.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#endif

#if defined(__APPLE__)
#include <libproc.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "lib/base/span.h"
#include "lib/base/status.h"
#include "lib/extras/butteraugli.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/jpegli.h"
#include "lib/extras/metrics.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/jpegli/apple_metal.h"
#include "lib/jpegli/decode.h"
#include "lib/jpegli/decode_profile.h"
#include "lib/jpegli/memory_manager.h"
#include "tools/cmdline.h"
#include "tools/file_io.h"
#include "tools/no_memory_manager.h"
#include "tools/ssimulacra2.h"
#include "tools/tool_version.h"

namespace jpegli_tools {
namespace {

namespace fs = std::filesystem;

constexpr size_t kOutputChannels = 4;
constexpr size_t kComparedChannels = 3;

struct Args {
  void AddCommandLineOptions(CommandLineParser* cmdline) {
    cmdline->AddPositionalOption(
        "INPUT", /*required=*/true,
        "a source image or directory of source images (archives must be "
        "extracted first)",
        &input);
    cmdline->AddOptionValue('q', "quality", "N",
                            "jpegli encode quality (default: 90)", &quality,
                            &ParseSigned);
    cmdline->AddOptionValue('\0', "chroma_subsampling", "444|440|422|420",
                            "jpegli chroma subsampling (default: 420)",
                            &chroma_subsampling, &ParseString);
    cmdline->AddOptionValue(
        'p', "progressive_level", "N",
        "jpegli progressive level, 0 is sequential (default: 0)",
        &progressive_level, &ParseSigned);
    cmdline->AddOptionValue('n', "iterations", "N",
                            "timed decode iterations per image (default: 7)",
                            &iterations, &ParseUnsigned);
    cmdline->AddOptionValue('\0', "warmups", "N",
                            "untimed warmup iterations (default: 1)", &warmups,
                            &ParseUnsigned);
    cmdline->AddOptionValue(
        '\0', "limit", "N",
        "only use the first N images after sorting; 0 means all", &limit,
        &ParseUnsigned);
    cmdline->AddOptionValue('\0', "csv", "PATH",
                            "write per-image results to this CSV file", &csv,
                            &ParseString);
    cmdline->AddOptionValue(
        '\0', "jpeg_dir", "PATH",
        "also save the jpegli-encoded benchmark inputs in this directory",
        &jpeg_dir, &ParseString);
    cmdline->AddOptionFlag(
        '\0', "perceptual_quality",
        "score decoder output against the source with SSIMULACRA2 and "
        "Butteraugli instead of timing decode",
        &perceptual_quality, &SetBooleanTrue);
    cmdline->AddOptionValue(
        '\0', "quality_levels", "LIST",
        "comma-separated jpegli qualities for --perceptual_quality "
        "(default: 50,75,90,95)",
        &quality_levels, &ParseString);
    cmdline->AddOptionValue(
        '\0', "quality_subsampling", "LIST",
        "comma-separated chroma modes for --perceptual_quality "
        "(default: 444,420)",
        &quality_subsampling, &ParseString);
    cmdline->AddOptionFlag(
        '\0', "decode_stage_profile",
        "profile jpegli decode stages instead of comparing decoder speed",
        &decode_stage_profile, &SetBooleanTrue);
    cmdline->AddOptionValue('\0', "stage_progressive_levels", "LIST",
                            "comma-separated jpegli progressive levels for "
                            "--decode_stage_profile (default: 0,2)",
                            &stage_progressive_levels, &ParseString);
    cmdline->AddOptionFlag(
        '\0', "apple_metal_benchmark",
        "benchmark CPU jpegli, Metal scanlines, and direct Metal output",
        &apple_metal_benchmark, &SetBooleanTrue);
  }

  const char* input = nullptr;
  int quality = 90;
  std::string chroma_subsampling = "420";
  int progressive_level = 0;
  size_t iterations = 7;
  size_t warmups = 1;
  size_t limit = 0;
  std::string csv;
  std::string jpeg_dir;
  bool perceptual_quality = false;
  std::string quality_levels = "50,75,90,95";
  std::string quality_subsampling = "444,420";
  bool decode_stage_profile = false;
  std::string stage_progressive_levels = "0,2";
  bool apple_metal_benchmark = false;
};

struct ImageBenchmark {
  std::string name;
  std::vector<uint8_t> jpeg;
  size_t width = 0;
  size_t height = 0;
  size_t output_size = 0;
  std::vector<double> jpegli_seconds;
  std::vector<double> turbojpeg_seconds;
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  std::vector<double> imageio_seconds;
#endif
  uint64_t jpegli_checksum = 0;
  uint64_t turbojpeg_checksum = 0;
  double turbojpeg_mean_abs_diff = 0.0;
  int turbojpeg_max_abs_diff = 0;
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  uint64_t imageio_checksum = 0;
  double imageio_mean_abs_diff = 0.0;
  int imageio_max_abs_diff = 0;
#endif
};

struct Summary {
  double median = 0.0;
  double minimum = 0.0;
  double mean = 0.0;
};

struct PerceptualScore {
  std::string decoder;
  double ssimulacra2 = 0.0;
  double butteraugli = 0.0;
};

struct QualityResult {
  std::string name;
  size_t width = 0;
  size_t height = 0;
  int quality = 0;
  std::string chroma_subsampling;
  int progressive_level = 0;
  size_t jpeg_bytes = 0;
  std::vector<PerceptualScore> scores;
};

struct QualityTask {
  fs::path path;
  int quality = 0;
  std::string chroma_subsampling;
};

enum class DecodeApiPhase : size_t {
  kCreateAndSource,
  kReadHeader,
  kStartDecompress,
  kReadScanlines,
  kFinishAndDestroy,
  kCount,
};

constexpr size_t kNumDecodeApiPhases =
    static_cast<size_t>(DecodeApiPhase::kCount);

struct DecodeStageSample {
  double total_seconds = 0.0;
  double api_seconds[kNumDecodeApiPhases] = {};
  jpegli::DecodeProfile internal;
};

struct DecodeStageBenchmark {
  ImageBenchmark image;
  int progressive_level = 0;
  std::vector<DecodeStageSample> samples;
};

enum class MetalDecodePath {
  kCpu,
  kMetalScanlines,
  kMetalDirect,
};

struct MetalDecodeSample {
  double ready_seconds = 0.0;
  double total_seconds = 0.0;
  double process_cpu_seconds = 0.0;
  uint64_t process_energy_nj = 0;
  bool process_energy_available = false;
  jpegli::DecodeProfile internal = {};
  JpegliAppleMetalStats metal = {};
  bool cold = false;
  uint64_t jpegli_memory_bytes = 0;
};

struct MetalImageBenchmark {
  ImageBenchmark image;
  std::vector<MetalDecodeSample> cpu;
  std::vector<MetalDecodeSample> metal_scanlines;
  std::vector<MetalDecodeSample> metal_direct;
};

struct JpegliErrorManager {
  jpeg_error_mgr pub;
  jmp_buf jump_buffer;
  char message[JMSG_LENGTH_MAX];
};

volatile uint64_t decode_sink = 0;

bool ReadProcessEnergy(uint64_t* energy_nj) {
#if defined(__APPLE__)
  rusage_info_v6 usage = {};
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V6,
                      reinterpret_cast<rusage_info_t*>(&usage)) != 0) {
    return false;
  }
  *energy_nj = usage.ri_energy_nj;
  return true;
#else
  (void)energy_nj;
  return false;
#endif
}

void JpegliErrorExit(j_common_ptr cinfo) {
  JpegliErrorManager* error = reinterpret_cast<JpegliErrorManager*>(cinfo->err);
  (*cinfo->err->format_message)(cinfo, error->message);
  longjmp(error->jump_buffer, 1);
}

std::string LowercaseExtension(const fs::path& path) {
  std::string extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char c) { return static_cast<char>(tolower(c)); });
  return extension;
}

bool IsSourceImage(const fs::path& path) {
  const std::string extension = LowercaseExtension(path);
  return extension == ".png" || extension == ".pnm" || extension == ".ppm" ||
         extension == ".pgm" || extension == ".pfm" || extension == ".pgx" ||
         extension == ".jpg" || extension == ".jpeg" || extension == ".gif" ||
         extension == ".exr";
}

bool CollectSourceImages(const fs::path& input, size_t limit,
                         std::vector<fs::path>* paths, std::string* error) {
  std::error_code ec;
  if (fs::is_regular_file(input, ec)) {
    if (LowercaseExtension(input) == ".zip") {
      *error = "ZIP archives are not read directly; extract " + input.string() +
               " and pass the resulting directory";
      return false;
    }
    if (!IsSourceImage(input)) {
      *error = "unsupported source image extension: " + input.string();
      return false;
    }
    paths->push_back(input);
  } else if (fs::is_directory(input, ec)) {
    fs::recursive_directory_iterator it(
        input, fs::directory_options::skip_permission_denied, ec);
    const fs::recursive_directory_iterator end;
    while (!ec && it != end) {
      if (it->is_regular_file(ec) && !ec && IsSourceImage(it->path())) {
        paths->push_back(it->path());
      }
      it.increment(ec);
    }
    if (ec) {
      *error = "failed while scanning " + input.string() + ": " + ec.message();
      return false;
    }
  } else {
    *error = "input does not exist or is not a regular file/directory: " +
             input.string();
    return false;
  }

  std::sort(paths->begin(), paths->end());
  if (limit != 0 && paths->size() > limit) paths->resize(limit);
  if (paths->empty()) {
    *error = "no supported source images found in " + input.string();
    return false;
  }
  return true;
}

bool CheckedOutputSize(size_t width, size_t height, size_t* output_size) {
  if (width == 0 || height == 0 ||
      width > std::numeric_limits<size_t>::max() / kOutputChannels ||
      height > std::numeric_limits<size_t>::max() / (width * kOutputChannels) ||
      width * kOutputChannels >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  *output_size = width * height * kOutputChannels;
  return true;
}

bool LoadSourceImage(const fs::path& path, jpegli::extras::PackedPixelFile* ppf,
                     std::string* error) {
  std::vector<uint8_t> source;
  if (!ReadFile(path.string(), &source)) {
    *error = "failed to read source image " + path.string();
    return false;
  }

  if (!jpegli::extras::DecodeBytes(jpegli::Bytes(source),
                                   jpegli::extras::ColorHints(), ppf)) {
    *error = "failed to decode source image " + path.string();
    return false;
  }
  if (ppf->num_frames() != 1) {
    *error = "source must contain exactly one frame: " + path.string();
    return false;
  }
  return true;
}

bool EncodeSourceImage(const fs::path& path,
                       const jpegli::extras::PackedPixelFile& ppf, int quality,
                       const std::string& chroma_subsampling,
                       int progressive_level, ImageBenchmark* image,
                       std::string* error) {
  jpegli::extras::JpegSettings settings;
  settings.quality = quality;
  settings.chroma_subsampling = chroma_subsampling;
  settings.progressive_level = progressive_level;
  if (!jpegli::extras::EncodeJpeg(ppf, settings, nullptr, &image->jpeg)) {
    *error = "jpegli failed to encode " + path.string();
    return false;
  }

  image->name = path.filename().string();
  image->width = ppf.xsize();
  image->height = ppf.ysize();
  if (!CheckedOutputSize(image->width, image->height, &image->output_size)) {
    *error = "unsupported image dimensions in " + path.string();
    return false;
  }
  return true;
}

bool EncodeImage(const fs::path& path, const Args& args, ImageBenchmark* image,
                 std::string* error) {
  jpegli::extras::PackedPixelFile ppf;
  return LoadSourceImage(path, &ppf, error) &&
         EncodeSourceImage(path, ppf, args.quality, args.chroma_subsampling,
                           args.progressive_level, image, error);
}

bool DecodeWithJpegli(
    const ImageBenchmark& image, uint8_t* output, std::string* error,
    DecodeStageSample* stage_sample = nullptr,
    JpegliAppleMetalMode metal_mode = JPEGLI_APPLE_METAL_DISABLED) {
  jpeg_decompress_struct cinfo = {};
  JpegliErrorManager jerr = {};
  volatile bool created = false;
  const auto now = []() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  };
  const double total_start = stage_sample == nullptr ? 0.0 : now();
  double phase_start = total_start;
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = JpegliErrorExit;
  if (setjmp(jerr.jump_buffer)) {
    if (created) jpegli_destroy_decompress(&cinfo);
    *error = std::string("jpegli: ") + jerr.message;
    return false;
  }

  jpegli_create_decompress(&cinfo);
  created = true;
  jpegli_apple_metal_set_mode(&cinfo, metal_mode);
  if (stage_sample != nullptr) {
    jpegli::SetDecodeProfileEnabled(&cinfo, true);
  }
  jpegli_mem_src(&cinfo, image.jpeg.data(), image.jpeg.size());
  if (stage_sample != nullptr) {
    const double phase_end = now();
    stage_sample
        ->api_seconds[static_cast<size_t>(DecodeApiPhase::kCreateAndSource)] =
        phase_end - phase_start;
    phase_start = phase_end;
  }
  if (jpegli_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
    *error = "jpegli: failed to read JPEG header";
    jpegli_destroy_decompress(&cinfo);
    return false;
  }
  if (stage_sample != nullptr) {
    const double phase_end = now();
    stage_sample
        ->api_seconds[static_cast<size_t>(DecodeApiPhase::kReadHeader)] =
        phase_end - phase_start;
    phase_start = phase_end;
  }
  cinfo.out_color_space = JCS_EXT_RGBA;
  if (!jpegli_start_decompress(&cinfo)) {
    *error = "jpegli: failed to start decompression";
    jpegli_destroy_decompress(&cinfo);
    return false;
  }
  if (stage_sample != nullptr) {
    const double phase_end = now();
    stage_sample
        ->api_seconds[static_cast<size_t>(DecodeApiPhase::kStartDecompress)] =
        phase_end - phase_start;
    phase_start = phase_end;
  }
  if (cinfo.output_width != image.width ||
      cinfo.output_height != image.height ||
      cinfo.output_components != kOutputChannels) {
    *error = "jpegli: unexpected output dimensions or pixel format";
    jpegli_destroy_decompress(&cinfo);
    return false;
  }

  const size_t stride = image.width * kOutputChannels;
  while (cinfo.output_scanline < cinfo.output_height) {
    JSAMPROW row = output + static_cast<size_t>(cinfo.output_scanline) * stride;
    if (jpegli_read_scanlines(&cinfo, &row, 1) != 1) {
      *error = "jpegli: scanline decode was suspended";
      jpegli_destroy_decompress(&cinfo);
      return false;
    }
  }
  if (stage_sample != nullptr) {
    const double phase_end = now();
    stage_sample
        ->api_seconds[static_cast<size_t>(DecodeApiPhase::kReadScanlines)] =
        phase_end - phase_start;
    phase_start = phase_end;
  }
  if (!jpegli_finish_decompress(&cinfo)) {
    *error = "jpegli: failed to finish decompression";
    jpegli_destroy_decompress(&cinfo);
    return false;
  }
  if (stage_sample != nullptr) {
    stage_sample->internal = jpegli::GetDecodeProfile(&cinfo);
  }
  jpegli_destroy_decompress(&cinfo);
  if (stage_sample != nullptr) {
    const double end = now();
    stage_sample
        ->api_seconds[static_cast<size_t>(DecodeApiPhase::kFinishAndDestroy)] =
        end - phase_start;
    stage_sample->total_seconds = end - total_start;
  }
  return true;
}

const char* MetalPathName(MetalDecodePath path) {
  switch (path) {
    case MetalDecodePath::kCpu:
      return "cpu";
    case MetalDecodePath::kMetalScanlines:
      return "metal_to_cpu";
    case MetalDecodePath::kMetalDirect:
      return "metal_direct";
  }
  return "unknown";
}

bool DecodeJpegliMetalPath(const ImageBenchmark& image, MetalDecodePath path,
                           uint8_t* validation_output,
                           MetalDecodeSample* sample, std::string* error) {
  jpeg_decompress_struct cinfo = {};
  JpegliErrorManager jerr = {};
  JpegliAppleMetalOutput metal_output = {};
  volatile bool created = false;
  volatile bool exported = false;
  cinfo.err = jpegli_std_error(&jerr.pub);
  jerr.pub.error_exit = JpegliErrorExit;
  if (setjmp(jerr.jump_buffer)) {
    if (exported) jpegli_apple_metal_release_output(&metal_output);
    if (created) jpegli_destroy_decompress(&cinfo);
    *error = std::string("jpegli: ") + jerr.message;
    return false;
  }

  uint64_t process_energy_start = 0;
  const bool process_energy_started = ReadProcessEnergy(&process_energy_start);
  const auto start = std::chrono::steady_clock::now();
  const std::clock_t process_cpu_start = std::clock();
  jpegli_create_decompress(&cinfo);
  created = true;
  jpegli::SetDecodeProfileEnabled(&cinfo, true);
  jpegli_apple_metal_set_mode(&cinfo, path == MetalDecodePath::kCpu
                                          ? JPEGLI_APPLE_METAL_DISABLED
                                          : JPEGLI_APPLE_METAL_FORCE);
  jpegli_mem_src(&cinfo, image.jpeg.data(), image.jpeg.size());
  if (jpegli_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
    *error = "jpegli: failed to read JPEG header";
    jpegli_destroy_decompress(&cinfo);
    return false;
  }
  cinfo.out_color_space = JCS_EXT_RGBA;

  if (path == MetalDecodePath::kMetalDirect) {
    if (!jpegli_start_decompress_to_apple_metal(&cinfo, &metal_output)) {
      JpegliAppleMetalStats stats = {};
      jpegli_apple_metal_get_stats(&cinfo, &stats);
      *error = "direct Metal decode was unavailable";
      if (stats.fallback_reason[0] != '\0') {
        *error += std::string(": ") + stats.fallback_reason;
      }
      jpegli_destroy_decompress(&cinfo);
      return false;
    }
    exported = true;
    sample->ready_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    sample->process_cpu_seconds =
        static_cast<double>(std::clock() - process_cpu_start) / CLOCKS_PER_SEC;
    uint64_t process_energy_end = 0;
    if (process_energy_started && ReadProcessEnergy(&process_energy_end) &&
        process_energy_end >= process_energy_start) {
      sample->process_energy_nj = process_energy_end - process_energy_start;
      sample->process_energy_available = true;
    }
    if (metal_output.width != image.width ||
        metal_output.height != image.height ||
        metal_output.buffer_contents == nullptr ||
        metal_output.row_bytes < image.width * kOutputChannels) {
      *error = "direct Metal output has invalid dimensions or layout";
      jpegli_apple_metal_release_output(&metal_output);
      jpegli_destroy_decompress(&cinfo);
      return false;
    }
    const uint8_t* source =
        static_cast<const uint8_t*>(metal_output.buffer_contents);
    if (validation_output != nullptr) {
      for (size_t y = 0; y < image.height; ++y) {
        memcpy(validation_output + y * image.width * kOutputChannels,
               source + y * metal_output.row_bytes,
               image.width * kOutputChannels);
      }
    }
    decode_sink ^= source[(image.height / 2) * metal_output.row_bytes +
                          (image.width / 2) * kOutputChannels];
  } else {
    if (!jpegli_start_decompress(&cinfo)) {
      *error = "jpegli: failed to start decompression";
      jpegli_destroy_decompress(&cinfo);
      return false;
    }
    if (cinfo.output_width != image.width ||
        cinfo.output_height != image.height ||
        cinfo.output_components != kOutputChannels) {
      *error = "jpegli: unexpected output dimensions or format";
      jpegli_destroy_decompress(&cinfo);
      return false;
    }
    if (validation_output == nullptr) {
      *error = "scanline benchmark requires an output buffer";
      jpegli_destroy_decompress(&cinfo);
      return false;
    }
    const size_t stride = image.width * kOutputChannels;
    while (cinfo.output_scanline < cinfo.output_height) {
      JSAMPROW row = validation_output +
                     static_cast<size_t>(cinfo.output_scanline) * stride;
      if (jpegli_read_scanlines(&cinfo, &row, 1) != 1) {
        *error = "jpegli: scanline decode was suspended";
        jpegli_destroy_decompress(&cinfo);
        return false;
      }
    }
    sample->ready_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    sample->process_cpu_seconds =
        static_cast<double>(std::clock() - process_cpu_start) / CLOCKS_PER_SEC;
    uint64_t process_energy_end = 0;
    if (process_energy_started && ReadProcessEnergy(&process_energy_end) &&
        process_energy_end >= process_energy_start) {
      sample->process_energy_nj = process_energy_end - process_energy_start;
      sample->process_energy_available = true;
    }
    decode_sink ^= validation_output[image.output_size / 2];
  }

  if (!jpegli_finish_decompress(&cinfo)) {
    *error = "jpegli: failed to finish decompression";
    if (exported) jpegli_apple_metal_release_output(&metal_output);
    jpegli_destroy_decompress(&cinfo);
    return false;
  }
  sample->jpegli_memory_bytes =
      jpegli::MemoryManagerPeakBytes(reinterpret_cast<j_common_ptr>(&cinfo));
  sample->internal = jpegli::GetDecodeProfile(&cinfo);
  jpegli_apple_metal_get_stats(&cinfo, &sample->metal);
  if (path != MetalDecodePath::kCpu && !sample->metal.used_metal) {
    *error = "forced Metal decode fell back to CPU";
    if (sample->metal.fallback_reason[0] != '\0') {
      *error += std::string(": ") + sample->metal.fallback_reason;
    }
    if (exported) jpegli_apple_metal_release_output(&metal_output);
    jpegli_destroy_decompress(&cinfo);
    return false;
  }
  jpegli_destroy_decompress(&cinfo);
  if (exported) jpegli_apple_metal_release_output(&metal_output);
  sample->total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  return true;
}

bool DecodeWithTurboJPEG(const ImageBenchmark& image, uint8_t* output,
                         std::string* error) {
  tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
  if (handle == nullptr) {
    *error = std::string("TurboJPEG: ") + tj3GetErrorStr(nullptr);
    return false;
  }
  const auto fail = [&](const char* operation) {
    *error =
        std::string("TurboJPEG ") + operation + ": " + tj3GetErrorStr(handle);
    tj3Destroy(handle);
    return false;
  };

  if (tj3DecompressHeader(handle, image.jpeg.data(), image.jpeg.size()) < 0) {
    return fail("header decode failed");
  }
  if (tj3Get(handle, TJPARAM_JPEGWIDTH) != static_cast<int>(image.width) ||
      tj3Get(handle, TJPARAM_JPEGHEIGHT) != static_cast<int>(image.height)) {
    return fail("reported unexpected dimensions");
  }
  const int pitch = static_cast<int>(image.width * kOutputChannels);
  if (tj3Decompress8(handle, image.jpeg.data(), image.jpeg.size(), output,
                     pitch, TJPF_RGBA) < 0) {
    return fail("pixel decode failed");
  }
  tj3Destroy(handle);
  return true;
}

#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
bool DecodeWithAppleImageIO(const ImageBenchmark& image, uint8_t* output,
                            std::string* error) {
  if (image.jpeg.size() >
      static_cast<size_t>(std::numeric_limits<CFIndex>::max())) {
    *error = "Apple ImageIO: JPEG input is too large";
    return false;
  }

  CFDataRef data = CFDataCreateWithBytesNoCopy(
      kCFAllocatorDefault, image.jpeg.data(),
      static_cast<CFIndex>(image.jpeg.size()), kCFAllocatorNull);
  if (data == nullptr) {
    *error = "Apple ImageIO: failed to create input data";
    return false;
  }
  CGImageSourceRef source = CGImageSourceCreateWithData(data, nullptr);
  if (source == nullptr) {
    CFRelease(data);
    *error = "Apple ImageIO: failed to create image source";
    return false;
  }
  CGImageRef cg_image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
  if (cg_image == nullptr) {
    CFRelease(source);
    CFRelease(data);
    *error = "Apple ImageIO: failed to decode JPEG";
    return false;
  }
  if (CGImageGetWidth(cg_image) != image.width ||
      CGImageGetHeight(cg_image) != image.height) {
    CGImageRelease(cg_image);
    CFRelease(source);
    CFRelease(data);
    *error = "Apple ImageIO: reported unexpected dimensions";
    return false;
  }

  CGColorSpaceRef color_space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  CGContextRef context =
      color_space == nullptr
          ? nullptr
          : CGBitmapContextCreate(
                output, image.width, image.height, 8,
                image.width * kOutputChannels, color_space,
                kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
  if (context == nullptr) {
    if (color_space != nullptr) CGColorSpaceRelease(color_space);
    CGImageRelease(cg_image);
    CFRelease(source);
    CFRelease(data);
    *error = "Apple ImageIO: failed to create 32-bit RGBA bitmap context";
    return false;
  }

  CGContextSetBlendMode(context, kCGBlendModeCopy);
  CGContextDrawImage(context,
                     CGRectMake(0, 0, static_cast<CGFloat>(image.width),
                                static_cast<CGFloat>(image.height)),
                     cg_image);
  CGContextFlush(context);

  CGContextRelease(context);
  CGColorSpaceRelease(color_space);
  CGImageRelease(cg_image);
  CFRelease(source);
  CFRelease(data);
  return true;
}
#endif

uint64_t Checksum(const uint8_t* bytes, size_t size) {
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

bool HasOpaqueAlpha(const uint8_t* bytes, size_t size) {
  for (size_t i = kComparedChannels; i < size; i += kOutputChannels) {
    if (bytes[i] != 255) return false;
  }
  return true;
}

void MeasureOutputDifference(const uint8_t* reference, const uint8_t* output,
                             size_t size, double* mean_abs_diff,
                             int* max_abs_diff) {
  uint64_t sum_abs_diff = 0;
  *max_abs_diff = 0;
  for (size_t i = 0; i < size; i += kOutputChannels) {
    for (size_t c = 0; c < kComparedChannels; ++c) {
      const int diff = std::abs(static_cast<int>(reference[i + c]) -
                                static_cast<int>(output[i + c]));
      sum_abs_diff += static_cast<unsigned int>(diff);
      *max_abs_diff = std::max(*max_abs_diff, diff);
    }
  }
  const size_t samples = size / kOutputChannels * kComparedChannels;
  *mean_abs_diff = static_cast<double>(sum_abs_diff) / samples;
}

Summary Summarize(const std::vector<double>& samples) {
  Summary summary;
  summary.minimum = *std::min_element(samples.begin(), samples.end());
  summary.mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const size_t middle = sorted.size() / 2;
  summary.median = sorted.size() % 2 == 0
                       ? 0.5 * (sorted[middle - 1] + sorted[middle])
                       : sorted[middle];
  return summary;
}

enum class Decoder {
  kJpegli,
  kTurboJPEG,
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  kAppleImageIO,
#endif
};

const char* DecoderName(Decoder decoder) {
  switch (decoder) {
    case Decoder::kJpegli:
      return "jpegli";
    case Decoder::kTurboJPEG:
      return "turbojpeg";
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    case Decoder::kAppleImageIO:
      return "apple_imageio";
#endif
  }
  return "unknown";
}

bool DecodeImage(Decoder decoder, const ImageBenchmark& image, uint8_t* output,
                 std::string* error) {
  switch (decoder) {
    case Decoder::kJpegli:
      return DecodeWithJpegli(image, output, error);
    case Decoder::kTurboJPEG:
      return DecodeWithTurboJPEG(image, output, error);
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    case Decoder::kAppleImageIO:
      return DecodeWithAppleImageIO(image, output, error);
#endif
  }
  *error = "unknown decoder";
  return false;
}

bool RunDecode(Decoder decoder, ImageBenchmark* image, uint8_t* output,
               double* elapsed, std::string* error) {
  error->clear();
  const auto start = std::chrono::steady_clock::now();
  const bool ok = DecodeImage(decoder, *image, output, error);
  const auto end = std::chrono::steady_clock::now();
  if (!ok) return false;
  *elapsed = std::chrono::duration<double>(end - start).count();
  decode_sink ^= output[image->output_size / 2];
  return true;
}

bool RunDecoders(ImageBenchmark* image, const std::vector<Decoder>& order,
                 bool record, uint8_t* jpegli_output, uint8_t* turbojpeg_output,
                 uint8_t* imageio_output, std::string* error) {
  for (Decoder decoder : order) {
    uint8_t* output = nullptr;
    std::vector<double>* samples = nullptr;
    switch (decoder) {
      case Decoder::kJpegli:
        output = jpegli_output;
        samples = &image->jpegli_seconds;
        break;
      case Decoder::kTurboJPEG:
        output = turbojpeg_output;
        samples = &image->turbojpeg_seconds;
        break;
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
      case Decoder::kAppleImageIO:
        output = imageio_output;
        samples = &image->imageio_seconds;
        break;
#endif
    }
    double elapsed = 0.0;
    if (!RunDecode(decoder, image, output, &elapsed, error)) return false;
    if (record) samples->push_back(elapsed);
  }
  return true;
}

std::string ShortName(const std::string& name) {
  if (name.size() <= 17) return name;
  return name.substr(0, 12) + "...";
}

std::string CsvEscape(const std::string& text) {
  if (text.find_first_of(",\"\r\n") == std::string::npos) return text;
  std::string escaped = "\"";
  for (char c : text) {
    if (c == '\"') escaped.push_back('\"');
    escaped.push_back(c);
  }
  escaped.push_back('\"');
  return escaped;
}

bool WriteCsv(const std::string& path,
              const std::vector<ImageBenchmark>& images, const Args& args,
              std::string* error) {
  std::ofstream out(path);
  if (!out) {
    *error = "failed to open CSV output " + path;
    return false;
  }
  out << "image,width,height,jpeg_bytes,quality,chroma_subsampling,"
         "progressive_level,iterations,jpegli_version,turbojpeg_api,"
         "jpegli_median_ms,turbojpeg_median_ms,";
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  out << "apple_imageio_median_ms,";
#endif
  out << "jpegli_mean_ms,turbojpeg_mean_ms,";
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  out << "apple_imageio_mean_ms,";
#endif
  out << "jpegli_min_ms,turbojpeg_min_ms,";
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  out << "apple_imageio_min_ms,";
#endif
  out << "jpegli_mpix_s,turbojpeg_mpix_s,";
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  out << "apple_imageio_mpix_s,";
#endif
  out << "jpegli_over_turbo,";
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  out << "jpegli_over_apple_imageio,";
#endif
  out << "turbojpeg_mean_abs_diff_vs_jpegli,"
         "turbojpeg_max_abs_diff_vs_jpegli,";
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  out << "apple_imageio_mean_abs_diff_vs_jpegli,"
         "apple_imageio_max_abs_diff_vs_jpegli,";
#endif
  out << "jpegli_checksum,turbojpeg_checksum";
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  out << ",apple_imageio_checksum";
#endif
  out << '\n';
  out << std::fixed << std::setprecision(6);
  for (const ImageBenchmark& image : images) {
    const Summary jpegli = Summarize(image.jpegli_seconds);
    const Summary turbojpeg = Summarize(image.turbojpeg_seconds);
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    const Summary imageio = Summarize(image.imageio_seconds);
#endif
    const double mpix = static_cast<double>(image.width) * image.height / 1e6;
    out << CsvEscape(image.name) << ',' << image.width << ',' << image.height
        << ',' << image.jpeg.size() << ',' << args.quality << ','
        << args.chroma_subsampling << ',' << args.progressive_level << ','
        << image.jpegli_seconds.size() << ',' << kJpegliVersion << ','
        << TURBOJPEG_VERSION_NUMBER << ',' << jpegli.median * 1e3 << ','
        << turbojpeg.median * 1e3 << ',';
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    out << imageio.median * 1e3 << ',';
#endif
    out << jpegli.mean * 1e3 << ',' << turbojpeg.mean * 1e3 << ',';
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    out << imageio.mean * 1e3 << ',';
#endif
    out << jpegli.minimum * 1e3 << ',' << turbojpeg.minimum * 1e3 << ',';
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    out << imageio.minimum * 1e3 << ',';
#endif
    out << mpix / jpegli.median << ',' << mpix / turbojpeg.median << ',';
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    out << mpix / imageio.median << ',';
#endif
    out << jpegli.median / turbojpeg.median << ',';
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    out << jpegli.median / imageio.median << ',';
#endif
    out << image.turbojpeg_mean_abs_diff << ',' << image.turbojpeg_max_abs_diff
        << ',';
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    out << image.imageio_mean_abs_diff << ',' << image.imageio_max_abs_diff
        << ',';
#endif
    out << image.jpegli_checksum << ',' << image.turbojpeg_checksum;
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    out << ',' << image.imageio_checksum;
#endif
    out << '\n';
  }
  if (!out) {
    *error = "failed while writing CSV output " + path;
    return false;
  }
  return true;
}

void PrintResults(const std::vector<ImageBenchmark>& images) {
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  printf("\n%-17s %11s %9s %11s %11s %11s\n", "image", "dimensions", "JPEG KiB",
         "jpegli ms", "turbo ms", "ImageIO ms");
  printf("%-17s %11s %9s %11s %11s %11s\n", "-----------------", "-----------",
         "---------", "-----------", "-----------", "-----------");
#else
  printf("\n%-17s %11s %9s %11s %11s %10s\n", "image", "dimensions", "JPEG KiB",
         "jpegli ms", "turbo ms", "turbo x");
  printf("%-17s %11s %9s %11s %11s %10s\n", "-----------------", "-----------",
         "---------", "-----------", "-----------", "----------");
#endif

  double total_pixels = 0.0;
  size_t total_jpeg_bytes = 0;
  double total_jpegli_seconds = 0.0;
  double total_turbojpeg_seconds = 0.0;
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  double total_imageio_seconds = 0.0;
#endif
  for (const ImageBenchmark& image : images) {
    const Summary jpegli = Summarize(image.jpegli_seconds);
    const Summary turbojpeg = Summarize(image.turbojpeg_seconds);
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    const Summary imageio = Summarize(image.imageio_seconds);
#endif
    std::ostringstream dimensions;
    dimensions << image.width << 'x' << image.height;
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    printf("%-17s %11s %9.1f %11.3f %11.3f %11.3f\n",
           ShortName(image.name).c_str(), dimensions.str().c_str(),
           image.jpeg.size() / 1024.0, jpegli.median * 1e3,
           turbojpeg.median * 1e3, imageio.median * 1e3);
#else
    printf("%-17s %11s %9.1f %11.3f %11.3f %10.2f\n",
           ShortName(image.name).c_str(), dimensions.str().c_str(),
           image.jpeg.size() / 1024.0, jpegli.median * 1e3,
           turbojpeg.median * 1e3, jpegli.median / turbojpeg.median);
#endif
    total_pixels += static_cast<double>(image.width) * image.height;
    total_jpeg_bytes += image.jpeg.size();
    total_jpegli_seconds += jpegli.median;
    total_turbojpeg_seconds += turbojpeg.median;
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    total_imageio_seconds += imageio.median;
#endif
  }

  printf("\n%d images, %.2f MP, %.2f MiB of JPEG data\n",
         static_cast<int>(images.size()), total_pixels / 1e6,
         total_jpeg_bytes / (1024.0 * 1024.0));
  printf("jpegli:    %8.3f ms, %8.2f MP/s\n", total_jpegli_seconds * 1e3,
         total_pixels / 1e6 / total_jpegli_seconds);
  printf("TurboJPEG: %8.3f ms, %8.2f MP/s\n", total_turbojpeg_seconds * 1e3,
         total_pixels / 1e6 / total_turbojpeg_seconds);
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  printf("ImageIO:    %8.3f ms, %8.2f MP/s\n", total_imageio_seconds * 1e3,
         total_pixels / 1e6 / total_imageio_seconds);
#endif
  printf("TurboJPEG speedup over jpegli: %.3fx\n",
         total_jpegli_seconds / total_turbojpeg_seconds);
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  printf("ImageIO speedup over jpegli:    %.3fx\n",
         total_jpegli_seconds / total_imageio_seconds);
#endif
}

jpegli::StatusOr<jpegli::extras::PackedPixelFile> MakeMetricImage(
    const jpegli::extras::PackedPixelFile& reference, const uint8_t* rgba,
    size_t width, size_t height) {
  jpegli::extras::PackedPixelFile ppf;
  ppf.info.xsize = width;
  ppf.info.ysize = height;
  ppf.info.bits_per_sample = 8;
  ppf.info.exponent_bits_per_sample = 0;
  ppf.info.intensity_target = reference.info.intensity_target;
  ppf.info.uses_original_profile = reference.info.uses_original_profile;
  ppf.info.num_color_channels = 3;
  ppf.info.num_extra_channels = 0;
  ppf.info.alpha_bits = 0;
  ppf.info.alpha_exponent_bits = 0;
  ppf.info.orientation = JPEGLI_ORIENT_IDENTITY;
  ppf.primary_color_representation =
      jpegli::extras::PackedPixelFile::kColorEncodingIsPrimary;
  ppf.color_encoding = jpegli::ColorEncoding::SRGB().ToExternal();

  const JpegliPixelFormat format{/*num_channels=*/3, JPEGLI_TYPE_UINT8,
                                 JPEGLI_NATIVE_ENDIAN, /*align=*/0};
  JPEGLI_ASSIGN_OR_RETURN(
      jpegli::extras::PackedFrame frame,
      jpegli::extras::PackedFrame::Create(width, height, format));
  for (size_t y = 0; y < height; ++y) {
    const uint8_t* src = rgba + y * width * kOutputChannels;
    uint8_t* dst = reinterpret_cast<uint8_t*>(frame.color.pixels()) +
                   y * frame.color.stride;
    for (size_t x = 0; x < width; ++x) {
      dst[3 * x + 0] = src[kOutputChannels * x + 0];
      dst[3 * x + 1] = src[kOutputChannels * x + 1];
      dst[3 * x + 2] = src[kOutputChannels * x + 2];
    }
  }
  ppf.frames.emplace_back(std::move(frame));
  return ppf;
}

jpegli::Status ComputePerceptualScore(
    const jpegli::extras::PackedPixelFile& reference, const uint8_t* rgba,
    size_t width, size_t height, PerceptualScore* score) {
  JPEGLI_ASSIGN_OR_RETURN(jpegli::extras::PackedPixelFile distorted,
                          MakeMetricImage(reference, rgba, width, height));
  JPEGLI_ASSIGN_OR_RETURN(Msssim msssim,
                          ComputeSSIMULACRA2(reference, distorted));
  score->ssimulacra2 = msssim.Score();

  jpegli::ButteraugliParams params;
  params.intensity_target = 80.0f;
  score->butteraugli = jpegli::ButteraugliDistance(
      NoMemoryManager(), reference, distorted, params, /*distmap=*/nullptr,
      /*pool=*/nullptr, /*ignore_alpha=*/true);
  JPEGLI_ENSURE(std::isfinite(score->ssimulacra2));
  JPEGLI_ENSURE(std::isfinite(score->butteraugli));
  JPEGLI_ENSURE(score->butteraugli < std::numeric_limits<float>::max());
  return true;
}

std::string Trim(const std::string& value) {
  const size_t begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) return std::string();
  const size_t end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

bool IsValidSubsampling(const std::string& value) {
  return value == "444" || value == "440" || value == "422" || value == "420";
}

bool ParseQualitySweep(const Args& args, std::vector<int>* qualities,
                       std::vector<std::string>* subsamplings,
                       std::string* error) {
  std::stringstream quality_stream(args.quality_levels);
  std::string token;
  while (std::getline(quality_stream, token, ',')) {
    token = Trim(token);
    char* end = nullptr;
    const long value = strtol(token.c_str(), &end, 10);
    if (token.empty() || *end != '\0' || value < 1 || value > 100) {
      *error = "invalid quality in --quality_levels: " + token;
      return false;
    }
    qualities->push_back(static_cast<int>(value));
  }
  if (qualities->empty()) {
    *error = "--quality_levels must not be empty";
    return false;
  }

  std::stringstream subsampling_stream(args.quality_subsampling);
  while (std::getline(subsampling_stream, token, ',')) {
    token = Trim(token);
    if (!IsValidSubsampling(token)) {
      *error = "invalid chroma mode in --quality_subsampling: " + token;
      return false;
    }
    subsamplings->push_back(token);
  }
  if (subsamplings->empty()) {
    *error = "--quality_subsampling must not be empty";
    return false;
  }
  return true;
}

bool RunQualityTask(const QualityTask& task, const Args& args,
                    QualityResult* result, std::string* error) {
  jpegli::extras::PackedPixelFile reference;
  if (!LoadSourceImage(task.path, &reference, error)) return false;
  if (reference.info.num_color_channels != 3 ||
      reference.info.num_extra_channels != 0) {
    *error = "perceptual quality mode requires an opaque RGB source: " +
             task.path.string();
    return false;
  }
  if (reference.info.orientation != JPEGLI_ORIENT_IDENTITY) {
    *error = "perceptual quality mode requires identity orientation: " +
             task.path.string();
    return false;
  }
  jpegli::ColorEncoding reference_encoding;
  if (!jpegli::extras::GetColorEncoding(reference, &reference_encoding) ||
      !reference_encoding.IsSRGB()) {
    *error = "perceptual quality mode requires an sRGB source: " +
             task.path.string();
    return false;
  }

  ImageBenchmark image;
  if (!EncodeSourceImage(task.path, reference, task.quality,
                         task.chroma_subsampling, args.progressive_level,
                         &image, error)) {
    return false;
  }

  result->name = image.name;
  result->width = image.width;
  result->height = image.height;
  result->quality = task.quality;
  result->chroma_subsampling = task.chroma_subsampling;
  result->progressive_level = args.progressive_level;
  result->jpeg_bytes = image.jpeg.size();

  std::vector<Decoder> decoders = {Decoder::kJpegli, Decoder::kTurboJPEG};
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  decoders.push_back(Decoder::kAppleImageIO);
#endif
  std::vector<uint8_t> output(image.output_size);
  for (Decoder decoder : decoders) {
    if (!DecodeImage(decoder, image, output.data(), error)) {
      *error = image.name + ": " + *error;
      return false;
    }
    if (!HasOpaqueAlpha(output.data(), image.output_size)) {
      *error = image.name + ": " + DecoderName(decoder) +
               " produced non-opaque RGBA output";
      return false;
    }
    PerceptualScore score;
    score.decoder = DecoderName(decoder);
    if (!ComputePerceptualScore(reference, output.data(), image.width,
                                image.height, &score)) {
      *error = image.name + ": metric computation failed for " +
               DecoderName(decoder);
      return false;
    }
    result->scores.push_back(std::move(score));
  }
  return true;
}

bool WriteQualityCsv(const std::string& path,
                     const std::vector<QualityResult>& results,
                     std::string* error) {
  std::ofstream out(path);
  if (!out) {
    *error = "failed to open CSV output " + path;
    return false;
  }
  out << "image,width,height,quality,chroma_subsampling,progressive_level,"
         "jpeg_bytes,bits_per_pixel,jpegli_version,turbojpeg_api,decoder,"
         "ssimulacra2,butteraugli\n";
  out << std::fixed << std::setprecision(9);
  for (const QualityResult& result : results) {
    const double pixels = static_cast<double>(result.width) * result.height;
    const double bits_per_pixel = 8.0 * result.jpeg_bytes / pixels;
    for (const PerceptualScore& score : result.scores) {
      out << CsvEscape(result.name) << ',' << result.width << ','
          << result.height << ',' << result.quality << ','
          << result.chroma_subsampling << ',' << result.progressive_level << ','
          << result.jpeg_bytes << ',' << bits_per_pixel << ',' << kJpegliVersion
          << ',' << TURBOJPEG_VERSION_NUMBER << ',' << score.decoder << ','
          << score.ssimulacra2 << ',' << score.butteraugli << '\n';
    }
  }
  if (!out) {
    *error = "failed while writing CSV output " + path;
    return false;
  }
  return true;
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle])
                                : values[middle];
}

struct QualityAggregate {
  size_t images = 0;
  double pixels = 0.0;
  double jpeg_bytes = 0.0;
  std::vector<double> ssimulacra2;
  std::vector<double> butteraugli;
};

void PrintQualityResults(const std::vector<QualityResult>& results) {
  using Key = std::tuple<int, std::string, std::string>;
  std::map<Key, QualityAggregate> aggregates;
  for (const QualityResult& result : results) {
    const double pixels = static_cast<double>(result.width) * result.height;
    for (const PerceptualScore& score : result.scores) {
      QualityAggregate& aggregate = aggregates[Key(
          result.quality, result.chroma_subsampling, score.decoder)];
      ++aggregate.images;
      aggregate.pixels += pixels;
      aggregate.jpeg_bytes += result.jpeg_bytes;
      aggregate.ssimulacra2.push_back(score.ssimulacra2);
      aggregate.butteraugli.push_back(score.butteraugli);
    }
  }

  printf("\n%3s %6s %-14s %8s %12s %12s %12s %12s %12s %12s\n", "q", "chroma",
         "decoder", "bpp", "SSIM2 mean", "SSIM2 med", "SSIM2 worst", "BA mean",
         "BA med", "BA worst");
  printf("%3s %6s %-14s %8s %12s %12s %12s %12s %12s %12s\n", "---", "------",
         "--------------", "--------", "------------", "------------",
         "------------", "------------", "------------", "------------");
  for (const auto& entry : aggregates) {
    const int quality = std::get<0>(entry.first);
    const std::string& subsampling = std::get<1>(entry.first);
    const std::string& decoder = std::get<2>(entry.first);
    const QualityAggregate& aggregate = entry.second;
    const double ssim_mean = std::accumulate(aggregate.ssimulacra2.begin(),
                                             aggregate.ssimulacra2.end(), 0.0) /
                             aggregate.ssimulacra2.size();
    const double butteraugli_mean =
        std::accumulate(aggregate.butteraugli.begin(),
                        aggregate.butteraugli.end(), 0.0) /
        aggregate.butteraugli.size();
    const double bits_per_pixel = 8.0 * aggregate.jpeg_bytes / aggregate.pixels;
    printf(
        "%3d %6s %-14s %8.4f %12.6f %12.6f %12.6f %12.6f %12.6f "
        "%12.6f\n",
        quality, subsampling.c_str(), decoder.c_str(), bits_per_pixel,
        ssim_mean, Median(aggregate.ssimulacra2),
        *std::min_element(aggregate.ssimulacra2.begin(),
                          aggregate.ssimulacra2.end()),
        butteraugli_mean, Median(aggregate.butteraugli),
        *std::max_element(aggregate.butteraugli.begin(),
                          aggregate.butteraugli.end()));
  }
  printf("\nSSIMULACRA2: higher is better. Butteraugli: lower is better.\n");
  printf(
      "Means and medians are across images; worst is min SSIMULACRA2 or "
      "max Butteraugli.\n");
}

int RunPerceptualQualitySweep(const std::vector<fs::path>& paths,
                              const Args& args) {
  std::vector<int> qualities;
  std::vector<std::string> subsamplings;
  std::string error;
  if (!ParseQualitySweep(args, &qualities, &subsamplings, &error)) {
    fprintf(stderr, "%s\n", error.c_str());
    return EXIT_FAILURE;
  }

  std::vector<QualityTask> tasks;
  tasks.reserve(paths.size() * qualities.size() * subsamplings.size());
  for (int quality : qualities) {
    for (const std::string& subsampling : subsamplings) {
      for (const fs::path& path : paths) {
        tasks.push_back(QualityTask{path, quality, subsampling});
      }
    }
  }
  std::vector<QualityResult> results(tasks.size());
  fprintf(stderr,
          "Scoring %zu image/setting combinations serially; each combination "
          "is decoded by every decoder.\n",
          tasks.size());
  fprintf(stderr,
          "Reference: original source pixels. Metrics: SSIMULACRA2 and "
          "Butteraugli (80 nit SDR).\n");
  fprintf(stderr, "jpegli %s; TurboJPEG API %d.%d.%d\n", kJpegliVersion,
          TURBOJPEG_VERSION_NUMBER / 1000000,
          (TURBOJPEG_VERSION_NUMBER / 1000) % 1000,
          TURBOJPEG_VERSION_NUMBER % 1000);
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  fprintf(stderr, "Apple ImageIO enabled with explicit sRGB output\n");
#endif

  const size_t progress_step = std::max<size_t>(1, tasks.size() / 20);
  for (size_t index = 0; index < tasks.size(); ++index) {
    std::string task_error;
    if (!RunQualityTask(tasks[index], args, &results[index], &task_error)) {
      fprintf(stderr, "%s\n", task_error.c_str());
      return EXIT_FAILURE;
    }
    const size_t done = index + 1;
    if (done == tasks.size() || done % progress_step == 0) {
      fprintf(stderr, "[%zu/%zu] perceptual scores complete\n", done,
              tasks.size());
    }
  }

  PrintQualityResults(results);
  if (!args.csv.empty()) {
    if (!WriteQualityCsv(args.csv, results, &error)) {
      fprintf(stderr, "%s\n", error.c_str());
      return EXIT_FAILURE;
    }
    printf("Per-image perceptual-quality CSV written to %s\n",
           args.csv.c_str());
  }
  return EXIT_SUCCESS;
}

const char* CodingModeName(int progressive_level) {
  return progressive_level == 0 ? "baseline" : "progressive";
}

const char* InternalStageName(jpegli::DecodeProfileStage stage) {
  switch (stage) {
    case jpegli::DecodeProfileStage::kSetup:
      return "setup";
    case jpegli::DecodeProfileStage::kMarkers:
      return "markers";
    case jpegli::DecodeProfileStage::kEntropy:
      return "entropy";
    case jpegli::DecodeProfileStage::kCoefficientAnalysis:
      return "coefficient_analysis";
    case jpegli::DecodeProfileStage::kIdct:
      return "idct";
    case jpegli::DecodeProfileStage::kUpsampling:
      return "upsample_row_prep";
    case jpegli::DecodeProfileStage::kColorConversion:
      return "color_conversion";
    case jpegli::DecodeProfileStage::kPixelOutput:
      return "pixel_output";
    case jpegli::DecodeProfileStage::kCount:
      break;
  }
  return "unknown";
}

const char* ApiPhaseName(DecodeApiPhase phase) {
  switch (phase) {
    case DecodeApiPhase::kCreateAndSource:
      return "create_and_source";
    case DecodeApiPhase::kReadHeader:
      return "read_header";
    case DecodeApiPhase::kStartDecompress:
      return "start_decompress";
    case DecodeApiPhase::kReadScanlines:
      return "read_scanlines";
    case DecodeApiPhase::kFinishAndDestroy:
      return "finish_and_destroy";
    case DecodeApiPhase::kCount:
      break;
  }
  return "unknown";
}

bool ParseStageProgressiveLevels(const Args& args, std::vector<int>* levels,
                                 std::string* error) {
  std::stringstream stream(args.stage_progressive_levels);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token = Trim(token);
    char* end = nullptr;
    const long value = strtol(token.c_str(), &end, 10);
    if (token.empty() || *end != '\0' || value < 0 || value > 2) {
      *error = "invalid level in --stage_progressive_levels: " + token;
      return false;
    }
    if (std::find(levels->begin(), levels->end(), value) != levels->end()) {
      *error = "duplicate level in --stage_progressive_levels: " + token;
      return false;
    }
    levels->push_back(static_cast<int>(value));
  }
  if (levels->empty()) {
    *error = "--stage_progressive_levels must not be empty";
    return false;
  }
  return true;
}

bool RunStageDecode(DecodeStageBenchmark* benchmark, uint8_t* output,
                    bool record, std::string* error) {
  DecodeStageSample sample;
  if (!DecodeWithJpegli(benchmark->image, output, error, &sample)) {
    return false;
  }
  decode_sink ^= output[benchmark->image.output_size / 2];
  if (record) benchmark->samples.push_back(sample);
  return true;
}

const DecodeStageSample& RepresentativeStageSample(
    const std::vector<DecodeStageSample>& samples) {
  std::vector<const DecodeStageSample*> sorted;
  sorted.reserve(samples.size());
  for (const DecodeStageSample& sample : samples) sorted.push_back(&sample);
  std::sort(sorted.begin(), sorted.end(),
            [](const DecodeStageSample* a, const DecodeStageSample* b) {
              return a->total_seconds < b->total_seconds;
            });
  return *sorted[sorted.size() / 2];
}

double InternalProfileSeconds(const jpegli::DecodeProfile& profile,
                              size_t stage) {
  return profile.nanoseconds[stage] * 1e-9;
}

double InternalProfileTotalSeconds(const jpegli::DecodeProfile& profile) {
  double total = 0.0;
  for (size_t stage = 0; stage < jpegli::kNumDecodeProfileStages; ++stage) {
    total += InternalProfileSeconds(profile, stage);
  }
  return total;
}

struct DecodeStageAggregate {
  size_t images = 0;
  double pixels = 0.0;
  double jpeg_bytes = 0.0;
  double total_seconds = 0.0;
  double api_seconds[kNumDecodeApiPhases] = {};
  double internal_seconds[jpegli::kNumDecodeProfileStages] = {};
  uint64_t internal_calls[jpegli::kNumDecodeProfileStages] = {};
};

void PrintStageResults(const std::vector<DecodeStageBenchmark>& benchmarks,
                       const std::vector<int>& progressive_levels) {
  std::map<int, DecodeStageAggregate> aggregates;
  for (const DecodeStageBenchmark& benchmark : benchmarks) {
    const DecodeStageSample& sample =
        RepresentativeStageSample(benchmark.samples);
    DecodeStageAggregate& aggregate = aggregates[benchmark.progressive_level];
    ++aggregate.images;
    aggregate.pixels +=
        static_cast<double>(benchmark.image.width) * benchmark.image.height;
    aggregate.jpeg_bytes += benchmark.image.jpeg.size();
    aggregate.total_seconds += sample.total_seconds;
    for (size_t phase = 0; phase < kNumDecodeApiPhases; ++phase) {
      aggregate.api_seconds[phase] += sample.api_seconds[phase];
    }
    for (size_t stage = 0; stage < jpegli::kNumDecodeProfileStages; ++stage) {
      aggregate.internal_seconds[stage] +=
          InternalProfileSeconds(sample.internal, stage);
      aggregate.internal_calls[stage] += sample.internal.calls[stage];
    }
  }

  for (int level : progressive_levels) {
    const DecodeStageAggregate& aggregate = aggregates.at(level);
    const double internal_total =
        std::accumulate(std::begin(aggregate.internal_seconds),
                        std::end(aggregate.internal_seconds), 0.0);
    const double unattributed =
        std::max(0.0, aggregate.total_seconds - internal_total);
    printf("\nProgressive level %d (%s): %zu images, %.2f MP, %.2f MiB\n",
           level, CodingModeName(level), aggregate.images,
           aggregate.pixels / 1e6, aggregate.jpeg_bytes / (1024.0 * 1024.0));
    printf("Total: %.3f ms, %.2f MP/s\n", aggregate.total_seconds * 1e3,
           aggregate.pixels / 1e6 / aggregate.total_seconds);

    printf("\n%-24s %12s %10s %14s\n", "internal stage", "time ms", "percent",
           "calls/image");
    printf("%-24s %12s %10s %14s\n", "------------------------", "------------",
           "----------", "--------------");
    for (size_t stage = 0; stage < jpegli::kNumDecodeProfileStages; ++stage) {
      const double seconds = aggregate.internal_seconds[stage];
      printf("%-24s %12.3f %9.2f%% %14.1f\n",
             InternalStageName(static_cast<jpegli::DecodeProfileStage>(stage)),
             seconds * 1e3, 100.0 * seconds / aggregate.total_seconds,
             static_cast<double>(aggregate.internal_calls[stage]) /
                 aggregate.images);
    }
    printf("%-24s %12.3f %9.2f%% %14s\n", "unattributed/overhead",
           unattributed * 1e3, 100.0 * unattributed / aggregate.total_seconds,
           "-");

    printf("\n%-24s %12s %10s\n", "libjpeg API phase", "time ms", "percent");
    printf("%-24s %12s %10s\n", "------------------------", "------------",
           "----------");
    for (size_t phase = 0; phase < kNumDecodeApiPhases; ++phase) {
      const double seconds = aggregate.api_seconds[phase];
      printf("%-24s %12.3f %9.2f%%\n",
             ApiPhaseName(static_cast<DecodeApiPhase>(phase)), seconds * 1e3,
             100.0 * seconds / aggregate.total_seconds);
    }
  }
  printf(
      "\nInternal stages are disjoint. API phases are an alternate view of "
      "the same total.\n");
  printf(
      "Each image contributes the complete stage record from its "
      "median-total timed iteration.\n");
}

bool WriteStageCsv(const std::string& path,
                   const std::vector<DecodeStageBenchmark>& benchmarks,
                   const Args& args, std::string* error) {
  std::ofstream out(path);
  if (!out) {
    *error = "failed to open CSV output " + path;
    return false;
  }
  out << "image,width,height,jpeg_bytes,quality,chroma_subsampling,"
         "progressive_level,coding_mode,iteration,jpegli_version,total_ms";
  for (size_t phase = 0; phase < kNumDecodeApiPhases; ++phase) {
    out << ',' << ApiPhaseName(static_cast<DecodeApiPhase>(phase)) << "_ms";
  }
  for (size_t stage = 0; stage < jpegli::kNumDecodeProfileStages; ++stage) {
    const char* name =
        InternalStageName(static_cast<jpegli::DecodeProfileStage>(stage));
    out << ',' << name << "_ms," << name << "_calls";
  }
  out << ",unattributed_ms\n";
  out << std::fixed << std::setprecision(9);
  for (const DecodeStageBenchmark& benchmark : benchmarks) {
    for (size_t iteration = 0; iteration < benchmark.samples.size();
         ++iteration) {
      const DecodeStageSample& sample = benchmark.samples[iteration];
      out << CsvEscape(benchmark.image.name) << ',' << benchmark.image.width
          << ',' << benchmark.image.height << ',' << benchmark.image.jpeg.size()
          << ',' << args.quality << ',' << args.chroma_subsampling << ','
          << benchmark.progressive_level << ','
          << CodingModeName(benchmark.progressive_level) << ',' << iteration
          << ',' << kJpegliVersion << ',' << sample.total_seconds * 1e3;
      for (size_t phase = 0; phase < kNumDecodeApiPhases; ++phase) {
        out << ',' << sample.api_seconds[phase] * 1e3;
      }
      for (size_t stage = 0; stage < jpegli::kNumDecodeProfileStages; ++stage) {
        out << ',' << InternalProfileSeconds(sample.internal, stage) * 1e3
            << ',' << sample.internal.calls[stage];
      }
      const double unattributed =
          std::max(0.0, sample.total_seconds -
                            InternalProfileTotalSeconds(sample.internal));
      out << ',' << unattributed * 1e3 << '\n';
    }
  }
  if (!out) {
    *error = "failed while writing CSV output " + path;
    return false;
  }
  return true;
}

int RunDecodeStageProfile(const std::vector<fs::path>& paths,
                          const Args& args) {
  std::vector<int> progressive_levels;
  std::string error;
  if (!ParseStageProgressiveLevels(args, &progressive_levels, &error)) {
    fprintf(stderr, "%s\n", error.c_str());
    return EXIT_FAILURE;
  }

  fprintf(stderr,
          "Encoding %zu source images at q%d/%s for progressive levels %s; "
          "encoding is not timed.\n",
          paths.size(), args.quality, args.chroma_subsampling.c_str(),
          args.stage_progressive_levels.c_str());
  std::vector<DecodeStageBenchmark> benchmarks;
  benchmarks.reserve(paths.size() * progressive_levels.size());
  size_t max_output_size = 0;
  for (size_t i = 0; i < paths.size(); ++i) {
    jpegli::extras::PackedPixelFile ppf;
    if (!LoadSourceImage(paths[i], &ppf, &error)) {
      fprintf(stderr, "%s\n", error.c_str());
      return EXIT_FAILURE;
    }
    for (int level : progressive_levels) {
      DecodeStageBenchmark benchmark;
      benchmark.progressive_level = level;
      if (!EncodeSourceImage(paths[i], ppf, args.quality,
                             args.chroma_subsampling, level, &benchmark.image,
                             &error)) {
        fprintf(stderr, "%s\n", error.c_str());
        return EXIT_FAILURE;
      }
      max_output_size = std::max(max_output_size, benchmark.image.output_size);
      benchmarks.push_back(std::move(benchmark));
    }
    fprintf(stderr, "[%zu/%zu] encoded stage-profile inputs\n", i + 1,
            paths.size());
  }

  fprintf(stderr,
          "jpegli %s; validating instrumented output, then running %zu "
          "warmup and %zu timed iteration(s) per image/mode.\n",
          kJpegliVersion, args.warmups, args.iterations);
  std::vector<uint8_t> output(max_output_size);
  for (DecodeStageBenchmark& benchmark : benchmarks) {
    if (!RunStageDecode(&benchmark, output.data(), /*record=*/false, &error)) {
      fprintf(stderr, "%s: %s\n", benchmark.image.name.c_str(), error.c_str());
      return EXIT_FAILURE;
    }
    if (!HasOpaqueAlpha(output.data(), benchmark.image.output_size)) {
      fprintf(stderr, "%s: jpegli produced non-opaque RGBA output\n",
              benchmark.image.name.c_str());
      return EXIT_FAILURE;
    }
  }

  std::vector<size_t> order(benchmarks.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(0x53544147U);
  for (size_t rep = 0; rep < args.warmups; ++rep) {
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t index : order) {
      if (!RunStageDecode(&benchmarks[index], output.data(), /*record=*/false,
                          &error)) {
        fprintf(stderr, "%s: %s\n", benchmarks[index].image.name.c_str(),
                error.c_str());
        return EXIT_FAILURE;
      }
    }
  }
  for (size_t rep = 0; rep < args.iterations; ++rep) {
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t index : order) {
      if (!RunStageDecode(&benchmarks[index], output.data(), /*record=*/true,
                          &error)) {
        fprintf(stderr, "%s: %s\n", benchmarks[index].image.name.c_str(),
                error.c_str());
        return EXIT_FAILURE;
      }
    }
  }

  PrintStageResults(benchmarks, progressive_levels);
  if (!args.csv.empty()) {
    if (!WriteStageCsv(args.csv, benchmarks, args, &error)) {
      fprintf(stderr, "%s\n", error.c_str());
      return EXIT_FAILURE;
    }
    printf("Per-iteration decode-stage CSV written to %s\n", args.csv.c_str());
  }
  return EXIT_SUCCESS;
}

double Percentile(std::vector<double> values, double percentile) {
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(
      std::ceil(percentile * static_cast<double>(values.size())) - 1.0);
  return values[std::min(index, values.size() - 1)];
}

std::vector<double> ReadySamples(const std::vector<MetalDecodeSample>& samples,
                                 bool cold) {
  std::vector<double> result;
  for (const MetalDecodeSample& sample : samples) {
    if (sample.cold == cold) result.push_back(sample.ready_seconds);
  }
  return result;
}

std::vector<double> ProcessCpuSamples(
    const std::vector<MetalDecodeSample>& samples, bool cold) {
  std::vector<double> result;
  for (const MetalDecodeSample& sample : samples) {
    if (sample.cold == cold) result.push_back(sample.process_cpu_seconds);
  }
  return result;
}

std::vector<double> ProcessEnergySamples(
    const std::vector<MetalDecodeSample>& samples, bool cold) {
  std::vector<double> result;
  for (const MetalDecodeSample& sample : samples) {
    if (sample.cold == cold && sample.process_energy_available) {
      result.push_back(sample.process_energy_nj * 1e-6);
    }
  }
  return result;
}

const MetalDecodeSample& RepresentativeMetalSample(
    const std::vector<MetalDecodeSample>& samples) {
  std::vector<const MetalDecodeSample*> warm;
  for (const MetalDecodeSample& sample : samples) {
    if (!sample.cold) warm.push_back(&sample);
  }
  std::sort(warm.begin(), warm.end(),
            [](const MetalDecodeSample* a, const MetalDecodeSample* b) {
              return a->ready_seconds < b->ready_seconds;
            });
  return *warm[warm.size() / 2];
}

bool WriteMetalCsv(const std::string& path,
                   const std::vector<MetalImageBenchmark>& benchmarks,
                   const Args& args, std::string* error) {
  std::ofstream out(path);
  if (!out) {
    *error = "failed to open CSV output " + path;
    return false;
  }
  out << "image,width,height,pixels,jpeg_bytes,quality,chroma_subsampling,"
         "progressive_level,path,cold,iteration,ready_ms,total_ms,"
         "ready_process_cpu_ms,ready_process_energy_mj,"
         "ready_process_energy_available,megapixels_per_second,"
         "cpu_entropy_ms,metal_initialization_ms,coefficient_analysis_ms,"
         "coefficient_copy_ms,command_encoding_ms,submission_overhead_ms,"
         "gpu_dequant_idct_ms,gpu_upsample_color_ms,cpu_output_copy_ms,"
         "reconstruction_total_ms,jpegli_peak_memory_bytes,cpu_decoder_bytes,"
         "metal_buffer_bytes,decoder_retained_bytes,used_metal,"
         "direct_output,fallback_reason\n";
  out << std::fixed << std::setprecision(9);
  for (const MetalImageBenchmark& benchmark : benchmarks) {
    const auto write_path = [&](MetalDecodePath path_name,
                                const std::vector<MetalDecodeSample>& samples) {
      size_t warm_iteration = 0;
      for (const MetalDecodeSample& sample : samples) {
        const size_t iteration = sample.cold ? 0 : warm_iteration++;
        out << CsvEscape(benchmark.image.name) << ',' << benchmark.image.width
            << ',' << benchmark.image.height << ','
            << benchmark.image.width * benchmark.image.height << ','
            << benchmark.image.jpeg.size() << ',' << args.quality << ','
            << args.chroma_subsampling << ',' << args.progressive_level << ','
            << MetalPathName(path_name) << ',' << (sample.cold ? 1 : 0) << ','
            << iteration << ',' << sample.ready_seconds * 1e3 << ','
            << sample.total_seconds * 1e3 << ','
            << sample.process_cpu_seconds * 1e3 << ','
            << sample.process_energy_nj * 1e-6 << ','
            << (sample.process_energy_available ? 1 : 0) << ','
            << benchmark.image.width * benchmark.image.height * 1e-6 /
                   sample.ready_seconds
            << ','
            << sample.internal.nanoseconds[static_cast<size_t>(
                   jpegli::DecodeProfileStage::kEntropy)] *
                   1e-6
            << ',' << sample.metal.metal_initialization_ns * 1e-6 << ','
            << sample.metal.coefficient_analysis_ns * 1e-6 << ','
            << sample.metal.coefficient_copy_ns * 1e-6 << ','
            << sample.metal.command_encoding_ns * 1e-6 << ','
            << sample.metal.submission_overhead_ns * 1e-6 << ','
            << sample.metal.gpu_dequant_idct_ns * 1e-6 << ','
            << sample.metal.gpu_upsample_color_ns * 1e-6 << ','
            << sample.metal.cpu_output_copy_ns * 1e-6 << ','
            << sample.metal.reconstruction_total_ns * 1e-6 << ','
            << sample.jpegli_memory_bytes << ','
            << sample.metal.cpu_decoder_bytes << ','
            << sample.metal.metal_buffer_bytes << ','
            << sample.metal.decoder_retained_bytes << ','
            << sample.metal.used_metal << ',' << sample.metal.direct_output
            << ',' << CsvEscape(sample.metal.fallback_reason) << '\n';
      }
    };
    write_path(MetalDecodePath::kCpu, benchmark.cpu);
    write_path(MetalDecodePath::kMetalScanlines, benchmark.metal_scanlines);
    write_path(MetalDecodePath::kMetalDirect, benchmark.metal_direct);
  }
  if (!out) {
    *error = "failed while writing CSV output " + path;
    return false;
  }
  return true;
}

int RunAppleMetalBenchmark(std::vector<ImageBenchmark> images,
                           const Args& args) {
  if (!jpegli_apple_metal_is_available()) {
    fprintf(stderr,
            "Apple Metal is unavailable; configure with "
            "-DJPEGLI_ENABLE_APPLE_METAL=ON on an Apple GPU\n");
    return EXIT_FAILURE;
  }
  std::vector<MetalImageBenchmark> benchmarks;
  benchmarks.reserve(images.size());
  size_t max_output_size = 0;
  for (ImageBenchmark& image : images) {
    max_output_size = std::max(max_output_size, image.output_size);
    MetalImageBenchmark benchmark;
    benchmark.image = std::move(image);
    benchmarks.push_back(std::move(benchmark));
  }
  std::vector<uint8_t> cpu_output(max_output_size);
  std::vector<uint8_t> metal_output(max_output_size);
  std::vector<uint8_t> direct_output(max_output_size);
  std::string error;

  fprintf(stderr,
          "Running one independently cold decode, %zu warmup(s), and %zu "
          "warm decode(s) per path/image. Cold outputs are also checked "
          "byte-for-byte; no tolerance is accepted. Warm path order rotates "
          "to avoid a systematic position bias.\n",
          args.warmups, args.iterations);
  for (MetalImageBenchmark& benchmark : benchmarks) {
    auto run = [&](MetalDecodePath path, bool cold,
                   std::vector<MetalDecodeSample>* samples) -> bool {
      if (cold && path != MetalDecodePath::kCpu) {
        jpegli_apple_metal_release_cached_resources(
            JPEGLI_APPLE_METAL_RELEASE_ALL);
      }
      MetalDecodeSample sample;
      sample.cold = cold;
      uint8_t* output =
          path == MetalDecodePath::kMetalDirect
              ? (cold ? direct_output.data() : nullptr)
              : (path == MetalDecodePath::kCpu ? cpu_output.data()
                                               : metal_output.data());
      if (!DecodeJpegliMetalPath(benchmark.image, path, output, &sample,
                                 &error)) {
        return false;
      }
      samples->push_back(sample);
      return true;
    };
    if (!run(MetalDecodePath::kCpu, true, &benchmark.cpu) ||
        !run(MetalDecodePath::kMetalScanlines, true,
             &benchmark.metal_scanlines) ||
        !run(MetalDecodePath::kMetalDirect, true, &benchmark.metal_direct)) {
      fprintf(stderr, "%s: %s\n", benchmark.image.name.c_str(), error.c_str());
      return EXIT_FAILURE;
    }
    if (memcmp(cpu_output.data(), metal_output.data(),
               benchmark.image.output_size) != 0 ||
        memcmp(cpu_output.data(), direct_output.data(),
               benchmark.image.output_size) != 0) {
      size_t differing = 0;
      int max_diff = 0;
      for (size_t i = 0; i < benchmark.image.output_size; ++i) {
        const int metal_diff = std::abs(static_cast<int>(cpu_output[i]) -
                                        static_cast<int>(metal_output[i]));
        const int direct_diff = std::abs(static_cast<int>(cpu_output[i]) -
                                         static_cast<int>(direct_output[i]));
        const int diff = std::max(metal_diff, direct_diff);
        if (diff != 0) ++differing;
        if (diff != 0 && differing <= 16) {
          fprintf(stderr,
                  "  byte %zu (x=%zu y=%zu c=%zu): CPU=%u Metal=%u "
                  "Direct=%u\n",
                  i, (i / kOutputChannels) % benchmark.image.width,
                  (i / kOutputChannels) / benchmark.image.width,
                  i % kOutputChannels, cpu_output[i], metal_output[i],
                  direct_output[i]);
        }
        max_diff = std::max(max_diff, diff);
      }
      fprintf(stderr,
              "%s: Metal output is not pixel-identical (%zu differing bytes, "
              "max difference %d)\n",
              benchmark.image.name.c_str(), differing, max_diff);
      return EXIT_FAILURE;
    }
    const std::array<MetalDecodePath, 3> paths = {
        MetalDecodePath::kCpu, MetalDecodePath::kMetalScanlines,
        MetalDecodePath::kMetalDirect};
    const auto rotated_path = [&](size_t repetition, size_t position) {
      return paths[(repetition + position) % paths.size()];
    };
    for (size_t repetition = 0; repetition < args.warmups; ++repetition) {
      for (size_t position = 0; position < paths.size(); ++position) {
        const MetalDecodePath path = rotated_path(repetition, position);
        MetalDecodeSample warmup;
        uint8_t* output = path == MetalDecodePath::kCpu
                              ? cpu_output.data()
                              : (path == MetalDecodePath::kMetalScanlines
                                     ? metal_output.data()
                                     : nullptr);
        if (!DecodeJpegliMetalPath(benchmark.image, path, output, &warmup,
                                   &error)) {
          fprintf(stderr, "%s: %s\n", benchmark.image.name.c_str(),
                  error.c_str());
          return EXIT_FAILURE;
        }
      }
    }
    for (size_t repetition = 0; repetition < args.iterations; ++repetition) {
      for (size_t position = 0; position < paths.size(); ++position) {
        const MetalDecodePath path =
            rotated_path(args.warmups + repetition, position);
        std::vector<MetalDecodeSample>* samples =
            path == MetalDecodePath::kCpu
                ? &benchmark.cpu
                : (path == MetalDecodePath::kMetalScanlines
                       ? &benchmark.metal_scanlines
                       : &benchmark.metal_direct);
        if (!run(path, false, samples)) {
          fprintf(stderr, "%s: %s\n", benchmark.image.name.c_str(),
                  error.c_str());
          return EXIT_FAILURE;
        }
      }
    }
    fprintf(stderr, "benchmarked %s (%zux%zu)\n", benchmark.image.name.c_str(),
            benchmark.image.width, benchmark.image.height);
  }

  printf("\n%-17s %9s %10s %10s %10s %10s %10s %10s\n", "image", "MP",
         "CPU p50", "CPU p95", "Mtl p50", "Mtl p95", "Dir p50", "Dir p95");
  printf("%-17s %9s %10s %10s %10s %10s %10s %10s\n", "-----------------",
         "---------", "----------", "----------", "----------", "----------",
         "----------", "----------");
  size_t crossover_pixels = std::numeric_limits<size_t>::max();
  for (const MetalImageBenchmark& benchmark : benchmarks) {
    const std::vector<double> cpu = ReadySamples(benchmark.cpu, false);
    const std::vector<double> metal =
        ReadySamples(benchmark.metal_scanlines, false);
    const std::vector<double> direct =
        ReadySamples(benchmark.metal_direct, false);
    const std::vector<double> cpu_process =
        ProcessCpuSamples(benchmark.cpu, false);
    const std::vector<double> metal_process =
        ProcessCpuSamples(benchmark.metal_scanlines, false);
    const std::vector<double> direct_process =
        ProcessCpuSamples(benchmark.metal_direct, false);
    const std::vector<double> cpu_energy =
        ProcessEnergySamples(benchmark.cpu, false);
    const std::vector<double> metal_energy =
        ProcessEnergySamples(benchmark.metal_scanlines, false);
    const std::vector<double> direct_energy =
        ProcessEnergySamples(benchmark.metal_direct, false);
    const double cpu_p50 = Percentile(cpu, 0.50);
    const double metal_p50 = Percentile(metal, 0.50);
    const size_t pixels = benchmark.image.width * benchmark.image.height;
    const double cpu_p95 = Percentile(cpu, 0.95);
    const double metal_p95 = Percentile(metal, 0.95);
    if (metal_p50 < cpu_p50 && metal_p95 < cpu_p95) {
      crossover_pixels = std::min(crossover_pixels, pixels);
    }
    printf("%-17s %9.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
           ShortName(benchmark.image.name).c_str(), pixels / 1e6, cpu_p50 * 1e3,
           cpu_p95 * 1e3, metal_p50 * 1e3, metal_p95 * 1e3,
           Percentile(direct, 0.50) * 1e3, Percentile(direct, 0.95) * 1e3);

    const MetalDecodeSample& stage =
        RepresentativeMetalSample(benchmark.metal_scanlines);
    const MetalDecodeSample& cold_metal = benchmark.metal_scanlines.front();
    const MetalDecodeSample& cpu_stage =
        RepresentativeMetalSample(benchmark.cpu);
    std::string energy_summary = "unavailable";
    if (!cpu_energy.empty() && !metal_energy.empty() &&
        !direct_energy.empty()) {
      std::ostringstream stream;
      stream << std::fixed << std::setprecision(3)
             << Percentile(cpu_energy, 0.50) << '/'
             << Percentile(metal_energy, 0.50) << '/'
             << Percentile(direct_energy, 0.50) << " mJ";
      energy_summary = stream.str();
    }
    printf(
        "  cold CPU/Metal/direct: %.3f / %.3f / %.3f ms; entropy %.3f ms; "
        "cold init %.3f ms; analysis %.3f ms; coeff copy %.3f ms; "
        "GPU IDCT %.3f ms; GPU upsample/color %.3f ms; readback copy "
        "%.3f ms; ready process CPU %.3f/%.3f/%.3f ms; process energy "
        "%s; %.1f/%.1f/%.1f "
        "MP/s "
        "CPU/Metal/direct; CPU peak "
        "%.2f MiB; Metal decoder CPU/buffers/total %.2f/%.2f/%.2f MiB\n",
        ReadySamples(benchmark.cpu, true)[0] * 1e3,
        ReadySamples(benchmark.metal_scanlines, true)[0] * 1e3,
        ReadySamples(benchmark.metal_direct, true)[0] * 1e3,
        stage.internal.nanoseconds[static_cast<size_t>(
            jpegli::DecodeProfileStage::kEntropy)] *
            1e-6,
        cold_metal.metal.metal_initialization_ns * 1e-6,
        stage.metal.coefficient_analysis_ns * 1e-6,
        stage.metal.coefficient_copy_ns * 1e-6,
        stage.metal.gpu_dequant_idct_ns * 1e-6,
        stage.metal.gpu_upsample_color_ns * 1e-6,
        stage.metal.cpu_output_copy_ns * 1e-6,
        Percentile(cpu_process, 0.50) * 1e3,
        Percentile(metal_process, 0.50) * 1e3,
        Percentile(direct_process, 0.50) * 1e3, energy_summary.c_str(),
        pixels * 1e-6 / Percentile(cpu, 0.50),
        pixels * 1e-6 / Percentile(metal, 0.50),
        pixels * 1e-6 / Percentile(direct, 0.50),
        cpu_stage.jpegli_memory_bytes / (1024.0 * 1024.0),
        stage.metal.cpu_decoder_bytes / (1024.0 * 1024.0),
        stage.metal.metal_buffer_bytes / (1024.0 * 1024.0),
        stage.metal.decoder_retained_bytes / (1024.0 * 1024.0));
  }
  if (crossover_pixels == std::numeric_limits<size_t>::max()) {
    printf("\nNo CPU/Metal crossover was observed in this corpus.\n");
  } else {
    printf(
        "\nMeasured p50/p95 crossover policy candidate: %zu pixels "
        "(%.3f MP).\n",
        crossover_pixels, crossover_pixels / 1e6);
  }
  printf(
      "All Metal scanline and direct-output bytes matched CPU jpegli "
      "exactly.\n");
  if (!args.csv.empty() && !WriteMetalCsv(args.csv, benchmarks, args, &error)) {
    fprintf(stderr, "%s\n", error.c_str());
    return EXIT_FAILURE;
  }
  if (!args.csv.empty())
    printf("Raw Metal benchmark CSV written to %s\n", args.csv.c_str());
  return EXIT_SUCCESS;
}

bool ValidateArgs(const Args& args) {
  const int special_modes = static_cast<int>(args.perceptual_quality) +
                            static_cast<int>(args.decode_stage_profile) +
                            static_cast<int>(args.apple_metal_benchmark);
  if (special_modes > 1) {
    fprintf(stderr,
            "--perceptual_quality, --decode_stage_profile, and "
            "--apple_metal_benchmark are mutually exclusive\n");
    return false;
  }
  if (args.quality < 1 || args.quality > 100) {
    fprintf(stderr, "--quality must be in the range 1..100\n");
    return false;
  }
  if (!IsValidSubsampling(args.chroma_subsampling)) {
    fprintf(stderr,
            "--chroma_subsampling must be one of 444, 440, 422, or 420\n");
    return false;
  }
  if (args.progressive_level < 0 || args.progressive_level > 2) {
    fprintf(stderr, "--progressive_level must be in the range 0..2\n");
    return false;
  }
  if (args.iterations == 0) {
    fprintf(stderr, "--iterations must be greater than zero\n");
    return false;
  }
  return true;
}

int DecodeBenchmarkMain(int argc, const char* argv[]) {
  Args args;
  CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);
  if (!cmdline.Parse(argc, argv)) {
    fprintf(stderr, "Use '%s -h' for more information.\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (cmdline.HelpFlagPassed() || args.input == nullptr) {
    cmdline.PrintHelp();
    return cmdline.HelpFlagPassed() ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  if (!ValidateArgs(args)) return EXIT_FAILURE;

  std::vector<fs::path> paths;
  std::string error;
  if (!CollectSourceImages(args.input, args.limit, &paths, &error)) {
    fprintf(stderr, "%s\n", error.c_str());
    return EXIT_FAILURE;
  }
  if (args.decode_stage_profile) {
    return RunDecodeStageProfile(paths, args);
  }
  if (args.perceptual_quality) {
    return RunPerceptualQualitySweep(paths, args);
  }

  if (!args.jpeg_dir.empty()) {
    std::error_code ec;
    fs::create_directories(args.jpeg_dir, ec);
    if (ec) {
      fprintf(stderr, "failed to create JPEG directory %s: %s\n",
              args.jpeg_dir.c_str(), ec.message().c_str());
      return EXIT_FAILURE;
    }
  }

  fprintf(stderr,
          "Encoding %zu source images with jpegli (q%d, %s, p%d); encoding "
          "is not timed.\n",
          paths.size(), args.quality, args.chroma_subsampling.c_str(),
          args.progressive_level);
  fprintf(stderr, "jpegli %s; TurboJPEG API %d.%d.%d\n", kJpegliVersion,
          TURBOJPEG_VERSION_NUMBER / 1000000,
          (TURBOJPEG_VERSION_NUMBER / 1000) % 1000,
          TURBOJPEG_VERSION_NUMBER % 1000);
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  fprintf(stderr, "Apple ImageIO enabled\n");
#endif
  std::vector<ImageBenchmark> images;
  images.reserve(paths.size());
  size_t max_output_size = 0;
  for (size_t i = 0; i < paths.size(); ++i) {
    ImageBenchmark image;
    if (!EncodeImage(paths[i], args, &image, &error)) {
      fprintf(stderr, "%s\n", error.c_str());
      return EXIT_FAILURE;
    }
    fprintf(stderr, "[%zu/%zu] %s: %zux%zu -> %.1f KiB\n", i + 1, paths.size(),
            image.name.c_str(), image.width, image.height,
            image.jpeg.size() / 1024.0);
    max_output_size = std::max(max_output_size, image.output_size);
    if (!args.jpeg_dir.empty()) {
      fs::path jpeg_path = fs::path(args.jpeg_dir) / paths[i].filename();
      jpeg_path.replace_extension(".jpg");
      if (!WriteFile(jpeg_path.string(), image.jpeg)) {
        fprintf(stderr, "failed to save %s\n", jpeg_path.string().c_str());
        return EXIT_FAILURE;
      }
    }
    images.push_back(std::move(image));
  }

  if (args.apple_metal_benchmark) {
    return RunAppleMetalBenchmark(std::move(images), args);
  }

  std::vector<uint8_t> jpegli_output(max_output_size);
  std::vector<uint8_t> turbojpeg_output(max_output_size);
  std::vector<uint8_t> imageio_output(max_output_size);
  fprintf(stderr,
          "Validating all decoders, then running %zu warmup and %zu timed "
          "iteration(s) per image.\n",
          args.warmups, args.iterations);

  std::vector<Decoder> decoder_order = {Decoder::kJpegli, Decoder::kTurboJPEG};
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
  decoder_order.push_back(Decoder::kAppleImageIO);
#endif

  // This initial pass verifies all decode paths and records how their pixel
  // outputs differ. It is deliberately outside the warmup and timed samples.
  for (size_t i = 0; i < images.size(); ++i) {
    if (!RunDecoders(&images[i], decoder_order, /*record=*/false,
                     jpegli_output.data(), turbojpeg_output.data(),
                     imageio_output.data(), &error)) {
      fprintf(stderr, "%s: %s\n", images[i].name.c_str(), error.c_str());
      return EXIT_FAILURE;
    }
    if (!HasOpaqueAlpha(jpegli_output.data(), images[i].output_size) ||
        !HasOpaqueAlpha(turbojpeg_output.data(), images[i].output_size)
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
        || !HasOpaqueAlpha(imageio_output.data(), images[i].output_size)
#endif
    ) {
      fprintf(stderr, "%s: a decoder produced non-opaque RGBA output\n",
              images[i].name.c_str());
      return EXIT_FAILURE;
    }
    images[i].jpegli_checksum =
        Checksum(jpegli_output.data(), images[i].output_size);
    images[i].turbojpeg_checksum =
        Checksum(turbojpeg_output.data(), images[i].output_size);
    MeasureOutputDifference(
        jpegli_output.data(), turbojpeg_output.data(), images[i].output_size,
        &images[i].turbojpeg_mean_abs_diff, &images[i].turbojpeg_max_abs_diff);
#if defined(JPEGLI_HAVE_APPLE_IMAGEIO)
    images[i].imageio_checksum =
        Checksum(imageio_output.data(), images[i].output_size);
    MeasureOutputDifference(
        jpegli_output.data(), imageio_output.data(), images[i].output_size,
        &images[i].imageio_mean_abs_diff, &images[i].imageio_max_abs_diff);
#endif
  }

  std::vector<size_t> order(images.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(0x4a504547U);
  for (size_t rep = 0; rep < args.warmups; ++rep) {
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t position = 0; position < order.size(); ++position) {
      const size_t i = order[position];
      std::shuffle(decoder_order.begin(), decoder_order.end(), rng);
      if (!RunDecoders(&images[i], decoder_order, /*record=*/false,
                       jpegli_output.data(), turbojpeg_output.data(),
                       imageio_output.data(), &error)) {
        fprintf(stderr, "%s: %s\n", images[i].name.c_str(), error.c_str());
        return EXIT_FAILURE;
      }
    }
  }

  for (size_t rep = 0; rep < args.iterations; ++rep) {
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t position = 0; position < order.size(); ++position) {
      const size_t i = order[position];
      std::shuffle(decoder_order.begin(), decoder_order.end(), rng);
      if (!RunDecoders(&images[i], decoder_order, /*record=*/true,
                       jpegli_output.data(), turbojpeg_output.data(),
                       imageio_output.data(), &error)) {
        fprintf(stderr, "%s: %s\n", images[i].name.c_str(), error.c_str());
        return EXIT_FAILURE;
      }
    }
  }

  PrintResults(images);
  if (!args.csv.empty()) {
    if (!WriteCsv(args.csv, images, args, &error)) {
      fprintf(stderr, "%s\n", error.c_str());
      return EXIT_FAILURE;
    }
    printf("Per-image CSV written to %s\n", args.csv.c_str());
  }
  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace jpegli_tools

int main(int argc, const char* argv[]) {
  return jpegli_tools::DecodeBenchmarkMain(argc, argv);
}

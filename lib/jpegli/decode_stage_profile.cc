// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jpegli/decode_stage_profile_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>

#include "lib/jpegli/decode_internal.h"

namespace jpegli {
namespace {

struct TimerCalibration {
  bool supported;
  uint64_t overhead_ticks;
  double ticks_per_second;
};

TimerCalibration CalibrateTimer() {
  char cpu100[100];
  if (!hwy::platform::HaveTimerStop(cpu100)) return {false, 0, 0.0};
  uint64_t best = std::numeric_limits<uint64_t>::max();
  for (int i = 0; i < 2048; ++i) {
    const uint64_t start = hwy::timer::Start();
    const uint64_t stop = hwy::timer::Stop();
    best = std::min(best, stop - start);
  }
  return {true, best, hwy::platform::InvariantTicksPerSecond()};
}

const TimerCalibration& GetTimerCalibration() {
  static const TimerCalibration calibration = CalibrateTimer();
  return calibration;
}

}  // namespace

bool EnableDecodeStageProfiling(j_decompress_ptr cinfo,
                                jpegli_decode_stage_profile* profile) {
#if JPEGLI_ENABLE_DECODE_STAGE_PROFILING
  if (cinfo == nullptr || cinfo->master == nullptr || profile == nullptr) {
    return false;
  }
  memset(profile, 0, sizeof(*profile));
  profile->abi_version = JPEGLI_DECODE_STAGE_PROFILE_ABI_VERSION;
  profile->struct_size = sizeof(*profile);
  const TimerCalibration& calibration = GetTimerCalibration();
  if (!calibration.supported) return false;
  profile->timer_overhead_ticks = calibration.overhead_ticks;
  profile->timer_ticks_per_second = calibration.ticks_per_second;
  cinfo->master->decode_stage_profile = profile;
  cinfo->master->decode_stage_profile_start_ns = 0;
  return true;
#else
  static_cast<void>(cinfo);
  static_cast<void>(profile);
  return false;
#endif
}

void DecodeStageProfileStart(jpeg_decomp_master* m) {
#if JPEGLI_ENABLE_DECODE_STAGE_PROFILING
  if (m->decode_stage_profile == nullptr ||
      m->decode_stage_profile_start_ns != 0) {
    return;
  }
  m->decode_stage_profile_start_ns = DecodeStageProfileNowNanos();
#else
  static_cast<void>(m);
#endif
}

void DecodeStageProfileFinish(jpeg_decomp_master* m) {
#if JPEGLI_ENABLE_DECODE_STAGE_PROFILING
  if (m->decode_stage_profile == nullptr ||
      m->decode_stage_profile_start_ns == 0) {
    return;
  }
  m->decode_stage_profile->decoder_elapsed_ns =
      DecodeStageProfileNowNanos() - m->decode_stage_profile_start_ns;
#else
  static_cast<void>(m);
#endif
}

}  // namespace jpegli

// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef JPEGLI_LIB_JPEGLI_DECODE_STAGE_PROFILE_INTERNAL_H_
#define JPEGLI_LIB_JPEGLI_DECODE_STAGE_PROFILE_INTERNAL_H_

#include <chrono>
#include <cstdint>

#include <hwy/timer.h>

#include "lib/jpegli/decode_stage_profile.h"

#ifndef JPEGLI_ENABLE_DECODE_STAGE_PROFILING
#define JPEGLI_ENABLE_DECODE_STAGE_PROFILING 0
#endif

namespace jpegli {

inline uint64_t DecodeStageProfileNowNanos() {
  using Clock = std::chrono::steady_clock;
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch())
          .count());
}

class DecodeStageProfileTimer {
 public:
  DecodeStageProfileTimer(jpegli_decode_stage_profile* profile,
                          jpegli_decode_stage_profile_stage stage)
#if JPEGLI_ENABLE_DECODE_STAGE_PROFILING
      : profile_(profile), entry_(nullptr), start_ticks_(0)
#endif
  {
#if JPEGLI_ENABLE_DECODE_STAGE_PROFILING
    if (profile_ == nullptr) return;
    entry_ = &profile_->stages[stage];
    ++entry_->observations;
    start_ticks_ = hwy::timer::Start();
#else
    static_cast<void>(profile);
    static_cast<void>(stage);
#endif
  }

  ~DecodeStageProfileTimer() {
#if JPEGLI_ENABLE_DECODE_STAGE_PROFILING
    if (entry_ == nullptr) return;
    const uint64_t stop_ticks = hwy::timer::Stop();
    const uint64_t elapsed_ticks = stop_ticks - start_ticks_;
    entry_->elapsed_ticks +=
        elapsed_ticks > profile_->timer_overhead_ticks
            ? elapsed_ticks - profile_->timer_overhead_ticks
            : 0;
#endif
  }

  DecodeStageProfileTimer(const DecodeStageProfileTimer&) = delete;
  DecodeStageProfileTimer& operator=(const DecodeStageProfileTimer&) = delete;

 private:
#if JPEGLI_ENABLE_DECODE_STAGE_PROFILING
  jpegli_decode_stage_profile* profile_;
  jpegli_decode_stage_profile_entry* entry_;
  uint64_t start_ticks_;
#endif
};

bool EnableDecodeStageProfiling(j_decompress_ptr cinfo,
                                jpegli_decode_stage_profile* profile);
void DecodeStageProfileStart(jpeg_decomp_master* m);
void DecodeStageProfileFinish(jpeg_decomp_master* m);

}  // namespace jpegli

#endif  // JPEGLI_LIB_JPEGLI_DECODE_STAGE_PROFILE_INTERNAL_H_

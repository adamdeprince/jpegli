// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "lib/jpegli/decode_scan.h"

#include <algorithm>
#include <cstring>
#include <hwy/base.h>  // HWY_ALIGN_MAX
#include <limits>

#include "lib/base/status.h"
#include "lib/jpegli/common.h"
#include "lib/jpegli/common_internal.h"
#include "lib/jpegli/decode_internal.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/huffman.h"

namespace jpegli {
namespace {

// Max 14 block per MCU (when 1 channel is subsampled)
// Max 64 nonzero coefficients per block
// Max 16 symbol bits plus 11 extra bits per nonzero symbol
// Max 2 bytes per 8 bits (worst case is all bytes are escaped 0xff)
constexpr int kMaxMCUByteSize = 6048;

// Helper structure to read bits from the entropy coded data segment.
struct BitReaderState {
  BitReaderState(const uint8_t* data, const size_t len, size_t pos)
      : data_(data), len_(len), start_pos_(pos) {
    Reset(pos);
  }

  void Reset(size_t pos) {
    pos_ = pos;
    val_ = 0;
    bits_left_ = 0;
    next_marker_pos_ = len_;
    FillBitWindow();
  }

  // Returns the next byte and skips the 0xff/0x00 escape sequences.
  uint8_t GetNextByte() {
    if (pos_ >= next_marker_pos_) {
      ++pos_;
      return 0;
    }
    uint8_t c = data_[pos_++];
    if (c == 0xff) {
      uint8_t escape = pos_ < len_ ? data_[pos_] : 0;
      if (escape == 0) {
        ++pos_;
      } else {
        // 0xff was followed by a non-zero byte, which means that we found the
        // start of the next marker segment.
        next_marker_pos_ = pos_ - 1;
      }
    }
    return c;
  }

  void FillBitWindow() {
    if (bits_left_ <= 16) {
      while (bits_left_ <= 56) {
        val_ <<= 8;
        val_ |= static_cast<uint64_t>(GetNextByte());
        bits_left_ += 8;
      }
    }
  }

  int ReadBits(int nbits) {
    FillBitWindow();
    uint64_t val = (val_ >> (bits_left_ - nbits)) & ((1ULL << nbits) - 1);
    bits_left_ -= nbits;
    return val;
  }

  // Sets *pos to the next stream position, and *bit_pos to the bit position
  // within the next byte where parsing should continue.
  // Returns false if the stream ended too early.
  bool FinishStream(size_t* pos, size_t* bit_pos) {
    *bit_pos = (8 - (bits_left_ & 7)) & 7;
    // Give back some bytes that we did not use.
    int unused_bytes_left = DivCeil(bits_left_, 8);
    while (unused_bytes_left-- > 0) {
      --pos_;
      // If we give back a 0 byte, we need to check if it was a 0xff/0x00 escape
      // sequence, and if yes, we need to give back one more byte.
      if (((pos_ == len_ && pos_ == next_marker_pos_) ||
           (pos_ > 0 && pos_ < next_marker_pos_ && data_[pos_] == 0)) &&
          (data_[pos_ - 1] == 0xff)) {
        --pos_;
      }
    }
    if (pos_ >= next_marker_pos_) {
      *pos = next_marker_pos_;
      if (pos_ > next_marker_pos_ || *bit_pos > 0) {
        // Data ran out before the scan was complete.
        return false;
      }
    }
    *pos = pos_;
    return true;
  }

  const uint8_t* data_;
  const size_t len_;
  size_t pos_;
  uint64_t val_;
  int bits_left_;
  size_t next_marker_pos_;
  size_t start_pos_;
};

// Returns the next Huffman-coded symbol.
int ReadSymbol(const HuffmanTableEntry* table, BitReaderState* br) {
  int nbits;
  br->FillBitWindow();
  int val = (br->val_ >> (br->bits_left_ - 8)) & 0xff;
  table += val;
  nbits = table->bits - 8;
  if (nbits > 0) {
    br->bits_left_ -= 8;
    table += table->value;
    val = (br->val_ >> (br->bits_left_ - nbits)) & ((1 << nbits) - 1);
    table += val;
  }
  br->bits_left_ -= table->bits;
  return table->value;
}

/**
 * Returns the DC diff or AC value for extra bits value x and prefix code s.
 *
 * CCITT Rec. T.81 (1992 E)
 * Table F.1 – Difference magnitude categories for DC coding
 *  SSSS | DIFF values
 * ------+--------------------------
 *     0 | 0
 *     1 | –1, 1
 *     2 | –3, –2, 2, 3
 *     3 | –7..–4, 4..7
 * ......|..........................
 *    11 | –2047..–1024, 1024..2047
 *
 * CCITT Rec. T.81 (1992 E)
 * Table F.2 – Categories assigned to coefficient values
 * [ Same as Table F.1, but does not include SSSS equal to 0 and 11]
 *
 *
 * CCITT Rec. T.81 (1992 E)
 * F.1.2.1.1 Structure of DC code table
 * For each category,... additional bits... appended... to uniquely identify
 * which difference... occurred... When DIFF is positive... SSSS... bits of DIFF
 * are appended. When DIFF is negative... SSSS... bits of (DIFF – 1) are
 * appended... Most significant bit... is 0 for negative differences and 1 for
 * positive differences.
 *
 * In other words the upper half of extra bits range represents DIFF as is.
 * The lower half represents the negative DIFFs with an offset.
 */
int HuffExtend(int x, int s) {
  JPEGLI_DASSERT(s > 0);
  int half = 1 << (s - 1);
  if (x >= half) {
    JPEGLI_DASSERT(x < (1 << s));
    return x;
  } else {
    return x - (1 << s) + 1;
  }
}

constexpr size_t kInvalidAmdDecodeBlock = std::numeric_limits<size_t>::max();

bool RecordsAmdDecodeEvents(const jpeg_decomp_master* m, size_t block_index) {
  return m->amd_decode_coefficients_active_ &&
         block_index != kInvalidAmdDecodeBlock;
}

void RecordAmdDecodeDelta(jpeg_decomp_master* m, size_t block_index,
                          int coefficient, int delta) {
  if (delta == 0 || !RecordsAmdDecodeEvents(m, block_index)) return;
  const size_t coefficient_index = block_index * DCTSIZE2 + coefficient;
  JPEGLI_DASSERT(coefficient_index < m->amd_decode_total_coefficients_);
  JPEGLI_DASSERT(coefficient_index <= std::numeric_limits<uint32_t>::max());
  m->amd_decode_coefficient_events_.push_back(
      {static_cast<uint32_t>(coefficient_index), delta});
}

void RecordAmdDecodeInitial(jpeg_decomp_master* m, size_t block_index,
                            int coefficient, int value) {
  if (value == 0 || !RecordsAmdDecodeEvents(m, block_index)) return;
  const uint64_t bit = uint64_t{1} << coefficient;
  m->amd_decode_significant_[block_index] |= bit;
  if (value < 0) {
    m->amd_decode_negative_[block_index] |= bit;
  }
  RecordAmdDecodeDelta(m, block_index, coefficient, value);
}

bool AmdDecodeCoefficientIsSignificant(const jpeg_decomp_master* m,
                                       size_t block_index, int coefficient) {
  JPEGLI_DASSERT(RecordsAmdDecodeEvents(m, block_index));
  return (m->amd_decode_significant_[block_index] >> coefficient) & 1u;
}

void RecordAmdDecodeACRefinement(jpeg_decomp_master* m, size_t block_index,
                                 int coefficient, int magnitude) {
  const uint64_t bit = uint64_t{1} << coefficient;
  const int delta = (m->amd_decode_negative_[block_index] & bit) != 0
                        ? -magnitude
                        : magnitude;
  RecordAmdDecodeDelta(m, block_index, coefficient, delta);
}

// Keep the stock coefficient path isolated from all experimental bookkeeping.
// This is intentionally the original decoder loop so disabling the GPU path
// does not add a mode check for every progressive coefficient.
bool DecodeDCTBlockCpu(const HuffmanTableEntry* dc_huff,
                       const HuffmanTableEntry* ac_huff, int Ss, int Se, int Al,
                       int* eobrun, BitReaderState* br, coeff_t* last_dc_coeff,
                       coeff_t* coeffs) {
  int Am = 1 << Al;
  bool eobrun_allowed = Ss > 0;
  if (Ss == 0) {
    int s = ReadSymbol(dc_huff, br);
    if (s >= kJpegDCAlphabetSize) return false;
    int diff = 0;
    if (s > 0) {
      int bits = br->ReadBits(s);
      diff = HuffExtend(bits, s);
    }
    int coeff = diff + *last_dc_coeff;
    const int dc_coeff = coeff * Am;
    coeffs[0] = dc_coeff;
    if (dc_coeff != coeffs[0]) return false;
    *last_dc_coeff = coeff;
    ++Ss;
  }
  if (Ss > Se) return true;
  if (*eobrun > 0) {
    --(*eobrun);
    return true;
  }
  for (int k = Ss; k <= Se; k++) {
    int sr = ReadSymbol(ac_huff, br);
    if (sr >= kJpegHuffmanAlphabetSize) return false;
    int r = sr >> 4;
    int s = sr & 15;
    if (s > 0) {
      k += r;
      if (k > Se || s + Al >= kJpegDCAlphabetSize) return false;
      int bits = br->ReadBits(s);
      int coeff = HuffExtend(bits, s);
      coeffs[kJPEGNaturalOrder[k]] = coeff * Am;
    } else if (r == 15) {
      k += 15;
    } else {
      *eobrun = 1 << r;
      if (r > 0) {
        if (!eobrun_allowed) return false;
        *eobrun += br->ReadBits(r);
      }
      break;
    }
  }
  --(*eobrun);
  return true;
}

bool RefineDCTBlockCpu(const HuffmanTableEntry* ac_huff, int Ss, int Se, int Al,
                       int* eobrun, BitReaderState* br, coeff_t* coeffs) {
  int Am = 1 << Al;
  bool eobrun_allowed = Ss > 0;
  if (Ss == 0) {
    int s = br->ReadBits(1);
    coeff_t dc_coeff = coeffs[0];
    dc_coeff |= s * Am;
    coeffs[0] = dc_coeff;
    ++Ss;
  }
  if (Ss > Se) return true;
  int p1 = Am;
  int m1 = -Am;
  int k = Ss;
  int r;
  int s;
  bool in_zero_run = false;
  if (*eobrun <= 0) {
    for (; k <= Se; k++) {
      s = ReadSymbol(ac_huff, br);
      if (s >= kJpegHuffmanAlphabetSize) return false;
      r = s >> 4;
      s &= 15;
      if (s) {
        if (s != 1) return false;
        s = br->ReadBits(1) ? p1 : m1;
        in_zero_run = false;
      } else {
        if (r != 15) {
          *eobrun = 1 << r;
          if (r > 0) {
            if (!eobrun_allowed) return false;
            *eobrun += br->ReadBits(r);
          }
          break;
        }
        in_zero_run = true;
      }
      do {
        coeff_t thiscoef = coeffs[kJPEGNaturalOrder[k]];
        if (thiscoef != 0) {
          if (br->ReadBits(1)) {
            if ((thiscoef & p1) == 0) {
              if (thiscoef >= 0) {
                thiscoef += p1;
              } else {
                thiscoef += m1;
              }
            }
          }
          coeffs[kJPEGNaturalOrder[k]] = thiscoef;
        } else if (--r < 0) {
          break;
        }
        k++;
      } while (k <= Se);
      if (s) {
        if (k > Se) return false;
        coeffs[kJPEGNaturalOrder[k]] = s;
      }
    }
  }
  if (in_zero_run) return false;
  if (*eobrun > 0) {
    for (; k <= Se; k++) {
      coeff_t thiscoef = coeffs[kJPEGNaturalOrder[k]];
      if (thiscoef != 0) {
        if (br->ReadBits(1)) {
          if ((thiscoef & p1) == 0) {
            if (thiscoef >= 0) {
              thiscoef += p1;
            } else {
              thiscoef += m1;
            }
          }
        }
        coeffs[kJPEGNaturalOrder[k]] = thiscoef;
      }
    }
  }
  --(*eobrun);
  return true;
}

// Decodes one 8x8 block of DCT coefficients from the bit stream.
template <bool kRecordEvents>
bool DecodeDCTBlock(const HuffmanTableEntry* dc_huff,
                    const HuffmanTableEntry* ac_huff, int Ss, int Se, int Al,
                    int* eobrun, BitReaderState* br, coeff_t* last_dc_coeff,
                    coeff_t* coeffs, jpeg_decomp_master* m,
                    size_t block_index) {
  // Nowadays multiplication is even faster than variable shift.
  int Am = 1 << Al;
  bool eobrun_allowed = Ss > 0;
  if (Ss == 0) {
    int s = ReadSymbol(dc_huff, br);
    if (s >= kJpegDCAlphabetSize) {
      return false;
    }
    int diff = 0;
    if (s > 0) {
      int bits = br->ReadBits(s);
      diff = HuffExtend(bits, s);
    }
    int coeff = diff + *last_dc_coeff;
    const int dc_coeff = coeff * Am;
    if (dc_coeff < std::numeric_limits<coeff_t>::min() ||
        dc_coeff > std::numeric_limits<coeff_t>::max()) {
      return false;
    }
    if constexpr (kRecordEvents) {
      RecordAmdDecodeInitial(m, block_index, 0, dc_coeff);
    } else {
      coeffs[0] = dc_coeff;
    }
    *last_dc_coeff = coeff;
    ++Ss;
  }
  if (Ss > Se) {
    return true;
  }
  if (*eobrun > 0) {
    --(*eobrun);
    return true;
  }
  for (int k = Ss; k <= Se; k++) {
    int sr = ReadSymbol(ac_huff, br);
    if (sr >= kJpegHuffmanAlphabetSize) {
      return false;
    }
    int r = sr >> 4;
    int s = sr & 15;
    if (s > 0) {
      k += r;
      if (k > Se) {
        return false;
      }
      if (s + Al >= kJpegDCAlphabetSize) {
        return false;
      }
      int bits = br->ReadBits(s);
      int coeff = HuffExtend(bits, s);
      const int coefficient = kJPEGNaturalOrder[k];
      const int value = coeff * Am;
      if constexpr (kRecordEvents) {
        RecordAmdDecodeInitial(m, block_index, coefficient, value);
      } else {
        coeffs[coefficient] = value;
      }
    } else if (r == 15) {
      k += 15;
    } else {
      *eobrun = 1 << r;
      if (r > 0) {
        if (!eobrun_allowed) {
          return false;
        }
        *eobrun += br->ReadBits(r);
      }
      break;
    }
  }
  --(*eobrun);
  return true;
}

template <bool kRecordEvents>
bool RefineDCTBlock(const HuffmanTableEntry* ac_huff, int Ss, int Se, int Al,
                    int* eobrun, BitReaderState* br, coeff_t* coeffs,
                    jpeg_decomp_master* m, size_t block_index) {
  // Nowadays multiplication is even faster than variable shift.
  int Am = 1 << Al;
  bool eobrun_allowed = Ss > 0;
  if (Ss == 0) {
    int s = br->ReadBits(1);
    if constexpr (kRecordEvents) {
      if (s != 0) RecordAmdDecodeDelta(m, block_index, 0, Am);
    } else {
      coeff_t dc_coeff = coeffs[0];
      dc_coeff |= s * Am;
      coeffs[0] = dc_coeff;
    }
    ++Ss;
  }
  if (Ss > Se) {
    return true;
  }
  int p1 = Am;
  int m1 = -Am;
  int k = Ss;
  int r;
  int s;
  bool in_zero_run = false;
  if (*eobrun <= 0) {
    for (; k <= Se; k++) {
      s = ReadSymbol(ac_huff, br);
      if (s >= kJpegHuffmanAlphabetSize) {
        return false;
      }
      r = s >> 4;
      s &= 15;
      if (s) {
        if (s != 1) {
          return false;
        }
        s = br->ReadBits(1) ? p1 : m1;
        in_zero_run = false;
      } else {
        if (r != 15) {
          *eobrun = 1 << r;
          if (r > 0) {
            if (!eobrun_allowed) {
              return false;
            }
            *eobrun += br->ReadBits(r);
          }
          break;
        }
        in_zero_run = true;
      }
      do {
        const int coefficient = kJPEGNaturalOrder[k];
        bool significant;
        if constexpr (kRecordEvents) {
          significant =
              AmdDecodeCoefficientIsSignificant(m, block_index, coefficient);
        } else {
          significant = coeffs[coefficient] != 0;
        }
        if (significant) {
          if (br->ReadBits(1) != 0) {
            if constexpr (kRecordEvents) {
              RecordAmdDecodeACRefinement(m, block_index, coefficient, p1);
            } else {
              coeff_t thiscoef = coeffs[coefficient];
              if ((thiscoef & p1) == 0) {
                if (thiscoef >= 0) {
                  thiscoef += p1;
                } else {
                  thiscoef += m1;
                }
              }
              coeffs[coefficient] = thiscoef;
            }
          }
        } else {
          if (--r < 0) {
            break;
          }
        }
        k++;
      } while (k <= Se);
      if (s) {
        if (k > Se) {
          return false;
        }
        const int coefficient = kJPEGNaturalOrder[k];
        if constexpr (kRecordEvents) {
          RecordAmdDecodeInitial(m, block_index, coefficient, s);
        } else {
          coeffs[coefficient] = s;
        }
      }
    }
  }
  if (in_zero_run) {
    return false;
  }
  if (*eobrun > 0) {
    for (; k <= Se; k++) {
      const int coefficient = kJPEGNaturalOrder[k];
      bool significant;
      if constexpr (kRecordEvents) {
        significant =
            AmdDecodeCoefficientIsSignificant(m, block_index, coefficient);
      } else {
        significant = coeffs[coefficient] != 0;
      }
      if (significant) {
        if (br->ReadBits(1) != 0) {
          if constexpr (kRecordEvents) {
            RecordAmdDecodeACRefinement(m, block_index, coefficient, p1);
          } else {
            coeff_t thiscoef = coeffs[coefficient];
            if ((thiscoef & p1) == 0) {
              if (thiscoef >= 0) {
                thiscoef += p1;
              } else {
                thiscoef += m1;
              }
            }
            coeffs[coefficient] = thiscoef;
          }
        }
      }
    }
  }
  --(*eobrun);
  return true;
}

void SaveMCUCodingState(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  memcpy(m->mcu_.last_dc_coeff, m->last_dc_coeff_, sizeof(m->last_dc_coeff_));
  m->mcu_.eobrun = m->eobrun_;
  if (m->amd_decode_coefficients_active_) {
    m->mcu_.amd_decode_event_count = m->amd_decode_coefficient_events_.size();
    m->mcu_.amd_decode_num_blocks = 0;
    for (int i = 0; i < cinfo->comps_in_scan; ++i) {
      const jpeg_component_info* comp = cinfo->cur_comp_info[i];
      const int c = comp->component_index;
      const size_t block_x = m->scan_mcu_col_ * comp->MCU_width;
      for (int iy = 0; iy < comp->MCU_height; ++iy) {
        const size_t block_y = m->scan_mcu_row_ * comp->MCU_height + iy;
        if (block_y >= comp->height_in_blocks) continue;
        const size_t nblocks =
            std::min<size_t>(comp->MCU_width, comp->width_in_blocks - block_x);
        for (size_t ix = 0; ix < nblocks; ++ix) {
          const size_t saved = m->mcu_.amd_decode_num_blocks;
          JPEGLI_DASSERT(saved < D_MAX_BLOCKS_IN_MCU);
          ++m->mcu_.amd_decode_num_blocks;
          const size_t block_index = m->amd_decode_component_block_offsets_[c] +
                                     block_y * comp->width_in_blocks + block_x +
                                     ix;
          m->mcu_.amd_decode_block_index[saved] = block_index;
          m->mcu_.amd_decode_significant[saved] =
              m->amd_decode_significant_[block_index];
          m->mcu_.amd_decode_negative[saved] =
              m->amd_decode_negative_[block_index];
        }
      }
    }
    return;
  }
  size_t offset = 0;
  for (int i = 0; i < cinfo->comps_in_scan; ++i) {
    const jpeg_component_info* comp = cinfo->cur_comp_info[i];
    int c = comp->component_index;
    size_t block_x = m->scan_mcu_col_ * comp->MCU_width;
    for (int iy = 0; iy < comp->MCU_height; ++iy) {
      size_t block_y = m->scan_mcu_row_ * comp->MCU_height + iy;
      size_t biy = block_y % comp->v_samp_factor;
      if (block_y >= comp->height_in_blocks) {
        continue;
      }
      size_t nblocks =
          std::min<size_t>(comp->MCU_width, comp->width_in_blocks - block_x);
      size_t ncoeffs = nblocks * DCTSIZE2;
      coeff_t* coeffs = &m->coeff_rows[c][biy][block_x][0];
      memcpy(&m->mcu_.coeffs[offset], coeffs, ncoeffs * sizeof(coeffs[0]));
      offset += ncoeffs;
    }
  }
}

void RestoreMCUCodingState(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  memcpy(m->last_dc_coeff_, m->mcu_.last_dc_coeff, sizeof(m->last_dc_coeff_));
  m->eobrun_ = m->mcu_.eobrun;
  if (m->amd_decode_coefficients_active_) {
    m->amd_decode_coefficient_events_.resize(m->mcu_.amd_decode_event_count);
    for (size_t i = 0; i < m->mcu_.amd_decode_num_blocks; ++i) {
      const size_t block_index = m->mcu_.amd_decode_block_index[i];
      m->amd_decode_significant_[block_index] =
          m->mcu_.amd_decode_significant[i];
      m->amd_decode_negative_[block_index] = m->mcu_.amd_decode_negative[i];
    }
    return;
  }
  size_t offset = 0;
  for (int i = 0; i < cinfo->comps_in_scan; ++i) {
    const jpeg_component_info* comp = cinfo->cur_comp_info[i];
    int c = comp->component_index;
    size_t block_x = m->scan_mcu_col_ * comp->MCU_width;
    for (int iy = 0; iy < comp->MCU_height; ++iy) {
      size_t block_y = m->scan_mcu_row_ * comp->MCU_height + iy;
      size_t biy = block_y % comp->v_samp_factor;
      if (block_y >= comp->height_in_blocks) {
        continue;
      }
      size_t nblocks =
          std::min<size_t>(comp->MCU_width, comp->width_in_blocks - block_x);
      size_t ncoeffs = nblocks * DCTSIZE2;
      coeff_t* coeffs = &m->coeff_rows[c][biy][block_x][0];
      memcpy(coeffs, &m->mcu_.coeffs[offset], ncoeffs * sizeof(coeffs[0]));
      offset += ncoeffs;
    }
  }
}

bool FinishScan(j_decompress_ptr cinfo, const uint8_t* data, const size_t len,
                size_t* pos, size_t* bit_pos) {
  jpeg_decomp_master* m = cinfo->master;
  if (m->eobrun_ > 0) {
    JPEGLI_ERROR("End-of-block run too long.");
  }
  m->eobrun_ = -1;
  memset(m->last_dc_coeff_, 0, sizeof(m->last_dc_coeff_));
  if (*bit_pos == 0) {
    return true;
  }
  if (data[*pos] == 0xff) {
    // After last br.FinishStream we checked that there is at least 2 bytes
    // in the buffer.
    JPEGLI_DASSERT(*pos + 1 < len);
    // br.FinishStream would have detected an early marker.
    JPEGLI_DASSERT(data[*pos + 1] == 0);
    *pos += 2;
  } else {
    *pos += 1;
  }
  *bit_pos = 0;
  return true;
}

}  // namespace

void PrepareForiMCURow(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  for (int i = 0; i < cinfo->comps_in_scan; ++i) {
    const jpeg_component_info* comp = cinfo->cur_comp_info[i];
    int c = comp->component_index;
    int by0 = cinfo->input_iMCU_row * comp->v_samp_factor;
    int block_rows_left = comp->height_in_blocks - by0;
    int max_block_rows = std::min(comp->v_samp_factor, block_rows_left);
    int offset = m->streaming_mode_ ? 0 : by0;
    m->coeff_rows[c] = (*cinfo->mem->access_virt_barray)(
        reinterpret_cast<j_common_ptr>(cinfo), m->coef_arrays[c], offset,
        max_block_rows, TRUE);
  }
}

template <bool kRecordEvents>
int ProcessScanImpl(j_decompress_ptr cinfo, const uint8_t* const data,
                    const size_t len, size_t* pos, size_t* bit_pos) {
  if (len == 0) {
    return kNeedMoreInput;
  }
  jpeg_decomp_master* m = cinfo->master;
  for (;;) {
    // Handle the restart intervals.
    if (cinfo->restart_interval > 0 && m->restarts_to_go_ == 0) {
      if (!FinishScan(cinfo, data, len, pos, bit_pos)) {
        return kNeedMoreInput;
      }
      // Go to the next marker, warn if we had to skip any data.
      size_t num_skipped = 0;
      while (*pos + 1 < len && (data[*pos] != 0xff || data[*pos + 1] == 0 ||
                                data[*pos + 1] == 0xff)) {
        ++(*pos);
        ++num_skipped;
      }
      if (num_skipped > 0) {
        JPEGLI_WARN("Skipped %d bytes before restart marker",
                    static_cast<int>(num_skipped));
      }
      if (*pos + 2 > len) {
        return kNeedMoreInput;
      }
      cinfo->unread_marker = data[*pos + 1];
      *pos += 2;
      return kHandleRestart;
    }

    size_t start_pos = *pos;
    BitReaderState br(data, len, start_pos);
    if (*bit_pos > 0) {
      br.ReadBits(*bit_pos);
    }
    if (start_pos + kMaxMCUByteSize > len) {
      SaveMCUCodingState(cinfo);
    }

    // Decode one MCU.
    HWY_ALIGN_MAX static coeff_t sink_block[DCTSIZE2] = {0};
    bool scan_ok = true;
    for (int i = 0; i < cinfo->comps_in_scan; ++i) {
      const jpeg_component_info* comp = cinfo->cur_comp_info[i];
      int c = comp->component_index;
      const HuffmanTableEntry* dc_lut =
          &m->dc_huff_lut_[comp->dc_tbl_no * kJpegHuffmanLutSize];
      const HuffmanTableEntry* ac_lut =
          &m->ac_huff_lut_[comp->ac_tbl_no * kJpegHuffmanLutSize];
      for (int iy = 0; iy < comp->MCU_height; ++iy) {
        size_t block_y = m->scan_mcu_row_ * comp->MCU_height + iy;
        int biy = block_y % comp->v_samp_factor;
        for (int ix = 0; ix < comp->MCU_width; ++ix) {
          size_t block_x = m->scan_mcu_col_ * comp->MCU_width + ix;
          coeff_t* coeffs;
          size_t amd_decode_block_index = kInvalidAmdDecodeBlock;
          if (block_x >= comp->width_in_blocks ||
              block_y >= comp->height_in_blocks) {
            // Note that it is OK that sink_block is uninitialized because
            // it will never be used in any branches, even in the RefineDCTBlock
            // case, because only DC scans can be interleaved and we don't use
            // the zero-ness of the DC coeff in the DC refinement code-path.
            coeffs = sink_block;
          } else {
            coeffs = &m->coeff_rows[c][biy][block_x][0];
            if constexpr (kRecordEvents) {
              amd_decode_block_index =
                  m->amd_decode_component_block_offsets_[c] +
                  block_y * comp->width_in_blocks + block_x;
            }
          }
          if (cinfo->Ah == 0) {
            bool decoded;
            if constexpr (kRecordEvents) {
              decoded = DecodeDCTBlock<true>(
                  dc_lut, ac_lut, cinfo->Ss, cinfo->Se, cinfo->Al, &m->eobrun_,
                  &br, &m->last_dc_coeff_[comp->component_index], coeffs, m,
                  amd_decode_block_index);
            } else {
              decoded = DecodeDCTBlockCpu(
                  dc_lut, ac_lut, cinfo->Ss, cinfo->Se, cinfo->Al, &m->eobrun_,
                  &br, &m->last_dc_coeff_[comp->component_index], coeffs);
            }
            if (!decoded) {
              scan_ok = false;
            }
          } else {
            bool decoded;
            if constexpr (kRecordEvents) {
              decoded = RefineDCTBlock<true>(ac_lut, cinfo->Ss, cinfo->Se,
                                             cinfo->Al, &m->eobrun_, &br,
                                             coeffs, m, amd_decode_block_index);
            } else {
              decoded = RefineDCTBlockCpu(ac_lut, cinfo->Ss, cinfo->Se,
                                          cinfo->Al, &m->eobrun_, &br, coeffs);
            }
            if (!decoded) {
              scan_ok = false;
            }
          }
        }
      }
    }
    size_t new_pos;
    size_t new_bit_pos;
    bool stream_ok = br.FinishStream(&new_pos, &new_bit_pos);
    if (new_pos + 2 > len) {
      // If reading stopped within the last two bytes, we have to request more
      // input even if FinishStream() returned true, since the Huffman code
      // reader could have peaked ahead some bits past the current input chunk
      // and thus the last prefix code length could have been wrong. We can do
      // this because a valid JPEG bit stream has two extra bytes at the end.
      RestoreMCUCodingState(cinfo);
      return kNeedMoreInput;
    }
    *pos = new_pos;
    *bit_pos = new_bit_pos;
    if (!stream_ok) {
      // We hit a marker during parsing.
      JPEGLI_DASSERT(data[*pos] == 0xff);
      JPEGLI_DASSERT(data[*pos + 1] != 0);
      RestoreMCUCodingState(cinfo);
      JPEGLI_WARN("Incomplete scan detected.");
      return JPEG_SCAN_COMPLETED;
    }
    if (!scan_ok) {
      JPEGLI_ERROR("Failed to decode DCT block");
    }
    if (m->restarts_to_go_ > 0) {
      --m->restarts_to_go_;
    }
    ++m->scan_mcu_col_;
    if (m->scan_mcu_col_ == cinfo->MCUs_per_row) {
      ++m->scan_mcu_row_;
      m->scan_mcu_col_ = 0;
      if (m->scan_mcu_row_ == cinfo->MCU_rows_in_scan) {
        if (!FinishScan(cinfo, data, len, pos, bit_pos)) {
          return kNeedMoreInput;
        }
        break;
      } else if ((m->scan_mcu_row_ % m->mcu_rows_per_iMCU_row_) == 0) {
        // Current iMCU row is done.
        break;
      }
    }
  }
  ++cinfo->input_iMCU_row;
  if (cinfo->input_iMCU_row < cinfo->total_iMCU_rows) {
    PrepareForiMCURow(cinfo);
    return JPEG_ROW_COMPLETED;
  }
  return JPEG_SCAN_COMPLETED;
}

int ProcessScan(j_decompress_ptr cinfo, const uint8_t* const data,
                const size_t len, size_t* pos, size_t* bit_pos) {
  if (AmdVulkanDecodeEntropyActive(cinfo->master)) {
    return AmdVulkanDecodeEntropyCaptureScan(cinfo, data, len, pos, bit_pos);
  }
  return cinfo->master->amd_decode_coefficients_active_
             ? ProcessScanImpl<true>(cinfo, data, len, pos, bit_pos)
             : ProcessScanImpl<false>(cinfo, data, len, pos, bit_pos);
}

}  // namespace jpegli

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_COPY_SVE_GENERIC_HPP_INCLUDED
#define SVENTT_COPY_SVE_GENERIC_HPP_INCLUDED

#include <cstdint>
#include <stdexcept>

#include <arm_sve.h>

#include "sventt/common/sve.hpp"

namespace sventt {

template <std::uint64_t size> class CopySVEGeneric {

public:
  static void copy(void *const dst_arg, const void *const src_arg) {
    const svbool_t ptrue{svptrue_b8()};

    std::uint8_t *dst{static_cast<std::uint8_t *>(dst_arg)};
    const std::uint8_t *src{static_cast<const std::uint8_t *>(src_arg)};

    if constexpr (size % (cntb * 16) == 0) {
      src += cntb * 8;
      dst += cntb * 8;
      for (std::uint64_t i{}; i < size;
           i += cntb * 16, src += cntb * 16, dst += cntb * 16) {
        svuint8_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13,
            x14, x15;

        x0 = svld1_vnum(ptrue, src, -8);
        x1 = svld1_vnum(ptrue, src, -7);
        x2 = svld1_vnum(ptrue, src, -6);
        x3 = svld1_vnum(ptrue, src, -5);
        x4 = svld1_vnum(ptrue, src, -4);
        x5 = svld1_vnum(ptrue, src, -3);
        x6 = svld1_vnum(ptrue, src, -2);
        x7 = svld1_vnum(ptrue, src, -1);
        x8 = svld1_vnum(ptrue, src, 0);
        x9 = svld1_vnum(ptrue, src, 1);
        x10 = svld1_vnum(ptrue, src, 2);
        x11 = svld1_vnum(ptrue, src, 3);
        x12 = svld1_vnum(ptrue, src, 4);
        x13 = svld1_vnum(ptrue, src, 5);
        x14 = svld1_vnum(ptrue, src, 6);
        x15 = svld1_vnum(ptrue, src, 7);

        svst1_vnum(ptrue, dst, -8, x0);
        svst1_vnum(ptrue, dst, -7, x1);
        svst1_vnum(ptrue, dst, -6, x2);
        svst1_vnum(ptrue, dst, -5, x3);
        svst1_vnum(ptrue, dst, -4, x4);
        svst1_vnum(ptrue, dst, -3, x5);
        svst1_vnum(ptrue, dst, -2, x6);
        svst1_vnum(ptrue, dst, -1, x7);
        svst1_vnum(ptrue, dst, 0, x8);
        svst1_vnum(ptrue, dst, 1, x9);
        svst1_vnum(ptrue, dst, 2, x10);
        svst1_vnum(ptrue, dst, 3, x11);
        svst1_vnum(ptrue, dst, 4, x12);
        svst1_vnum(ptrue, dst, 5, x13);
        svst1_vnum(ptrue, dst, 6, x14);
        svst1_vnum(ptrue, dst, 7, x15);
      }
    } else if constexpr (size % (cntb * 8) == 0) {
      for (std::uint64_t i{}; i < size;
           i += cntb * 8, src += cntb * 8, dst += cntb * 8) {
        svuint8_t x0, x1, x2, x3, x4, x5, x6, x7;

        x0 = svld1_vnum(ptrue, src, 0);
        x1 = svld1_vnum(ptrue, src, 1);
        x2 = svld1_vnum(ptrue, src, 2);
        x3 = svld1_vnum(ptrue, src, 3);
        x4 = svld1_vnum(ptrue, src, 4);
        x5 = svld1_vnum(ptrue, src, 5);
        x6 = svld1_vnum(ptrue, src, 6);
        x7 = svld1_vnum(ptrue, src, 7);

        svst1_vnum(ptrue, dst, 0, x0);
        svst1_vnum(ptrue, dst, 1, x1);
        svst1_vnum(ptrue, dst, 2, x2);
        svst1_vnum(ptrue, dst, 3, x3);
        svst1_vnum(ptrue, dst, 4, x4);
        svst1_vnum(ptrue, dst, 5, x5);
        svst1_vnum(ptrue, dst, 6, x6);
        svst1_vnum(ptrue, dst, 7, x7);
      }
    } else if constexpr (size % (cntb * 4) == 0) {
      for (std::uint64_t i{}; i < size;
           i += cntb * 4, src += cntb * 4, dst += cntb * 4) {
        svuint8_t x0, x1, x2, x3;

        x0 = svld1_vnum(ptrue, src, 0);
        x1 = svld1_vnum(ptrue, src, 1);
        x2 = svld1_vnum(ptrue, src, 2);
        x3 = svld1_vnum(ptrue, src, 3);

        svst1_vnum(ptrue, dst, 0, x0);
        svst1_vnum(ptrue, dst, 1, x1);
        svst1_vnum(ptrue, dst, 2, x2);
        svst1_vnum(ptrue, dst, 3, x3);
      }
    } else if constexpr (size % (cntb * 2) == 0) {
      for (std::uint64_t i{}; i < size;
           i += cntb * 2, src += cntb * 2, dst += cntb * 2) {
        svuint8_t x0, x1;

        x0 = svld1_vnum(ptrue, src, 0);
        x1 = svld1_vnum(ptrue, src, 1);

        svst1_vnum(ptrue, dst, 0, x0);
        svst1_vnum(ptrue, dst, 1, x1);
      }
    } else if constexpr (size % (cntb * 1) == 0) {
      for (std::uint64_t i{}; i < size; i += cntb, src += cntb, dst += cntb) {
        svuint8_t x0;

        x0 = svld1_vnum(ptrue, src, 0);

        svst1_vnum(ptrue, dst, 0, x0);
      }
    } else {
      throw std::invalid_argument{"Unsupported copy size"};
    }
  }
};

} // namespace sventt

#endif /* SVENTT_COPY_SVE_GENERIC_HPP_INCLUDED */

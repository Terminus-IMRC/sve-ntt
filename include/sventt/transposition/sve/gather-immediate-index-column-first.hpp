// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_TRANSPOSITION_SVE_GATHER_IMMEDIATE_INDEX_COLUMN_FIRST_HPP_INCLUDED
#define SVENTT_TRANSPOSITION_SVE_GATHER_IMMEDIATE_INDEX_COLUMN_FIRST_HPP_INCLUDED

#include <cstdint>
#include <stdexcept>

#include <arm_sve.h>

#include "sventt/common/sve.hpp"

namespace sventt {

template <std::uint64_t block_rows, std::uint64_t block_cols>
class TransposeParallelSVEGatherColumnFirst {

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    const svbool_t ptrue{svptrue_b8()};

    if (src_rows % block_rows != 0 || src_cols % block_cols != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }
    if (block_rows % cntd != 0) {
      throw std::invalid_argument{
          "Block dimensions are not divisible by SVE vector length"};
    }

    if (block_cols == 1) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
        for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
          svuint64_t src_base{svindex_u64(
              reinterpret_cast<std::uintptr_t>(&src[ld_src * i + j]),
              8 * ld_src)};
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0;
            x0 = svld1_gather_offset_u64(ptrue, src_base, 0);
            svst1(ptrue, &dst[ld_dst * j + i + ii], x0);
            src_base = svadd_x(ptrue, src_base, 8 * ld_src * cntd);
          }
        }
      }
    } else if (block_cols == 2) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
        for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
          svuint64_t src_base{svindex_u64(
              reinterpret_cast<std::uintptr_t>(&src[ld_src * i + j]),
              8 * ld_src)};
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1;
            x0 = svld1_gather_offset_u64(ptrue, src_base, cntd * 0);
            x1 = svld1_gather_offset_u64(ptrue, src_base, cntd * 1);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            src_base = svadd_x(ptrue, src_base, 8 * ld_src * cntd);
          }
        }
      }
    } else if (block_cols == 4) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
        for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
          svuint64_t src_base{svindex_u64(
              reinterpret_cast<std::uintptr_t>(&src[ld_src * i + j]),
              8 * ld_src)};
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3;
            x0 = svld1_gather_offset_u64(ptrue, src_base, cntd * 0);
            x1 = svld1_gather_offset_u64(ptrue, src_base, cntd * 1);
            x2 = svld1_gather_offset_u64(ptrue, src_base, cntd * 2);
            x3 = svld1_gather_offset_u64(ptrue, src_base, cntd * 3);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
            src_base = svadd_x(ptrue, src_base, 8 * ld_src * cntd);
          }
        }
      }
    } else if (block_cols == 8) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
        for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
          svuint64_t src_base{svindex_u64(
              reinterpret_cast<std::uintptr_t>(&src[ld_src * i + j]),
              8 * ld_src)};
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
            x0 = svld1_gather_offset_u64(ptrue, src_base, cntd * 0);
            x1 = svld1_gather_offset_u64(ptrue, src_base, cntd * 1);
            x2 = svld1_gather_offset_u64(ptrue, src_base, cntd * 2);
            x3 = svld1_gather_offset_u64(ptrue, src_base, cntd * 3);
            x4 = svld1_gather_offset_u64(ptrue, src_base, cntd * 4);
            x5 = svld1_gather_offset_u64(ptrue, src_base, cntd * 5);
            x6 = svld1_gather_offset_u64(ptrue, src_base, cntd * 6);
            x7 = svld1_gather_offset_u64(ptrue, src_base, cntd * 7);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
            svst1(ptrue, &dst[ld_dst * (j + 4) + i + ii], x4);
            svst1(ptrue, &dst[ld_dst * (j + 5) + i + ii], x5);
            svst1(ptrue, &dst[ld_dst * (j + 6) + i + ii], x6);
            svst1(ptrue, &dst[ld_dst * (j + 7) + i + ii], x7);
            src_base = svadd_x(ptrue, src_base, 8 * ld_src * cntd);
          }
        }
      }
    } else if (block_cols == 16) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
        for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
          svuint64_t src_base{svindex_u64(
              reinterpret_cast<std::uintptr_t>(&src[ld_src * i + j]),
              8 * ld_src)};
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xa, xb, xc, xd,
                xe, xf;
            x0 = svld1_gather_offset_u64(ptrue, src_base, cntd * 0);
            x1 = svld1_gather_offset_u64(ptrue, src_base, cntd * 1);
            x2 = svld1_gather_offset_u64(ptrue, src_base, cntd * 2);
            x3 = svld1_gather_offset_u64(ptrue, src_base, cntd * 3);
            x4 = svld1_gather_offset_u64(ptrue, src_base, cntd * 4);
            x5 = svld1_gather_offset_u64(ptrue, src_base, cntd * 5);
            x6 = svld1_gather_offset_u64(ptrue, src_base, cntd * 6);
            x7 = svld1_gather_offset_u64(ptrue, src_base, cntd * 7);
            x8 = svld1_gather_offset_u64(ptrue, src_base, cntd * 8);
            x9 = svld1_gather_offset_u64(ptrue, src_base, cntd * 9);
            xa = svld1_gather_offset_u64(ptrue, src_base, cntd * 10);
            xb = svld1_gather_offset_u64(ptrue, src_base, cntd * 11);
            xc = svld1_gather_offset_u64(ptrue, src_base, cntd * 12);
            xd = svld1_gather_offset_u64(ptrue, src_base, cntd * 13);
            xe = svld1_gather_offset_u64(ptrue, src_base, cntd * 14);
            xf = svld1_gather_offset_u64(ptrue, src_base, cntd * 15);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
            svst1(ptrue, &dst[ld_dst * (j + 4) + i + ii], x4);
            svst1(ptrue, &dst[ld_dst * (j + 5) + i + ii], x5);
            svst1(ptrue, &dst[ld_dst * (j + 6) + i + ii], x6);
            svst1(ptrue, &dst[ld_dst * (j + 7) + i + ii], x7);
            svst1(ptrue, &dst[ld_dst * (j + 8) + i + ii], x8);
            svst1(ptrue, &dst[ld_dst * (j + 9) + i + ii], x9);
            svst1(ptrue, &dst[ld_dst * (j + 10) + i + ii], xa);
            svst1(ptrue, &dst[ld_dst * (j + 11) + i + ii], xb);
            svst1(ptrue, &dst[ld_dst * (j + 12) + i + ii], xc);
            svst1(ptrue, &dst[ld_dst * (j + 13) + i + ii], xd);
            svst1(ptrue, &dst[ld_dst * (j + 14) + i + ii], xe);
            svst1(ptrue, &dst[ld_dst * (j + 15) + i + ii], xf);
            src_base = svadd_x(ptrue, src_base, 8 * ld_src * cntd);
          }
        }
      }
    } else if (block_cols == 32) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
        for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
          svuint64_t src_base{svindex_u64(
              reinterpret_cast<std::uintptr_t>(&src[ld_src * i + j]),
              8 * ld_src)};
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xa, xb, xc, xd,
                xe, xf, xg, xh, xi, xj, xk, xl, xm, xn, xo, xp, xq, xr, xs, xt,
                xu, xv;
            x0 = svld1_gather_offset_u64(ptrue, src_base, cntd * 0);
            x1 = svld1_gather_offset_u64(ptrue, src_base, cntd * 1);
            x2 = svld1_gather_offset_u64(ptrue, src_base, cntd * 2);
            x3 = svld1_gather_offset_u64(ptrue, src_base, cntd * 3);
            x4 = svld1_gather_offset_u64(ptrue, src_base, cntd * 4);
            x5 = svld1_gather_offset_u64(ptrue, src_base, cntd * 5);
            x6 = svld1_gather_offset_u64(ptrue, src_base, cntd * 6);
            x7 = svld1_gather_offset_u64(ptrue, src_base, cntd * 7);
            x8 = svld1_gather_offset_u64(ptrue, src_base, cntd * 8);
            x9 = svld1_gather_offset_u64(ptrue, src_base, cntd * 9);
            xa = svld1_gather_offset_u64(ptrue, src_base, cntd * 10);
            xb = svld1_gather_offset_u64(ptrue, src_base, cntd * 11);
            xc = svld1_gather_offset_u64(ptrue, src_base, cntd * 12);
            xd = svld1_gather_offset_u64(ptrue, src_base, cntd * 13);
            xe = svld1_gather_offset_u64(ptrue, src_base, cntd * 14);
            xf = svld1_gather_offset_u64(ptrue, src_base, cntd * 15);
            xg = svld1_gather_offset_u64(ptrue, src_base, cntd * 16);
            xh = svld1_gather_offset_u64(ptrue, src_base, cntd * 17);
            xi = svld1_gather_offset_u64(ptrue, src_base, cntd * 18);
            xj = svld1_gather_offset_u64(ptrue, src_base, cntd * 19);
            xk = svld1_gather_offset_u64(ptrue, src_base, cntd * 20);
            xl = svld1_gather_offset_u64(ptrue, src_base, cntd * 21);
            xm = svld1_gather_offset_u64(ptrue, src_base, cntd * 22);
            xn = svld1_gather_offset_u64(ptrue, src_base, cntd * 23);
            xo = svld1_gather_offset_u64(ptrue, src_base, cntd * 24);
            xp = svld1_gather_offset_u64(ptrue, src_base, cntd * 25);
            xq = svld1_gather_offset_u64(ptrue, src_base, cntd * 26);
            xr = svld1_gather_offset_u64(ptrue, src_base, cntd * 27);
            xs = svld1_gather_offset_u64(ptrue, src_base, cntd * 28);
            xt = svld1_gather_offset_u64(ptrue, src_base, cntd * 29);
            xu = svld1_gather_offset_u64(ptrue, src_base, cntd * 30);
            xv = svld1_gather_offset_u64(ptrue, src_base, cntd * 31);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
            svst1(ptrue, &dst[ld_dst * (j + 4) + i + ii], x4);
            svst1(ptrue, &dst[ld_dst * (j + 5) + i + ii], x5);
            svst1(ptrue, &dst[ld_dst * (j + 6) + i + ii], x6);
            svst1(ptrue, &dst[ld_dst * (j + 7) + i + ii], x7);
            svst1(ptrue, &dst[ld_dst * (j + 8) + i + ii], x8);
            svst1(ptrue, &dst[ld_dst * (j + 9) + i + ii], x9);
            svst1(ptrue, &dst[ld_dst * (j + 10) + i + ii], xa);
            svst1(ptrue, &dst[ld_dst * (j + 11) + i + ii], xb);
            svst1(ptrue, &dst[ld_dst * (j + 12) + i + ii], xc);
            svst1(ptrue, &dst[ld_dst * (j + 13) + i + ii], xd);
            svst1(ptrue, &dst[ld_dst * (j + 14) + i + ii], xe);
            svst1(ptrue, &dst[ld_dst * (j + 15) + i + ii], xf);
            svst1(ptrue, &dst[ld_dst * (j + 16) + i + ii], xg);
            svst1(ptrue, &dst[ld_dst * (j + 17) + i + ii], xh);
            svst1(ptrue, &dst[ld_dst * (j + 18) + i + ii], xi);
            svst1(ptrue, &dst[ld_dst * (j + 19) + i + ii], xj);
            svst1(ptrue, &dst[ld_dst * (j + 20) + i + ii], xk);
            svst1(ptrue, &dst[ld_dst * (j + 21) + i + ii], xl);
            svst1(ptrue, &dst[ld_dst * (j + 22) + i + ii], xm);
            svst1(ptrue, &dst[ld_dst * (j + 23) + i + ii], xn);
            svst1(ptrue, &dst[ld_dst * (j + 24) + i + ii], xo);
            svst1(ptrue, &dst[ld_dst * (j + 25) + i + ii], xp);
            svst1(ptrue, &dst[ld_dst * (j + 26) + i + ii], xq);
            svst1(ptrue, &dst[ld_dst * (j + 27) + i + ii], xr);
            svst1(ptrue, &dst[ld_dst * (j + 28) + i + ii], xs);
            svst1(ptrue, &dst[ld_dst * (j + 29) + i + ii], xt);
            svst1(ptrue, &dst[ld_dst * (j + 30) + i + ii], xu);
            svst1(ptrue, &dst[ld_dst * (j + 31) + i + ii], xv);
            src_base = svadd_x(ptrue, src_base, 8 * ld_src * cntd);
          }
        }
      }
    } else {
      throw std::invalid_argument{"Unsupported block_cols"};
    }
  }
};

} // namespace sventt

#endif /* SVENTT_TRANSPOSITION_SVE_GATHER_IMMEDIATE_INDEX_COLUMN_FIRST_HPP_INCLUDED \
        */

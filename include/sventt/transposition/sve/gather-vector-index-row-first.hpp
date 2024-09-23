// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_TRANSPOSITION_SVE_GATHER_VECTOR_INDEX_ROW_FIRST_HPP_INCLUDED
#define SVENTT_TRANSPOSITION_SVE_GATHER_VECTOR_INDEX_ROW_FIRST_HPP_INCLUDED

#include <bit>
#include <cstdint>
#include <stdexcept>

#include <arm_sve.h>

#include "sventt/common/sve.hpp"

namespace sventt {

template <std::uint64_t block_rows, std::uint64_t block_cols>
class TransposeSVEGatherVectorIndexRowFirst {

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    if (src_rows % block_rows != 0 || src_cols % block_cols != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }
    if (block_rows % cntd != 0) {
      throw std::invalid_argument{
          "Block dimensions are not divisible by SVE vector length"};
    }

    const svbool_t ptrue{svptrue_b8()};
    const svuint64_t offset{svindex_u64(0, ld_src)};

    if (block_cols == 1) {
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j], offset);
            svst1(ptrue, &dst[ld_dst * j + i + ii], x0);
          }
        }
      }
    } else if (block_cols == 2) {
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 0],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 1],
                                    offset);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
          }
        }
      }
    } else if (block_cols == 4) {
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 0],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 1],
                                    offset);
            x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 2],
                                    offset);
            x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 3],
                                    offset);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
          }
        }
      }
    } else if (block_cols == 8) {
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 0],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 1],
                                    offset);
            x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 2],
                                    offset);
            x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 3],
                                    offset);
            x4 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 4],
                                    offset);
            x5 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 5],
                                    offset);
            x6 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 6],
                                    offset);
            x7 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 7],
                                    offset);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
            svst1(ptrue, &dst[ld_dst * (j + 4) + i + ii], x4);
            svst1(ptrue, &dst[ld_dst * (j + 5) + i + ii], x5);
            svst1(ptrue, &dst[ld_dst * (j + 6) + i + ii], x6);
            svst1(ptrue, &dst[ld_dst * (j + 7) + i + ii], x7);
          }
        }
      }
    } else if (block_cols == 16) {
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xa, xb, xc, xd,
                xe, xf;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 0],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 1],
                                    offset);
            x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 2],
                                    offset);
            x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 3],
                                    offset);
            x4 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 4],
                                    offset);
            x5 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 5],
                                    offset);
            x6 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 6],
                                    offset);
            x7 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 7],
                                    offset);
            x8 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 8],
                                    offset);
            x9 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 9],
                                    offset);
            xa = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 10],
                                    offset);
            xb = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 11],
                                    offset);
            xc = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 12],
                                    offset);
            xd = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 13],
                                    offset);
            xe = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 14],
                                    offset);
            xf = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 15],
                                    offset);
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
          }
        }
      }
    } else if (block_cols == 32) {
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xa, xb, xc, xd,
                xe, xf, xg, xh, xi, xj, xk, xl, xm, xn, xo, xp, xq, xr, xs, xt,
                xu, xv;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 0],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 1],
                                    offset);
            x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 2],
                                    offset);
            x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 3],
                                    offset);
            x4 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 4],
                                    offset);
            x5 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 5],
                                    offset);
            x6 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 6],
                                    offset);
            x7 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 7],
                                    offset);
            x8 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 8],
                                    offset);
            x9 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 9],
                                    offset);
            xa = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 10],
                                    offset);
            xb = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 11],
                                    offset);
            xc = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 12],
                                    offset);
            xd = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 13],
                                    offset);
            xe = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 14],
                                    offset);
            xf = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 15],
                                    offset);
            xg = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 16],
                                    offset);
            xh = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 17],
                                    offset);
            xi = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 18],
                                    offset);
            xj = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 19],
                                    offset);
            xk = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 20],
                                    offset);
            xl = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 21],
                                    offset);
            xm = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 22],
                                    offset);
            xn = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 23],
                                    offset);
            xo = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 24],
                                    offset);
            xp = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 25],
                                    offset);
            xq = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 26],
                                    offset);
            xr = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 27],
                                    offset);
            xs = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 28],
                                    offset);
            xt = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 29],
                                    offset);
            xu = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 30],
                                    offset);
            xv = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 31],
                                    offset);
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
          }
        }
      }
    } else {
      throw std::invalid_argument{"Unsupported block_cols"};
    }
  }

  static void transpose(std::uint64_t *const dst, const std::uint64_t dim) {
    if (block_rows != block_cols) {
      throw std::invalid_argument{
          "Block dimensions need to be equal at least for now"};
    }
    if (dim % block_rows != 0 || dim % block_cols != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }
    if (block_rows % cntd != 0) {
      throw std::invalid_argument{
          "Block dimensions are not divisible by SVE vector length"};
    }

    const svbool_t ptrue{svptrue_b8()};
    const svuint64_t offset{svindex_u64(0, dim)};

    if (block_cols == 8) {
      for (std::uint64_t i = 0; i < dim; i += block_rows) {
        for (std::uint64_t j = 0; j <= i; j += block_cols) {
          /* TODO: Support cntd != 8. */
          if (j < i) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
            svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;
            x0 = svld1_gather_index(ptrue, &dst[dim * i + j + 0], offset);
            x1 = svld1_gather_index(ptrue, &dst[dim * i + j + 1], offset);
            x2 = svld1_gather_index(ptrue, &dst[dim * i + j + 2], offset);
            x3 = svld1_gather_index(ptrue, &dst[dim * i + j + 3], offset);
            x4 = svld1_gather_index(ptrue, &dst[dim * i + j + 4], offset);
            x5 = svld1_gather_index(ptrue, &dst[dim * i + j + 5], offset);
            x6 = svld1_gather_index(ptrue, &dst[dim * i + j + 6], offset);
            x7 = svld1_gather_index(ptrue, &dst[dim * i + j + 7], offset);
            y0 = svld1_gather_index(ptrue, &dst[dim * j + i + 0], offset);
            y1 = svld1_gather_index(ptrue, &dst[dim * j + i + 1], offset);
            y2 = svld1_gather_index(ptrue, &dst[dim * j + i + 2], offset);
            y3 = svld1_gather_index(ptrue, &dst[dim * j + i + 3], offset);
            y4 = svld1_gather_index(ptrue, &dst[dim * j + i + 4], offset);
            y5 = svld1_gather_index(ptrue, &dst[dim * j + i + 5], offset);
            y6 = svld1_gather_index(ptrue, &dst[dim * j + i + 6], offset);
            y7 = svld1_gather_index(ptrue, &dst[dim * j + i + 7], offset);
            svst1(ptrue, &dst[dim * (j + 0) + i], x0);
            svst1(ptrue, &dst[dim * (j + 1) + i], x1);
            svst1(ptrue, &dst[dim * (j + 2) + i], x2);
            svst1(ptrue, &dst[dim * (j + 3) + i], x3);
            svst1(ptrue, &dst[dim * (j + 4) + i], x4);
            svst1(ptrue, &dst[dim * (j + 5) + i], x5);
            svst1(ptrue, &dst[dim * (j + 6) + i], x6);
            svst1(ptrue, &dst[dim * (j + 7) + i], x7);
            svst1(ptrue, &dst[dim * (i + 0) + j], y0);
            svst1(ptrue, &dst[dim * (i + 1) + j], y1);
            svst1(ptrue, &dst[dim * (i + 2) + j], y2);
            svst1(ptrue, &dst[dim * (i + 3) + j], y3);
            svst1(ptrue, &dst[dim * (i + 4) + j], y4);
            svst1(ptrue, &dst[dim * (i + 5) + j], y5);
            svst1(ptrue, &dst[dim * (i + 6) + j], y6);
            svst1(ptrue, &dst[dim * (i + 7) + j], y7);
          } else {
            /* The remaining i = j case. */
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
            x0 = svld1_gather_index(ptrue, &dst[dim * i + j + 0], offset);
            x1 = svld1_gather_index(ptrue, &dst[dim * i + j + 1], offset);
            x2 = svld1_gather_index(ptrue, &dst[dim * i + j + 2], offset);
            x3 = svld1_gather_index(ptrue, &dst[dim * i + j + 3], offset);
            x4 = svld1_gather_index(ptrue, &dst[dim * i + j + 4], offset);
            x5 = svld1_gather_index(ptrue, &dst[dim * i + j + 5], offset);
            x6 = svld1_gather_index(ptrue, &dst[dim * i + j + 6], offset);
            x7 = svld1_gather_index(ptrue, &dst[dim * i + j + 7], offset);
            svst1(ptrue, &dst[dim * (j + 0) + i], x0);
            svst1(ptrue, &dst[dim * (j + 1) + i], x1);
            svst1(ptrue, &dst[dim * (j + 2) + i], x2);
            svst1(ptrue, &dst[dim * (j + 3) + i], x3);
            svst1(ptrue, &dst[dim * (j + 4) + i], x4);
            svst1(ptrue, &dst[dim * (j + 5) + i], x5);
            svst1(ptrue, &dst[dim * (j + 6) + i], x6);
            svst1(ptrue, &dst[dim * (j + 7) + i], x7);
          }
        }
      }
    } else {
      throw std::invalid_argument{"Unsupported block_cols"};
    }
  }
};

template <std::uint64_t block_rows, std::uint64_t block_cols>
class TransposeParallelSVEGatherVectorIndexRowFirst {

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    if (src_rows % block_rows != 0 || src_cols % block_cols != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }
    if (block_rows % cntd != 0) {
      throw std::invalid_argument{
          "Block dimensions are not divisible by SVE vector length"};
    }

    const svbool_t ptrue{svptrue_b8()};
    const svuint64_t offset{svindex_u64(0, ld_src)};

    if (block_cols == 1) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j], offset);
            svst1(ptrue, &dst[ld_dst * j + i + ii], x0);
          }
        }
      }
    } else if (block_cols == 2) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 0],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 1],
                                    offset);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
          }
        }
      }
    } else if (block_cols == 4) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 0],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 1],
                                    offset);
            x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 2],
                                    offset);
            x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 3],
                                    offset);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
          }
        }
      }
    } else if (block_cols == 8) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 0],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 1],
                                    offset);
            x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 2],
                                    offset);
            x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 3],
                                    offset);
            x4 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 4],
                                    offset);
            x5 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 5],
                                    offset);
            x6 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 6],
                                    offset);
            x7 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 7],
                                    offset);
            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
            svst1(ptrue, &dst[ld_dst * (j + 4) + i + ii], x4);
            svst1(ptrue, &dst[ld_dst * (j + 5) + i + ii], x5);
            svst1(ptrue, &dst[ld_dst * (j + 6) + i + ii], x6);
            svst1(ptrue, &dst[ld_dst * (j + 7) + i + ii], x7);
          }
        }
      }
    } else if (block_cols == 16) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xa, xb, xc, xd,
                xe, xf;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 0],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 1],
                                    offset);
            x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 2],
                                    offset);
            x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 3],
                                    offset);
            x4 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 4],
                                    offset);
            x5 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 5],
                                    offset);
            x6 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 6],
                                    offset);
            x7 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 7],
                                    offset);
            x8 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 8],
                                    offset);
            x9 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 9],
                                    offset);
            xa = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 10],
                                    offset);
            xb = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 11],
                                    offset);
            xc = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 12],
                                    offset);
            xd = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 13],
                                    offset);
            xe = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 14],
                                    offset);
            xf = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 15],
                                    offset);
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
          }
        }
      }
    } else if (block_cols == 32) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xa, xb, xc, xd,
                xe, xf, xg, xh, xi, xj, xk, xl, xm, xn, xo, xp, xq, xr, xs, xt,
                xu, xv;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 0],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 1],
                                    offset);
            x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 2],
                                    offset);
            x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 3],
                                    offset);
            x4 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 4],
                                    offset);
            x5 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 5],
                                    offset);
            x6 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 6],
                                    offset);
            x7 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 7],
                                    offset);
            x8 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 8],
                                    offset);
            x9 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 9],
                                    offset);
            xa = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 10],
                                    offset);
            xb = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 11],
                                    offset);
            xc = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 12],
                                    offset);
            xd = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 13],
                                    offset);
            xe = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 14],
                                    offset);
            xf = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 15],
                                    offset);
            xg = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 16],
                                    offset);
            xh = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 17],
                                    offset);
            xi = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 18],
                                    offset);
            xj = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 19],
                                    offset);
            xk = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 20],
                                    offset);
            xl = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 21],
                                    offset);
            xm = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 22],
                                    offset);
            xn = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 23],
                                    offset);
            xo = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 24],
                                    offset);
            xp = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 25],
                                    offset);
            xq = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 26],
                                    offset);
            xr = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 27],
                                    offset);
            xs = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 28],
                                    offset);
            xt = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 29],
                                    offset);
            xu = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 30],
                                    offset);
            xv = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + 31],
                                    offset);
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
          }
        }
      }
    } else {
      throw std::invalid_argument{"Unsupported block_cols"};
    }
  }

  static void transpose(std::uint64_t *const dst, const std::uint64_t dim) {
    if (block_rows != block_cols) {
      throw std::invalid_argument{
          "Block dimensions need to be equal at least for now"};
    }
    if (dim % block_rows != 0 || dim % block_cols != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }
    if (block_rows % cntd != 0) {
      throw std::invalid_argument{
          "Block dimensions are not divisible by SVE vector length"};
    }

    const svbool_t ptrue{svptrue_b8()};
    const svuint64_t offset{svindex_u64(0, dim)};

    if (block_cols == 8) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < dim; i += block_rows) {
        for (std::uint64_t j = 0; j <= i; j += block_cols) {
          /* TODO: Support cntd != 8. */
          if (j < i) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
            svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;
            x0 = svld1_gather_index(ptrue, &dst[dim * i + j + 0], offset);
            x1 = svld1_gather_index(ptrue, &dst[dim * i + j + 1], offset);
            x2 = svld1_gather_index(ptrue, &dst[dim * i + j + 2], offset);
            x3 = svld1_gather_index(ptrue, &dst[dim * i + j + 3], offset);
            x4 = svld1_gather_index(ptrue, &dst[dim * i + j + 4], offset);
            x5 = svld1_gather_index(ptrue, &dst[dim * i + j + 5], offset);
            x6 = svld1_gather_index(ptrue, &dst[dim * i + j + 6], offset);
            x7 = svld1_gather_index(ptrue, &dst[dim * i + j + 7], offset);
            y0 = svld1_gather_index(ptrue, &dst[dim * j + i + 0], offset);
            y1 = svld1_gather_index(ptrue, &dst[dim * j + i + 1], offset);
            y2 = svld1_gather_index(ptrue, &dst[dim * j + i + 2], offset);
            y3 = svld1_gather_index(ptrue, &dst[dim * j + i + 3], offset);
            y4 = svld1_gather_index(ptrue, &dst[dim * j + i + 4], offset);
            y5 = svld1_gather_index(ptrue, &dst[dim * j + i + 5], offset);
            y6 = svld1_gather_index(ptrue, &dst[dim * j + i + 6], offset);
            y7 = svld1_gather_index(ptrue, &dst[dim * j + i + 7], offset);
            svst1(ptrue, &dst[dim * (j + 0) + i], x0);
            svst1(ptrue, &dst[dim * (j + 1) + i], x1);
            svst1(ptrue, &dst[dim * (j + 2) + i], x2);
            svst1(ptrue, &dst[dim * (j + 3) + i], x3);
            svst1(ptrue, &dst[dim * (j + 4) + i], x4);
            svst1(ptrue, &dst[dim * (j + 5) + i], x5);
            svst1(ptrue, &dst[dim * (j + 6) + i], x6);
            svst1(ptrue, &dst[dim * (j + 7) + i], x7);
            svst1(ptrue, &dst[dim * (i + 0) + j], y0);
            svst1(ptrue, &dst[dim * (i + 1) + j], y1);
            svst1(ptrue, &dst[dim * (i + 2) + j], y2);
            svst1(ptrue, &dst[dim * (i + 3) + j], y3);
            svst1(ptrue, &dst[dim * (i + 4) + j], y4);
            svst1(ptrue, &dst[dim * (i + 5) + j], y5);
            svst1(ptrue, &dst[dim * (i + 6) + j], y6);
            svst1(ptrue, &dst[dim * (i + 7) + j], y7);
          } else {
            /* The remaining i = j case. */
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
            x0 = svld1_gather_index(ptrue, &dst[dim * i + j + 0], offset);
            x1 = svld1_gather_index(ptrue, &dst[dim * i + j + 1], offset);
            x2 = svld1_gather_index(ptrue, &dst[dim * i + j + 2], offset);
            x3 = svld1_gather_index(ptrue, &dst[dim * i + j + 3], offset);
            x4 = svld1_gather_index(ptrue, &dst[dim * i + j + 4], offset);
            x5 = svld1_gather_index(ptrue, &dst[dim * i + j + 5], offset);
            x6 = svld1_gather_index(ptrue, &dst[dim * i + j + 6], offset);
            x7 = svld1_gather_index(ptrue, &dst[dim * i + j + 7], offset);
            svst1(ptrue, &dst[dim * (j + 0) + i], x0);
            svst1(ptrue, &dst[dim * (j + 1) + i], x1);
            svst1(ptrue, &dst[dim * (j + 2) + i], x2);
            svst1(ptrue, &dst[dim * (j + 3) + i], x3);
            svst1(ptrue, &dst[dim * (j + 4) + i], x4);
            svst1(ptrue, &dst[dim * (j + 5) + i], x5);
            svst1(ptrue, &dst[dim * (j + 6) + i], x6);
            svst1(ptrue, &dst[dim * (j + 7) + i], x7);
          }
        }
      }
    } else {
      throw std::invalid_argument{"Unsupported block_cols"};
    }
  }
};

template <std::uint64_t block_rows, std::uint64_t block_cols>
class TransposeSVEGatherCombinedColumnVectorIndexRowFirst {

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    if (src_rows % block_rows != 0 || src_cols % block_cols != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }
    if (block_rows % cntd != 0) {
      throw std::invalid_argument{
          "Block dimensions are not divisible by SVE vector length"};
    }

    const svbool_t ptrue{svptrue_b8()};
    static_assert(std::has_single_bit(block_cols));
    const svuint64_t offset{
        svmla_m(ptrue, svand_x(ptrue, svindex_u64(0, 1), block_cols - 1),
                svlsr_x(ptrue, svindex_u64(0, 1), std::countr_zero(block_cols)),
                ld_src)};

    if (block_cols == 1) {
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j], offset);
            svst1(ptrue, &dst[ld_dst * j + i + ii], x0);
          }
        }
      }
    } else if (block_cols == 2) {
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1;
            svuint64_t y0, y1;

            /*
             * x0: 00 10 01 11 02 12 03 06
             * x1: 04 14 05 15 06 16 07 0a
             */
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 0) + j],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 4) + j],
                                    offset);
            /*
             * y0: 00 01 02 03 04 05 06 07
             * y1: 10 11 12 13 14 15 16 17
             */
            y0 = svuzp1(x0, x1);
            y1 = svuzp2(x0, x1);

            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], y0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], y1);
          }
        }
      }
    } else if (block_cols == 4) {
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3;
            svuint64_t y0, y1, y2, y3;

            /*
             * x0: 00 10 20 30 01 11 21 31
             * x1: 02 12 22 32 03 13 23 33
             * x2: 04 14 24 34 05 15 25 35
             * x3: 06 16 26 36 07 17 27 37
             */
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 0) + j],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 2) + j],
                                    offset);
            x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 4) + j],
                                    offset);
            x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 6) + j],
                                    offset);

            /*
             * y0: 00 20 01 21 02 22 03 23
             * y1: 10 30 11 31 12 32 13 33
             * y2: 04 24 05 25 06 26 07 27
             * y3: 14 34 15 35 16 36 17 37
             */
            y0 = svuzp1(x0, x1);
            y1 = svuzp2(x0, x1);
            y2 = svuzp1(x2, x3);
            y3 = svuzp2(x2, x3);

            /*
             * x0: 00 01 02 03 04 05 06 07
             * x1: 10 11 12 13 14 15 16 17
             * x2: 20 21 22 23 24 25 26 27
             * x3: 30 31 32 33 34 35 36 37
             */
            x0 = svuzp1(y0, y2);
            x1 = svuzp1(y1, y3);
            x2 = svuzp2(y0, y2);
            x3 = svuzp2(y1, y3);

            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
          }
        }
      }
    } else if (block_cols == 8) {
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

            x0 = svld1(ptrue, &src[ld_src * (i + ii + 0) + j]);
            x1 = svld1(ptrue, &src[ld_src * (i + ii + 1) + j]);
            x2 = svld1(ptrue, &src[ld_src * (i + ii + 2) + j]);
            x3 = svld1(ptrue, &src[ld_src * (i + ii + 3) + j]);
            x4 = svld1(ptrue, &src[ld_src * (i + ii + 4) + j]);
            x5 = svld1(ptrue, &src[ld_src * (i + ii + 5) + j]);
            x6 = svld1(ptrue, &src[ld_src * (i + ii + 6) + j]);
            x7 = svld1(ptrue, &src[ld_src * (i + ii + 7) + j]);

            transpose_8x8(x0, x1, x2, x3, x4, x5, x6, x7);

            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
            svst1(ptrue, &dst[ld_dst * (j + 4) + i + ii], x4);
            svst1(ptrue, &dst[ld_dst * (j + 5) + i + ii], x5);
            svst1(ptrue, &dst[ld_dst * (j + 6) + i + ii], x6);
            svst1(ptrue, &dst[ld_dst * (j + 7) + i + ii], x7);
          }
        }
      }
    } else {
      throw std::invalid_argument{"Unsupported block_cols"};
    }
  }
};

template <std::uint64_t block_rows, std::uint64_t block_cols>
class TransposeParallelSVEGatherCombinedColumnVectorIndexRowFirst {

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    if (src_rows % block_rows != 0 || src_cols % block_cols != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }
    if (block_rows % cntd != 0) {
      throw std::invalid_argument{
          "Block dimensions are not divisible by SVE vector length"};
    }

    const svbool_t ptrue{svptrue_b8()};
    static_assert(std::has_single_bit(block_cols));
    const svuint64_t offset{
        svmla_m(ptrue, svand_x(ptrue, svindex_u64(0, 1), block_cols - 1),
                svlsr_x(ptrue, svindex_u64(0, 1), std::countr_zero(block_cols)),
                ld_src)};

    if (block_cols == 1) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0;
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j], offset);
            svst1(ptrue, &dst[ld_dst * j + i + ii], x0);
          }
        }
      }
    } else if (block_cols == 2) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1;
            svuint64_t y0, y1;

            /*
             * x0: 00 10 01 11 02 12 03 06
             * x1: 04 14 05 15 06 16 07 0a
             */
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 0) + j],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 4) + j],
                                    offset);
            /*
             * y0: 00 01 02 03 04 05 06 07
             * y1: 10 11 12 13 14 15 16 17
             */
            y0 = svuzp1(x0, x1);
            y1 = svuzp2(x0, x1);

            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], y0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], y1);
          }
        }
      }
    } else if (block_cols == 4) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3;
            svuint64_t y0, y1, y2, y3;

            /*
             * x0: 00 10 20 30 01 11 21 31
             * x1: 02 12 22 32 03 13 23 33
             * x2: 04 14 24 34 05 15 25 35
             * x3: 06 16 26 36 07 17 27 37
             */
            x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 0) + j],
                                    offset);
            x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 2) + j],
                                    offset);
            x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 4) + j],
                                    offset);
            x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 6) + j],
                                    offset);

            /*
             * y0: 00 20 01 21 02 22 03 23
             * y1: 10 30 11 31 12 32 13 33
             * y2: 04 24 05 25 06 26 07 27
             * y3: 14 34 15 35 16 36 17 37
             */
            y0 = svuzp1(x0, x1);
            y1 = svuzp2(x0, x1);
            y2 = svuzp1(x2, x3);
            y3 = svuzp2(x2, x3);

            /*
             * x0: 00 01 02 03 04 05 06 07
             * x1: 10 11 12 13 14 15 16 17
             * x2: 20 21 22 23 24 25 26 27
             * x3: 30 31 32 33 34 35 36 37
             */
            x0 = svuzp1(y0, y2);
            x1 = svuzp1(y1, y3);
            x2 = svuzp2(y0, y2);
            x3 = svuzp2(y1, y3);

            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
          }
        }
      }
    } else if (block_cols == 8) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += cntd) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

            x0 = svld1(ptrue, &src[ld_src * (i + ii + 0) + j]);
            x1 = svld1(ptrue, &src[ld_src * (i + ii + 1) + j]);
            x2 = svld1(ptrue, &src[ld_src * (i + ii + 2) + j]);
            x3 = svld1(ptrue, &src[ld_src * (i + ii + 3) + j]);
            x4 = svld1(ptrue, &src[ld_src * (i + ii + 4) + j]);
            x5 = svld1(ptrue, &src[ld_src * (i + ii + 5) + j]);
            x6 = svld1(ptrue, &src[ld_src * (i + ii + 6) + j]);
            x7 = svld1(ptrue, &src[ld_src * (i + ii + 7) + j]);

            transpose_8x8(x0, x1, x2, x3, x4, x5, x6, x7);

            svst1(ptrue, &dst[ld_dst * (j + 0) + i + ii], x0);
            svst1(ptrue, &dst[ld_dst * (j + 1) + i + ii], x1);
            svst1(ptrue, &dst[ld_dst * (j + 2) + i + ii], x2);
            svst1(ptrue, &dst[ld_dst * (j + 3) + i + ii], x3);
            svst1(ptrue, &dst[ld_dst * (j + 4) + i + ii], x4);
            svst1(ptrue, &dst[ld_dst * (j + 5) + i + ii], x5);
            svst1(ptrue, &dst[ld_dst * (j + 6) + i + ii], x6);
            svst1(ptrue, &dst[ld_dst * (j + 7) + i + ii], x7);
          }
        }
      }
    } else {
      throw std::invalid_argument{"Unsupported block_cols"};
    }
  }
};

} // namespace sventt

#endif /* SVENTT_TRANSPOSITION_SVE_GATHER_VECTOR_INDEX_ROW_FIRST_HPP_INCLUDED  \
        */

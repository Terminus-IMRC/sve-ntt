// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_TRANSPOSITION_SVE_IN_REGISTER_HPP_INCLUDED
#define SVENTT_TRANSPOSITION_SVE_IN_REGISTER_HPP_INCLUDED

#include <cstdint>
#include <stdexcept>

#include <arm_sve.h>

#include "sventt/common/sve.hpp"

namespace sventt {

template <std::uint64_t block_rows, std::uint64_t block_cols>
class TransposeSVEInRegister {

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
    if (block_rows % cntd != 0 || block_cols % cntd != 0) {
      throw std::invalid_argument{
          "Block dimensions are not divisible by SVE vector length"};
    }

    if constexpr (cntd == 2) {
      for (std::uint64_t i{}; i < src_rows; i += block_rows) {
        for (std::uint64_t j{}; j < src_cols; j += block_cols) {
          for (std::uint64_t ii{}; ii < block_rows; ii += 2) {
            for (std::uint64_t jj{}; jj < block_cols; jj += 2) {
              svuint64_t x0, x1;

              x0 = svld1(ptrue, &src[ld_src * (i + ii + 0) + j + jj]);
              x1 = svld1(ptrue, &src[ld_src * (i + ii + 1) + j + jj]);

              transpose_2x2(x0, x1);

              svst1(ptrue, &dst[ld_dst * (j + jj + 0) + i + ii], x0);
              svst1(ptrue, &dst[ld_dst * (j + jj + 1) + i + ii], x1);
            }
          }
        }
      }
    } else if constexpr (cntd == 4) {
      for (std::uint64_t i{}; i < src_rows; i += block_rows) {
        for (std::uint64_t j{}; j < src_cols; j += block_cols) {
          for (std::uint64_t ii{}; ii < block_rows; ii += 4) {
            for (std::uint64_t jj{}; jj < block_cols; jj += 4) {
              svuint64_t x0, x1, x2, x3;

              x0 = svld1(ptrue, &src[ld_src * (i + ii + 0) + j + jj]);
              x1 = svld1(ptrue, &src[ld_src * (i + ii + 1) + j + jj]);
              x2 = svld1(ptrue, &src[ld_src * (i + ii + 2) + j + jj]);
              x3 = svld1(ptrue, &src[ld_src * (i + ii + 3) + j + jj]);

              transpose_4x4_lazy(x0, x1, x2, x3);

              svst1(ptrue, &dst[ld_dst * (j + jj + 0) + i + ii], x0);
              svst1(ptrue, &dst[ld_dst * (j + jj + 1) + i + ii], x1);
              svst1(ptrue, &dst[ld_dst * (j + jj + 2) + i + ii], x2);
              svst1(ptrue, &dst[ld_dst * (j + jj + 3) + i + ii], x3);
            }
          }
        }
      }
    } else if constexpr (cntd == 8) {
      for (std::uint64_t i{}; i < src_rows; i += block_rows) {
        for (std::uint64_t j{}; j < src_cols; j += block_cols) {
          for (std::uint64_t ii{}; ii < block_rows; ii += 8) {
            for (std::uint64_t jj{}; jj < block_cols; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

              x0 = svld1(ptrue, &src[ld_src * (i + ii + 0) + j + jj]);
              x1 = svld1(ptrue, &src[ld_src * (i + ii + 1) + j + jj]);
              x2 = svld1(ptrue, &src[ld_src * (i + ii + 2) + j + jj]);
              x3 = svld1(ptrue, &src[ld_src * (i + ii + 3) + j + jj]);
              x4 = svld1(ptrue, &src[ld_src * (i + ii + 4) + j + jj]);
              x5 = svld1(ptrue, &src[ld_src * (i + ii + 5) + j + jj]);
              x6 = svld1(ptrue, &src[ld_src * (i + ii + 6) + j + jj]);
              x7 = svld1(ptrue, &src[ld_src * (i + ii + 7) + j + jj]);

              transpose_8x8(x0, x1, x2, x3, x4, x5, x6, x7);

              svst1(ptrue, &dst[ld_dst * (j + jj + 0) + i + ii], x0);
              svst1(ptrue, &dst[ld_dst * (j + jj + 1) + i + ii], x1);
              svst1(ptrue, &dst[ld_dst * (j + jj + 2) + i + ii], x2);
              svst1(ptrue, &dst[ld_dst * (j + jj + 3) + i + ii], x3);
              svst1(ptrue, &dst[ld_dst * (j + jj + 4) + i + ii], x4);
              svst1(ptrue, &dst[ld_dst * (j + jj + 5) + i + ii], x5);
              svst1(ptrue, &dst[ld_dst * (j + jj + 6) + i + ii], x6);
              svst1(ptrue, &dst[ld_dst * (j + jj + 7) + i + ii], x7);
            }
          }
        }
      }
    } else {
      throw std::invalid_argument{"Unsupported SVE vector length"};
    }
  }
};

template <std::uint64_t block_rows, std::uint64_t block_cols>
class TransposeParallelSVEInRegister {

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
    if (block_rows % cntd != 0 || block_cols % cntd != 0) {
      throw std::invalid_argument{
          "Block dimensions are not divisible by SVE vector length"};
    }

    /* TODO: Support block_rows != block_cols cases? */
    if (block_rows == block_cols && dst == src && src_rows == src_cols &&
        ld_dst == src_rows && ld_src == src_cols) {
      transpose(dst, src_rows);
      return;
    }

    if constexpr (cntd == 2) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += 2) {
            for (std::uint64_t jj = 0; jj < block_cols; jj += 2) {
              svuint64_t x0, x1;

              x0 = svld1(ptrue, &src[ld_src * (i + ii + 0) + j + jj]);
              x1 = svld1(ptrue, &src[ld_src * (i + ii + 1) + j + jj]);

              transpose_2x2(x0, x1);

              svst1(ptrue, &dst[ld_dst * (j + jj + 0) + i + ii], x0);
              svst1(ptrue, &dst[ld_dst * (j + jj + 1) + i + ii], x1);
            }
          }
        }
      }
    } else if constexpr (cntd == 4) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += 4) {
            for (std::uint64_t jj = 0; jj < block_cols; jj += 4) {
              svuint64_t x0, x1, x2, x3;

              x0 = svld1(ptrue, &src[ld_src * (i + ii + 0) + j + jj]);
              x1 = svld1(ptrue, &src[ld_src * (i + ii + 1) + j + jj]);
              x2 = svld1(ptrue, &src[ld_src * (i + ii + 2) + j + jj]);
              x3 = svld1(ptrue, &src[ld_src * (i + ii + 3) + j + jj]);

              transpose_4x4_lazy(x0, x1, x2, x3);

              svst1(ptrue, &dst[ld_dst * (j + jj + 0) + i + ii], x0);
              svst1(ptrue, &dst[ld_dst * (j + jj + 1) + i + ii], x1);
              svst1(ptrue, &dst[ld_dst * (j + jj + 2) + i + ii], x2);
              svst1(ptrue, &dst[ld_dst * (j + jj + 3) + i + ii], x3);
            }
          }
        }
      }
    } else if constexpr (cntd == 8) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
        for (std::uint64_t j = 0; j < src_cols; j += block_cols) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < block_cols; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

              x0 = svld1(ptrue, &src[ld_src * (i + ii + 0) + j + jj]);
              x1 = svld1(ptrue, &src[ld_src * (i + ii + 1) + j + jj]);
              x2 = svld1(ptrue, &src[ld_src * (i + ii + 2) + j + jj]);
              x3 = svld1(ptrue, &src[ld_src * (i + ii + 3) + j + jj]);
              x4 = svld1(ptrue, &src[ld_src * (i + ii + 4) + j + jj]);
              x5 = svld1(ptrue, &src[ld_src * (i + ii + 5) + j + jj]);
              x6 = svld1(ptrue, &src[ld_src * (i + ii + 6) + j + jj]);
              x7 = svld1(ptrue, &src[ld_src * (i + ii + 7) + j + jj]);

              transpose_8x8(x0, x1, x2, x3, x4, x5, x6, x7);

              svst1(ptrue, &dst[ld_dst * (j + jj + 0) + i + ii], x0);
              svst1(ptrue, &dst[ld_dst * (j + jj + 1) + i + ii], x1);
              svst1(ptrue, &dst[ld_dst * (j + jj + 2) + i + ii], x2);
              svst1(ptrue, &dst[ld_dst * (j + jj + 3) + i + ii], x3);
              svst1(ptrue, &dst[ld_dst * (j + jj + 4) + i + ii], x4);
              svst1(ptrue, &dst[ld_dst * (j + jj + 5) + i + ii], x5);
              svst1(ptrue, &dst[ld_dst * (j + jj + 6) + i + ii], x6);
              svst1(ptrue, &dst[ld_dst * (j + jj + 7) + i + ii], x7);
            }
          }
        }
      }
    } else {
      throw std::invalid_argument{"Unsupported SVE vector length"};
    }
  }

  static void transpose(std::uint64_t *const dst, const std::uint64_t dim) {
    const svbool_t ptrue{svptrue_b8()};

    if (block_rows != block_cols) {
      throw std::invalid_argument{
          "Block dimensions need to be equal at least for now"};
    }
    if (dim % block_rows != 0 || dim % block_cols != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }
    if (block_rows % cntd != 0 || block_cols % cntd != 0) {
      throw std::invalid_argument{
          "Block dimensions are not divisible by SVE vector length"};
    }

    if constexpr (cntd == 8) {
      [[omp::directive(for, collapse(2))]]
      for (std::uint64_t i = 0; i < dim; i += block_rows) {
        for (std::uint64_t j = 0; j <= i; j += block_cols) {
          if (j < i) {
            for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
              for (std::uint64_t jj = 0; jj < block_cols; jj += 8) {
                svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
                svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

                x0 = svld1(ptrue, &dst[dim * (i + ii + 0) + j + jj]);
                x1 = svld1(ptrue, &dst[dim * (i + ii + 1) + j + jj]);
                x2 = svld1(ptrue, &dst[dim * (i + ii + 2) + j + jj]);
                x3 = svld1(ptrue, &dst[dim * (i + ii + 3) + j + jj]);
                x4 = svld1(ptrue, &dst[dim * (i + ii + 4) + j + jj]);
                x5 = svld1(ptrue, &dst[dim * (i + ii + 5) + j + jj]);
                x6 = svld1(ptrue, &dst[dim * (i + ii + 6) + j + jj]);
                x7 = svld1(ptrue, &dst[dim * (i + ii + 7) + j + jj]);
                transpose_8x8(x0, x1, x2, x3, x4, x5, x6, x7);

                y0 = svld1(ptrue, &dst[dim * (j + jj + 0) + i + ii]);
                y1 = svld1(ptrue, &dst[dim * (j + jj + 1) + i + ii]);
                y2 = svld1(ptrue, &dst[dim * (j + jj + 2) + i + ii]);
                y3 = svld1(ptrue, &dst[dim * (j + jj + 3) + i + ii]);
                y4 = svld1(ptrue, &dst[dim * (j + jj + 4) + i + ii]);
                y5 = svld1(ptrue, &dst[dim * (j + jj + 5) + i + ii]);
                y6 = svld1(ptrue, &dst[dim * (j + jj + 6) + i + ii]);
                y7 = svld1(ptrue, &dst[dim * (j + jj + 7) + i + ii]);
                transpose_8x8(y0, y1, y2, y3, y4, y5, y6, y7);

                svst1(ptrue, &dst[dim * (j + jj + 0) + i + ii], x0);
                svst1(ptrue, &dst[dim * (j + jj + 1) + i + ii], x1);
                svst1(ptrue, &dst[dim * (j + jj + 2) + i + ii], x2);
                svst1(ptrue, &dst[dim * (j + jj + 3) + i + ii], x3);
                svst1(ptrue, &dst[dim * (j + jj + 4) + i + ii], x4);
                svst1(ptrue, &dst[dim * (j + jj + 5) + i + ii], x5);
                svst1(ptrue, &dst[dim * (j + jj + 6) + i + ii], x6);
                svst1(ptrue, &dst[dim * (j + jj + 7) + i + ii], x7);

                svst1(ptrue, &dst[dim * (i + ii + 0) + j + jj], y0);
                svst1(ptrue, &dst[dim * (i + ii + 1) + j + jj], y1);
                svst1(ptrue, &dst[dim * (i + ii + 2) + j + jj], y2);
                svst1(ptrue, &dst[dim * (i + ii + 3) + j + jj], y3);
                svst1(ptrue, &dst[dim * (i + ii + 4) + j + jj], y4);
                svst1(ptrue, &dst[dim * (i + ii + 5) + j + jj], y5);
                svst1(ptrue, &dst[dim * (i + ii + 6) + j + jj], y6);
                svst1(ptrue, &dst[dim * (i + ii + 7) + j + jj], y7);
              }
            }
          } else {
            /* The remaining i = j case. */
            for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
              for (std::uint64_t jj = 0; jj < ii; jj += 8) {
                svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
                svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

                x0 = svld1(ptrue, &dst[dim * (i + ii + 0) + i + jj]);
                x1 = svld1(ptrue, &dst[dim * (i + ii + 1) + i + jj]);
                x2 = svld1(ptrue, &dst[dim * (i + ii + 2) + i + jj]);
                x3 = svld1(ptrue, &dst[dim * (i + ii + 3) + i + jj]);
                x4 = svld1(ptrue, &dst[dim * (i + ii + 4) + i + jj]);
                x5 = svld1(ptrue, &dst[dim * (i + ii + 5) + i + jj]);
                x6 = svld1(ptrue, &dst[dim * (i + ii + 6) + i + jj]);
                x7 = svld1(ptrue, &dst[dim * (i + ii + 7) + i + jj]);
                transpose_8x8(x0, x1, x2, x3, x4, x5, x6, x7);

                y0 = svld1(ptrue, &dst[dim * (i + jj + 0) + i + ii]);
                y1 = svld1(ptrue, &dst[dim * (i + jj + 1) + i + ii]);
                y2 = svld1(ptrue, &dst[dim * (i + jj + 2) + i + ii]);
                y3 = svld1(ptrue, &dst[dim * (i + jj + 3) + i + ii]);
                y4 = svld1(ptrue, &dst[dim * (i + jj + 4) + i + ii]);
                y5 = svld1(ptrue, &dst[dim * (i + jj + 5) + i + ii]);
                y6 = svld1(ptrue, &dst[dim * (i + jj + 6) + i + ii]);
                y7 = svld1(ptrue, &dst[dim * (i + jj + 7) + i + ii]);
                transpose_8x8(y0, y1, y2, y3, y4, y5, y6, y7);

                svst1(ptrue, &dst[dim * (i + jj + 0) + i + ii], x0);
                svst1(ptrue, &dst[dim * (i + jj + 1) + i + ii], x1);
                svst1(ptrue, &dst[dim * (i + jj + 2) + i + ii], x2);
                svst1(ptrue, &dst[dim * (i + jj + 3) + i + ii], x3);
                svst1(ptrue, &dst[dim * (i + jj + 4) + i + ii], x4);
                svst1(ptrue, &dst[dim * (i + jj + 5) + i + ii], x5);
                svst1(ptrue, &dst[dim * (i + jj + 6) + i + ii], x6);
                svst1(ptrue, &dst[dim * (i + jj + 7) + i + ii], x7);

                svst1(ptrue, &dst[dim * (i + ii + 0) + i + jj], y0);
                svst1(ptrue, &dst[dim * (i + ii + 1) + i + jj], y1);
                svst1(ptrue, &dst[dim * (i + ii + 2) + i + jj], y2);
                svst1(ptrue, &dst[dim * (i + ii + 3) + i + jj], y3);
                svst1(ptrue, &dst[dim * (i + ii + 4) + i + jj], y4);
                svst1(ptrue, &dst[dim * (i + ii + 5) + i + jj], y5);
                svst1(ptrue, &dst[dim * (i + ii + 6) + i + jj], y6);
                svst1(ptrue, &dst[dim * (i + ii + 7) + i + jj], y7);
              }

              /* The remaining ii = jj case. */
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
              svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

              x0 = svld1(ptrue, &dst[dim * (i + ii + 0) + i + ii]);
              x1 = svld1(ptrue, &dst[dim * (i + ii + 1) + i + ii]);
              x2 = svld1(ptrue, &dst[dim * (i + ii + 2) + i + ii]);
              x3 = svld1(ptrue, &dst[dim * (i + ii + 3) + i + ii]);
              x4 = svld1(ptrue, &dst[dim * (i + ii + 4) + i + ii]);
              x5 = svld1(ptrue, &dst[dim * (i + ii + 5) + i + ii]);
              x6 = svld1(ptrue, &dst[dim * (i + ii + 6) + i + ii]);
              x7 = svld1(ptrue, &dst[dim * (i + ii + 7) + i + ii]);
              transpose_8x8(x0, x1, x2, x3, x4, x5, x6, x7);

              y0 = svld1(ptrue, &dst[dim * (i + ii + 0) + i + ii]);
              y1 = svld1(ptrue, &dst[dim * (i + ii + 1) + i + ii]);
              y2 = svld1(ptrue, &dst[dim * (i + ii + 2) + i + ii]);
              y3 = svld1(ptrue, &dst[dim * (i + ii + 3) + i + ii]);
              y4 = svld1(ptrue, &dst[dim * (i + ii + 4) + i + ii]);
              y5 = svld1(ptrue, &dst[dim * (i + ii + 5) + i + ii]);
              y6 = svld1(ptrue, &dst[dim * (i + ii + 6) + i + ii]);
              y7 = svld1(ptrue, &dst[dim * (i + ii + 7) + i + ii]);
              transpose_8x8(y0, y1, y2, y3, y4, y5, y6, y7);

              svst1(ptrue, &dst[dim * (i + ii + 0) + i + ii], x0);
              svst1(ptrue, &dst[dim * (i + ii + 1) + i + ii], x1);
              svst1(ptrue, &dst[dim * (i + ii + 2) + i + ii], x2);
              svst1(ptrue, &dst[dim * (i + ii + 3) + i + ii], x3);
              svst1(ptrue, &dst[dim * (i + ii + 4) + i + ii], x4);
              svst1(ptrue, &dst[dim * (i + ii + 5) + i + ii], x5);
              svst1(ptrue, &dst[dim * (i + ii + 6) + i + ii], x6);
              svst1(ptrue, &dst[dim * (i + ii + 7) + i + ii], x7);

              svst1(ptrue, &dst[dim * (i + ii + 0) + i + ii], y0);
              svst1(ptrue, &dst[dim * (i + ii + 1) + i + ii], y1);
              svst1(ptrue, &dst[dim * (i + ii + 2) + i + ii], y2);
              svst1(ptrue, &dst[dim * (i + ii + 3) + i + ii], y3);
              svst1(ptrue, &dst[dim * (i + ii + 4) + i + ii], y4);
              svst1(ptrue, &dst[dim * (i + ii + 5) + i + ii], y5);
              svst1(ptrue, &dst[dim * (i + ii + 6) + i + ii], y6);
              svst1(ptrue, &dst[dim * (i + ii + 7) + i + ii], y7);
            }
          }
        }
      }
    } else {
      throw std::invalid_argument{"Unsupported SVE vector length"};
    }
  }
};

} // namespace sventt

#endif /* SVENTT_TRANSPOSITION_SVE_IN_REGISTER_HPP_INCLUDED */

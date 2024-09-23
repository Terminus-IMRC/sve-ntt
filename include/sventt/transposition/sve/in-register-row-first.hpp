// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_TRANSPOSITION_SVE_IN_REGISTER_ROW_FIRST_HPP_INCLUDED
#define SVENTT_TRANSPOSITION_SVE_IN_REGISTER_ROW_FIRST_HPP_INCLUDED

#include <cstdint>
#include <stdexcept>

#include <arm_sve.h>

#include "sventt/common/sve.hpp"

namespace sventt {

template <std::uint64_t num_shuffle_stages> class TransposeSVEInRegisterBase {

  static_assert(0 <= num_shuffle_stages && num_shuffle_stages <= 3);

public:
  static void
  load_subblock(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2, svuint64_t &x3,
                svuint64_t &x4, svuint64_t &x5, svuint64_t &x6, svuint64_t &x7,
                const std::uint64_t *const src, const std::uint64_t ld_src,
                const std::uint64_t i, const std::uint64_t j,
                const std::uint64_t ii, const std::uint64_t jj)
    requires(num_shuffle_stages == 0)
  {
    const svbool_t ptrue{svptrue_b8()};

    const svuint64_t offset{svindex_u64(0, ld_src)};

    x0 =
        svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + jj + 0], offset);
    x1 =
        svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + jj + 1], offset);
    x2 =
        svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + jj + 2], offset);
    x3 =
        svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + jj + 3], offset);
    x4 =
        svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + jj + 4], offset);
    x5 =
        svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + jj + 5], offset);
    x6 =
        svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + jj + 6], offset);
    x7 =
        svld1_gather_index(ptrue, &src[ld_src * (i + ii) + j + jj + 7], offset);
  }

  static void
  load_subblock(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2, svuint64_t &x3,
                svuint64_t &x4, svuint64_t &x5, svuint64_t &x6, svuint64_t &x7,
                const std::uint64_t *const src, const std::uint64_t ld_src,
                const std::uint64_t i, const std::uint64_t j,
                const std::uint64_t ii, const std::uint64_t jj)
    requires(num_shuffle_stages == 1)
  {
    const svbool_t ptrue{svptrue_b8()};

    const svuint64_t offset{svmla_m(ptrue, svand_x(ptrue, svindex_u64(0, 1), 1),
                                    svlsr_x(ptrue, svindex_u64(0, 1), 1),
                                    ld_src)};

    svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

    y0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 0) + j + jj + 0],
                            offset);
    y1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 0) + j + jj + 2],
                            offset);
    y2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 0) + j + jj + 4],
                            offset);
    y3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 0) + j + jj + 6],
                            offset);
    y4 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 4) + j + jj + 0],
                            offset);
    y5 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 4) + j + jj + 2],
                            offset);
    y6 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 4) + j + jj + 4],
                            offset);
    y7 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 4) + j + jj + 6],
                            offset);

    x0 = svuzp1(y0, y4);
    x1 = svuzp1(y1, y5);
    x2 = svuzp1(y2, y6);
    x3 = svuzp1(y3, y7);
    x4 = svuzp2(y0, y4);
    x5 = svuzp2(y1, y5);
    x6 = svuzp2(y2, y6);
    x7 = svuzp2(y3, y7);
  }

  static void
  load_subblock(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2, svuint64_t &x3,
                svuint64_t &x4, svuint64_t &x5, svuint64_t &x6, svuint64_t &x7,
                const std::uint64_t *const src, const std::uint64_t ld_src,
                const std::uint64_t i, const std::uint64_t j,
                const std::uint64_t ii, const std::uint64_t jj)
    requires(num_shuffle_stages == 2)
  {
    const svbool_t ptrue{svptrue_b8()};

    const svuint64_t offset{svmla_m(ptrue, svand_x(ptrue, svindex_u64(0, 1), 3),
                                    svlsr_x(ptrue, svindex_u64(0, 1), 2),
                                    ld_src)};

    svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

    x0 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 0) + j + jj + 0],
                            offset);
    x1 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 0) + j + jj + 4],
                            offset);
    x2 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 2) + j + jj + 0],
                            offset);
    x3 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 2) + j + jj + 4],
                            offset);
    x4 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 4) + j + jj + 0],
                            offset);
    x5 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 4) + j + jj + 4],
                            offset);
    x6 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 6) + j + jj + 0],
                            offset);
    x7 = svld1_gather_index(ptrue, &src[ld_src * (i + ii + 6) + j + jj + 4],
                            offset);

    y0 = svuzp1(x0, x2);
    y1 = svuzp1(x1, x3);
    y2 = svuzp2(x0, x2);
    y3 = svuzp2(x1, x3);
    y4 = svuzp1(x4, x6);
    y5 = svuzp1(x5, x7);
    y6 = svuzp2(x4, x6);
    y7 = svuzp2(x5, x7);

    x0 = svuzp1(y0, y4);
    x1 = svuzp1(y1, y5);
    x2 = svuzp1(y2, y6);
    x3 = svuzp1(y3, y7);
    x4 = svuzp2(y0, y4);
    x5 = svuzp2(y1, y5);
    x6 = svuzp2(y2, y6);
    x7 = svuzp2(y3, y7);
  }

  static void
  load_subblock(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2, svuint64_t &x3,
                svuint64_t &x4, svuint64_t &x5, svuint64_t &x6, svuint64_t &x7,
                const std::uint64_t *const src, const std::uint64_t ld_src,
                const std::uint64_t i, const std::uint64_t j,
                const std::uint64_t ii, const std::uint64_t jj)
    requires(num_shuffle_stages == 3)
  {
    const svbool_t ptrue{svptrue_b8()};

    x0 = svld1(ptrue, &src[ld_src * (i + ii + 0) + j + jj]);
    x1 = svld1(ptrue, &src[ld_src * (i + ii + 1) + j + jj]);
    x2 = svld1(ptrue, &src[ld_src * (i + ii + 2) + j + jj]);
    x3 = svld1(ptrue, &src[ld_src * (i + ii + 3) + j + jj]);
    x4 = svld1(ptrue, &src[ld_src * (i + ii + 4) + j + jj]);
    x5 = svld1(ptrue, &src[ld_src * (i + ii + 5) + j + jj]);
    x6 = svld1(ptrue, &src[ld_src * (i + ii + 6) + j + jj]);
    x7 = svld1(ptrue, &src[ld_src * (i + ii + 7) + j + jj]);

    transpose_8x8(x0, x1, x2, x3, x4, x5, x6, x7);
  }

  static void
  store_subblock(std::uint64_t *const dst, const std::uint64_t ld_dst,
                 const std::uint64_t i, const std::uint64_t j,
                 const std::uint64_t ii, const std::uint64_t jj,
                 const svuint64_t x0, const svuint64_t x1, const svuint64_t x2,
                 const svuint64_t x3, const svuint64_t x4, const svuint64_t x5,
                 const svuint64_t x6, const svuint64_t x7)
    requires(num_shuffle_stages == 0)
  {
    const svbool_t ptrue{svptrue_b8()};

    svst1(ptrue, &dst[ld_dst * (j + jj + 0) + i + ii], x0);
    svst1(ptrue, &dst[ld_dst * (j + jj + 1) + i + ii], x1);
    svst1(ptrue, &dst[ld_dst * (j + jj + 2) + i + ii], x2);
    svst1(ptrue, &dst[ld_dst * (j + jj + 3) + i + ii], x3);
    svst1(ptrue, &dst[ld_dst * (j + jj + 4) + i + ii], x4);
    svst1(ptrue, &dst[ld_dst * (j + jj + 5) + i + ii], x5);
    svst1(ptrue, &dst[ld_dst * (j + jj + 6) + i + ii], x6);
    svst1(ptrue, &dst[ld_dst * (j + jj + 7) + i + ii], x7);
  }

  static void
  store_subblock(std::uint64_t *const dst, const std::uint64_t ld_dst,
                 const std::uint64_t i, const std::uint64_t j,
                 const std::uint64_t ii, const std::uint64_t jj,
                 const svuint64_t x0, const svuint64_t x1, const svuint64_t x2,
                 const svuint64_t x3, const svuint64_t x4, const svuint64_t x5,
                 const svuint64_t x6, const svuint64_t x7)
    requires(num_shuffle_stages == 1)
  {
    const svbool_t ptrue{svptrue_b8()};

    svst1(ptrue, &dst[ld_dst * (j + jj + 0) + i + ii], x0);
    svst1(ptrue, &dst[ld_dst * (j + jj + 1) + i + ii], x4);
    svst1(ptrue, &dst[ld_dst * (j + jj + 2) + i + ii], x1);
    svst1(ptrue, &dst[ld_dst * (j + jj + 3) + i + ii], x5);
    svst1(ptrue, &dst[ld_dst * (j + jj + 4) + i + ii], x2);
    svst1(ptrue, &dst[ld_dst * (j + jj + 5) + i + ii], x6);
    svst1(ptrue, &dst[ld_dst * (j + jj + 6) + i + ii], x3);
    svst1(ptrue, &dst[ld_dst * (j + jj + 7) + i + ii], x7);
  }

  static void
  store_subblock(std::uint64_t *const dst, const std::uint64_t ld_dst,
                 const std::uint64_t i, const std::uint64_t j,
                 const std::uint64_t ii, const std::uint64_t jj,
                 const svuint64_t x0, const svuint64_t x1, const svuint64_t x2,
                 const svuint64_t x3, const svuint64_t x4, const svuint64_t x5,
                 const svuint64_t x6, const svuint64_t x7)
    requires(num_shuffle_stages == 2)
  {
    const svbool_t ptrue{svptrue_b8()};

    svst1(ptrue, &dst[ld_dst * (j + jj + 0) + i + ii], x0);
    svst1(ptrue, &dst[ld_dst * (j + jj + 1) + i + ii], x2);
    svst1(ptrue, &dst[ld_dst * (j + jj + 2) + i + ii], x4);
    svst1(ptrue, &dst[ld_dst * (j + jj + 3) + i + ii], x6);
    svst1(ptrue, &dst[ld_dst * (j + jj + 4) + i + ii], x1);
    svst1(ptrue, &dst[ld_dst * (j + jj + 5) + i + ii], x3);
    svst1(ptrue, &dst[ld_dst * (j + jj + 6) + i + ii], x5);
    svst1(ptrue, &dst[ld_dst * (j + jj + 7) + i + ii], x7);
  }

  static void
  store_subblock(std::uint64_t *const dst, const std::uint64_t ld_dst,
                 const std::uint64_t i, const std::uint64_t j,
                 const std::uint64_t ii, const std::uint64_t jj,
                 const svuint64_t x0, const svuint64_t x1, const svuint64_t x2,
                 const svuint64_t x3, const svuint64_t x4, const svuint64_t x5,
                 const svuint64_t x6, const svuint64_t x7)
    requires(num_shuffle_stages == 3)
  {
    const svbool_t ptrue{svptrue_b8()};

    svst1(ptrue, &dst[ld_dst * (j + jj + 0) + i + ii], x0);
    svst1(ptrue, &dst[ld_dst * (j + jj + 1) + i + ii], x1);
    svst1(ptrue, &dst[ld_dst * (j + jj + 2) + i + ii], x2);
    svst1(ptrue, &dst[ld_dst * (j + jj + 3) + i + ii], x3);
    svst1(ptrue, &dst[ld_dst * (j + jj + 4) + i + ii], x4);
    svst1(ptrue, &dst[ld_dst * (j + jj + 5) + i + ii], x5);
    svst1(ptrue, &dst[ld_dst * (j + jj + 6) + i + ii], x6);
    svst1(ptrue, &dst[ld_dst * (j + jj + 7) + i + ii], x7);
  }

  static void
  transpose_subblock(std::uint64_t *const dst, const std::uint64_t *const src,
                     const std::uint64_t ld_dst, const std::uint64_t ld_src,
                     const std::uint64_t i, const std::uint64_t j,
                     const std::uint64_t ii, const std::uint64_t jj) {
    svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

    load_subblock(x0, x1, x2, x3, x4, x5, x6, x7, src, ld_src, i, j, ii, jj);

    store_subblock(dst, ld_dst, i, j, ii, jj, x0, x1, x2, x3, x4, x5, x6, x7);
  }
};

template <std::uint64_t block_rows, std::uint64_t block_columns,
          std::uint64_t num_shuffle_stages>
class TransposeSVEInRegisterRowFirst {

  static_assert(block_rows % 8 == 0 && block_columns % 8 == 0);

  using base_type = TransposeSVEInRegisterBase<num_shuffle_stages>;

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    if (src_rows % block_rows != 0 || src_cols % block_columns != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }

    for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
      for (std::uint64_t j = 0; j < src_cols; j += block_columns) {
        for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
          for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
            base_type::transpose_subblock(dst, src, ld_dst, ld_src, i, j, ii,
                                          jj);
          }
        }
      }
    }
  }

  static void transpose(std::uint64_t *const dst, const std::uint64_t dim) {
    if (block_rows != block_columns) {
      throw std::invalid_argument{
          "Block dimensions need to be equal at least for now"};
    }
    if (dim % block_rows != 0 || dim % block_columns != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }

    for (std::uint64_t i = 0; i < dim; i += block_rows) {
      for (std::uint64_t j = 0; j <= i; j += block_columns) {
        if (j < i) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
              svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7, dst, dim,
                                       i, j, ii, jj);
              base_type::load_subblock(y0, y1, y2, y3, y4, y5, y6, y7, dst, dim,
                                       j, i, jj, ii);

              base_type::store_subblock(dst, dim, i, j, ii, jj, x0, x1, x2, x3,
                                        x4, x5, x6, x7);
              base_type::store_subblock(dst, dim, j, i, jj, ii, y0, y1, y2, y3,
                                        y4, y5, y6, y7);
            }
          }
        } else {
          /* The remaining i = j case. */
          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < ii; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
              svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7, dst, dim,
                                       i, i, ii, jj);
              base_type::load_subblock(y0, y1, y2, y3, y4, y5, y6, y7, dst, dim,
                                       i, i, jj, ii);

              base_type::store_subblock(dst, dim, i, i, ii, jj, x0, x1, x2, x3,
                                        x4, x5, x6, x7);
              base_type::store_subblock(dst, dim, i, i, jj, ii, y0, y1, y2, y3,
                                        y4, y5, y6, y7);
            }

            /* The remaining ii = jj case. */
            base_type::transpose_subblock(dst, dst, dim, dim, i, i, ii, ii);
          }
        }
      }
    }
  }
};

template <std::uint64_t block_rows, std::uint64_t block_columns,
          std::uint64_t num_shuffle_stages>
class TransposeParallelSVEInRegisterRowFirst {

  static_assert(block_rows % 8 == 0 && block_columns % 8 == 0);

  using base_type = TransposeSVEInRegisterBase<num_shuffle_stages>;

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    if (src_rows % block_rows != 0 || src_cols % block_columns != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }

    [[omp::directive(for, collapse(2))]]
    for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
      for (std::uint64_t j = 0; j < src_cols; j += block_columns) {
        for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
          for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
            base_type::transpose_subblock(dst, src, ld_dst, ld_src, i, j, ii,
                                          jj);
          }
        }
      }
    }
  }

  static void transpose(std::uint64_t *const dst, const std::uint64_t dim) {
    if (block_rows != block_columns) {
      throw std::invalid_argument{
          "Block dimensions need to be equal at least for now"};
    }
    if (dim % block_rows != 0 || dim % block_columns != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }

    [[omp::directive(for, collapse(2))]]
    for (std::uint64_t i = 0; i < dim; i += block_rows) {
      for (std::uint64_t j = 0; j <= i; j += block_columns) {
        if (j < i) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
              svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7, dst, dim,
                                       i, j, ii, jj);
              base_type::load_subblock(y0, y1, y2, y3, y4, y5, y6, y7, dst, dim,
                                       j, i, jj, ii);

              base_type::store_subblock(dst, dim, i, j, ii, jj, x0, x1, x2, x3,
                                        x4, x5, x6, x7);
              base_type::store_subblock(dst, dim, j, i, jj, ii, y0, y1, y2, y3,
                                        y4, y5, y6, y7);
            }
          }
        } else {
          /* The remaining i = j case. */
          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < ii; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
              svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7, dst, dim,
                                       i, i, ii, jj);
              base_type::load_subblock(y0, y1, y2, y3, y4, y5, y6, y7, dst, dim,
                                       i, i, jj, ii);

              base_type::store_subblock(dst, dim, i, i, ii, jj, x0, x1, x2, x3,
                                        x4, x5, x6, x7);
              base_type::store_subblock(dst, dim, i, i, jj, ii, y0, y1, y2, y3,
                                        y4, y5, y6, y7);
            }

            /* The remaining ii = jj case. */
            base_type::transpose_subblock(dst, dst, dim, dim, i, i, ii, ii);
          }
        }
      }
    }
  }
};

} // namespace sventt

#endif /* SVENTT_TRANSPOSITION_SVE_IN_REGISTER_ROW_FIRST_HPP_INCLUDED */

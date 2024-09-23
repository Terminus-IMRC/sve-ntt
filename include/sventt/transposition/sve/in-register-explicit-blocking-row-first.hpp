// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_TRANSPOSITION_SVE_IN_REGISTER_EXPLICIT_BLOCKING_ROW_FIRST_HPP_INCLUDED
#define SVENTT_TRANSPOSITION_SVE_IN_REGISTER_EXPLICIT_BLOCKING_ROW_FIRST_HPP_INCLUDED

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "sventt/copy/sve/generic.hpp"
#include "sventt/transposition/sve/in-register-row-first.hpp"
#include "sventt/vector.hpp"

namespace sventt {

template <std::uint64_t block_rows, std::uint64_t block_columns,
          std::uint64_t ld_block, std::uint64_t num_shuffle_stages>
class TransposeSVEInRegisterExplicitBlockingRowFirst {

  static_assert(block_rows % 8 == 0 && block_columns % 8 == 0);
  static_assert(ld_block >= block_columns);

  using base_type = TransposeSVEInRegisterBase<num_shuffle_stages>;
  using copy_type = CopySVEGeneric<sizeof(std::uint64_t) * block_columns>;

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    if (src_rows % block_rows != 0 || src_cols % block_columns != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }

    PageMemory<std::uint64_t> block{ld_block * (block_rows - 1) + block_columns,
                                    true};

    for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
      for (std::uint64_t j = 0; j < src_cols; j += block_columns) {
        for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
          copy_type::copy(&block[ld_block * ii], &src[ld_src * (i + ii) + j]);
        }

        for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

            base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7, &block[0],
                                     ld_block, 0, 0, ii, jj);

            base_type::store_subblock(dst, ld_dst, i, j, ii, jj, x0, x1, x2, x3,
                                      x4, x5, x6, x7);
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

    PageMemory<std::uint64_t> block{
        ld_block * (block_rows * 2 - 1) + block_columns, true};
    std::uint64_t *const block0{&block[0]}, *const block1{
                                                &block[ld_block * block_rows]};

    for (std::uint64_t i = 0; i < dim; i += block_rows) {
      for (std::uint64_t j = 0; j <= i; j += block_columns) {
        if (j < i) {
          for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
            copy_type::copy(&block0[ld_block * ii], &dst[dim * (i + ii) + j]);
          }
          for (std::uint64_t jj = 0; jj < block_columns; ++jj) {
            copy_type::copy(&block1[ld_block * jj], &dst[dim * (j + jj) + i]);
          }

          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
              svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7,
                                       &block0[0], ld_block, 0, 0, ii, jj);
              base_type::load_subblock(y0, y1, y2, y3, y4, y5, y6, y7,
                                       &block1[0], ld_block, 0, 0, jj, ii);

              base_type::store_subblock(dst, dim, i, j, ii, jj, x0, x1, x2, x3,
                                        x4, x5, x6, x7);
              base_type::store_subblock(dst, dim, j, i, jj, ii, y0, y1, y2, y3,
                                        y4, y5, y6, y7);
            }
          }
        } else {
          /* The remaining i = j case. */

          for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
            copy_type::copy(&block0[ld_block * ii], &dst[dim * (i + ii) + j]);
          }

          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < ii; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7,
                                       &block0[0], ld_block, 0, 0, ii, jj);

              base_type::store_subblock(dst, dim, i, i, ii, jj, x0, x1, x2, x3,
                                        x4, x5, x6, x7);
            }
          }
        }
      }
    }
  }
};

template <std::uint64_t block_rows, std::uint64_t block_columns,
          std::uint64_t ld_block, std::uint64_t num_shuffle_stages>
class TransposeParallelSVEInRegisterExplicitBlockingRowFirst {

  static_assert(block_rows % 8 == 0 && block_columns % 8 == 0);
  static_assert(ld_block >= block_columns);

  using base_type = TransposeSVEInRegisterBase<num_shuffle_stages>;
  using copy_type = CopySVEGeneric<sizeof(std::uint64_t) * block_columns>;

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    if (src_rows % block_rows != 0 || src_cols % block_columns != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }

    /* TODO: Manage this buffer in a smarter way. */
    thread_local PageMemory<std::uint64_t> block{
        ld_block * (block_rows - 1) + block_columns, true};

    [[omp::directive(for, collapse(2))]]
    for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
      for (std::uint64_t j = 0; j < src_cols; j += block_columns) {
        for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
          copy_type::copy(&block[ld_block * ii], &src[ld_src * (i + ii) + j]);
        }

        for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

            base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7, &block[0],
                                     ld_block, 0, 0, ii, jj);

            base_type::store_subblock(dst, ld_dst, i, j, ii, jj, x0, x1, x2, x3,
                                      x4, x5, x6, x7);
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

    /* TODO: Manage this buffer in a smarter way. */
    thread_local PageMemory<std::uint64_t> block{
        ld_block * (block_rows * 2 - 1) + block_columns, true};
    std::uint64_t *const block0{&block[0]}, *const block1{
                                                &block[ld_block * block_rows]};

    [[omp::directive(for, collapse(2))]]
    for (std::uint64_t i = 0; i < dim; i += block_rows) {
      for (std::uint64_t j = 0; j <= i; j += block_columns) {
        if (j < i) {
          for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
            copy_type::copy(&block0[ld_block * ii], &dst[dim * (i + ii) + j]);
          }
          for (std::uint64_t jj = 0; jj < block_columns; ++jj) {
            copy_type::copy(&block1[ld_block * jj], &dst[dim * (j + jj) + i]);
          }

          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
              svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7,
                                       &block0[0], ld_block, 0, 0, ii, jj);
              base_type::load_subblock(y0, y1, y2, y3, y4, y5, y6, y7,
                                       &block1[0], ld_block, 0, 0, jj, ii);

              base_type::store_subblock(dst, dim, i, j, ii, jj, x0, x1, x2, x3,
                                        x4, x5, x6, x7);
              base_type::store_subblock(dst, dim, j, i, jj, ii, y0, y1, y2, y3,
                                        y4, y5, y6, y7);
            }
          }
        } else {
          /* The remaining i = j case. */

          for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
            copy_type::copy(&block0[ld_block * ii], &dst[dim * (i + ii) + j]);
          }

          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < ii; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
              svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7,
                                       &block0[0], ld_block, 0, 0, ii, jj);
              base_type::load_subblock(y0, y1, y2, y3, y4, y5, y6, y7,
                                       &block0[0], ld_block, 0, 0, jj, ii);

              base_type::store_subblock(dst, dim, i, j, ii, jj, x0, x1, x2, x3,
                                        x4, x5, x6, x7);
              base_type::store_subblock(dst, dim, j, i, jj, ii, y0, y1, y2, y3,
                                        y4, y5, y6, y7);
            }

            /* The remaining ii = jj case. */

            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

            base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7, &block0[0],
                                     ld_block, 0, 0, ii, ii);

            base_type::store_subblock(dst, dim, i, j, ii, ii, x0, x1, x2, x3,
                                      x4, x5, x6, x7);
          }
        }
      }
    }
  }
};

} // namespace sventt

#endif /* SVENTT_TRANSPOSITION_SVE_IN_REGISTER_EXPLICIT_BLOCKING_ROW_FIRST_HPP_INCLUDED \
        */

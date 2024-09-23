// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_TRANSPOSITION_SVE_IN_REGISTER_FULL_BLOCKING_ROW_FIRST_HPP_INCLUDED
#define SVENTT_TRANSPOSITION_SVE_IN_REGISTER_FULL_BLOCKING_ROW_FIRST_HPP_INCLUDED

#include <cstdint>
#include <stdexcept>

#include "sventt/copy/sve/generic.hpp"
#include "sventt/transposition/sve/in-register-row-first.hpp"
#include "sventt/vector.hpp"

namespace sventt {

template <std::uint64_t block_rows, std::uint64_t block_columns,
          std::uint64_t ld_block_row, std::uint64_t ld_block_column,
          std::uint64_t num_shuffle_stages>
class TransposeSVEInRegisterFullBlockingRowFirst {

  static_assert(block_rows % 8 == 0 && block_columns % 8 == 0);
  static_assert(ld_block_row >= block_columns && ld_block_column >= block_rows);

  using base_type = TransposeSVEInRegisterBase<num_shuffle_stages>;
  using copy_src_type = CopySVEGeneric<sizeof(std::uint64_t) * block_columns>;
  using copy_dst_type = CopySVEGeneric<sizeof(std::uint64_t) * block_rows>;

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    if (src_rows % block_rows != 0 || src_cols % block_columns != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }

    PageMemory<std::uint64_t> block{
        (ld_block_row * (block_rows - 1) + block_columns) +
            (ld_block_column * (block_columns - 1) + block_rows),
        !true};
    std::uint64_t *const block_src{&block[0]},
        *const block_dst{
            &block[ld_block_row * (block_rows - 1) + block_columns]};

    for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
      for (std::uint64_t j = 0; j < src_cols; j += block_columns) {
        for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
          copy_src_type::copy(&block_src[ld_block_row * ii],
                              &src[ld_src * (i + ii) + j]);
        }

        for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

            base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7,
                                     &block_src[0], ld_block_row, 0, 0, ii, jj);

            base_type::store_subblock(&block_dst[0], ld_block_column, 0, 0, ii,
                                      jj, x0, x1, x2, x3, x4, x5, x6, x7);
          }
        }

        for (std::uint64_t jj = 0; jj < block_columns; ++jj) {
          copy_dst_type::copy(&dst[ld_dst * (j + jj) + i],
                              &block_dst[ld_block_column * jj]);
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
        (ld_block_row * (block_rows - 1) + block_columns) +
            (ld_block_column * (block_columns - 1) + block_rows),
        !true};
    std::uint64_t *const block_src{&block[0]},
        *const block_dst{
            &block[ld_block_row * (block_rows - 1) + block_columns]};

    for (std::uint64_t i = 0; i < dim; i += block_rows) {
      for (std::uint64_t j = 0; j <= i; j += block_columns) {
        if (j < i) {
          for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
            copy_src_type::copy(&block_src[ld_block_row * ii],
                                &dst[dim * (i + ii) + j]);
          }
          for (std::uint64_t jj = 0; jj < block_columns; ++jj) {
            copy_dst_type::copy(&block_dst[ld_block_column * jj],
                                &dst[dim * (j + jj) + i]);
          }

          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
              svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7,
                                       &block_src[0], ld_block_row, 0, 0, ii,
                                       jj);
              base_type::load_subblock(y0, y1, y2, y3, y4, y5, y6, y7,
                                       &block_dst[0], ld_block_column, 0, 0, jj,
                                       ii);

              base_type::store_subblock(&block_dst[0], ld_block_column, 0, 0,
                                        ii, jj, x0, x1, x2, x3, x4, x5, x6, x7);
              base_type::store_subblock(&block_src[0], ld_block_row, 0, 0, jj,
                                        ii, y0, y1, y2, y3, y4, y5, y6, y7);
            }
          }

          for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
            copy_src_type::copy(&dst[dim * (i + ii) + j],
                                &block_src[ld_block_row * ii]);
          }
          for (std::uint64_t jj = 0; jj < block_columns; ++jj) {
            copy_dst_type::copy(&dst[dim * (j + jj) + i],
                                &block_dst[ld_block_column * jj]);
          }
        } else {
          /* The remaining i = j case. */

          for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
            copy_src_type::copy(&block_src[ld_block_row * ii],
                                &dst[dim * (i + ii) + j]);
          }

          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7,
                                       &block_src[0], ld_block_row, 0, 0, ii,
                                       jj);

              base_type::store_subblock(&block_dst[0], ld_block_column, 0, 0,
                                        ii, jj, x0, x1, x2, x3, x4, x5, x6, x7);
            }
          }

          for (std::uint64_t jj = 0; jj < block_columns; ++jj) {
            copy_dst_type::copy(&dst[dim * (j + jj) + i],
                                &block_dst[ld_block_column * jj]);
          }
        }
      }
    }
  }
};

template <std::uint64_t block_rows, std::uint64_t block_columns,
          std::uint64_t ld_block_row, std::uint64_t ld_block_column,
          std::uint64_t num_shuffle_stages>
class TransposeParallelSVEInRegisterFullBlockingRowFirst {

  static_assert(block_rows % 8 == 0 && block_columns % 8 == 0);
  static_assert(ld_block_row >= block_columns && ld_block_column >= block_rows);

  using base_type = TransposeSVEInRegisterBase<num_shuffle_stages>;
  using copy_src_type = CopySVEGeneric<sizeof(std::uint64_t) * block_columns>;
  using copy_dst_type = CopySVEGeneric<sizeof(std::uint64_t) * block_rows>;

public:
  static void
  transpose(std::uint64_t *const dst, const std::uint64_t *const src,
            const std::uint64_t src_rows, const std::uint64_t src_cols,
            const std::uint64_t ld_dst, const std::uint64_t ld_src) {
    if (src_rows % block_rows != 0 || src_cols % block_columns != 0) {
      throw std::invalid_argument{
          "Matrix dimensions are not divisible by block dimensions"};
    }

    thread_local PageMemory<std::uint64_t> block{
        (ld_block_row * (block_rows - 1) + block_columns) +
            (ld_block_column * (block_columns - 1) + block_rows),
        !true};

    [[omp::directive(for, collapse(2))]]
    for (std::uint64_t i = 0; i < src_rows; i += block_rows) {
      for (std::uint64_t j = 0; j < src_cols; j += block_columns) {
        std::uint64_t *const block_src{&block[0]},
            *const block_dst{
                &block[ld_block_row * (block_rows - 1) + block_columns]};

        for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
          copy_src_type::copy(&block_src[ld_block_row * ii],
                              &src[ld_src * (i + ii) + j]);
        }

        for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

            base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7,
                                     &block_src[0], ld_block_row, 0, 0, ii, jj);

            base_type::store_subblock(&block_dst[0], ld_block_column, 0, 0, ii,
                                      jj, x0, x1, x2, x3, x4, x5, x6, x7);
          }
        }

        for (std::uint64_t jj = 0; jj < block_columns; ++jj) {
          copy_dst_type::copy(&dst[ld_dst * (j + jj) + i],
                              &block_dst[ld_block_column * jj]);
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

    thread_local PageMemory<std::uint64_t> block{
        (ld_block_row * (block_rows - 1) + block_columns) +
            (ld_block_column * (block_columns - 1) + block_rows),
        !true};

    [[omp::directive(for, collapse(2))]]
    for (std::uint64_t i = 0; i < dim; i += block_rows) {
      for (std::uint64_t j = 0; j <= i; j += block_columns) {
        std::uint64_t *const block_src{&block[0]},
            *const block_dst{
                &block[ld_block_row * (block_rows - 1) + block_columns]};

        if (j < i) {
          for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
            copy_src_type::copy(&block_src[ld_block_row * ii],
                                &dst[dim * (i + ii) + j]);
          }
          for (std::uint64_t jj = 0; jj < block_columns; ++jj) {
            copy_dst_type::copy(&block_dst[ld_block_column * jj],
                                &dst[dim * (j + jj) + i]);
          }

          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
              svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7,
                                       &block_src[0], ld_block_row, 0, 0, ii,
                                       jj);
              base_type::load_subblock(y0, y1, y2, y3, y4, y5, y6, y7,
                                       &block_dst[0], ld_block_column, 0, 0, jj,
                                       ii);

              base_type::store_subblock(&block_dst[0], ld_block_column, 0, 0,
                                        ii, jj, x0, x1, x2, x3, x4, x5, x6, x7);
              base_type::store_subblock(&block_src[0], ld_block_row, 0, 0, jj,
                                        ii, y0, y1, y2, y3, y4, y5, y6, y7);
            }
          }

          for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
            copy_src_type::copy(&dst[dim * (i + ii) + j],
                                &block_src[ld_block_row * ii]);
          }
          for (std::uint64_t jj = 0; jj < block_columns; ++jj) {
            copy_dst_type::copy(&dst[dim * (j + jj) + i],
                                &block_dst[ld_block_column * jj]);
          }
        } else {
          /* The remaining i = j case. */

          for (std::uint64_t ii = 0; ii < block_rows; ++ii) {
            copy_src_type::copy(&block_src[ld_block_row * ii],
                                &dst[dim * (i + ii) + j]);
          }

          for (std::uint64_t ii = 0; ii < block_rows; ii += 8) {
            for (std::uint64_t jj = 0; jj < block_columns; jj += 8) {
              svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;

              base_type::load_subblock(x0, x1, x2, x3, x4, x5, x6, x7,
                                       &block_src[0], ld_block_row, 0, 0, ii,
                                       jj);

              base_type::store_subblock(&block_dst[0], ld_block_column, 0, 0,
                                        ii, jj, x0, x1, x2, x3, x4, x5, x6, x7);
            }
          }

          for (std::uint64_t jj = 0; jj < block_columns; ++jj) {
            copy_dst_type::copy(&dst[dim * (j + jj) + i],
                                &block_dst[ld_block_column * jj]);
          }
        }
      }
    }
  }
};

} // namespace sventt

#endif /* SVENTT_TRANSPOSITION_SVE_IN_REGISTER_FULL_BLOCKING_ROW_FIRST_HPP_INCLUDED \
        */

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_LAYER_SVE_BLOCKED_GENERIC_HPP_INCLUDED
#define SVENTT_LAYER_SVE_BLOCKED_GENERIC_HPP_INCLUDED

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <arm_sve.h>

#include <omp.h>

#include "sventt/common/sve.hpp"
#include "sventt/layer/sve/generic.hpp"
#include "sventt/vector.hpp"

namespace sventt {

template <class modmul_type_, std::uint64_t m, class inner_kernel_type_,
          std::uint64_t block_padding_elements,
          std::uint64_t twiddle_unroll_count, std::uint64_t block_rows,
          class transposition_type>
class BlockedGenericSVELayer {

public:
  using modmul_type = modmul_type_;
  using modulus_type = modmul_type::modulus_type;
  using inner_kernel_type = inner_kernel_type_;

private:
  using generic_layer_type =
      GenericSVELayer<modmul_type, m, inner_kernel_type, block_padding_elements,
                      twiddle_unroll_count, void>;

public:
  static constexpr std::uint64_t get_m(void) { return m; }

  static constexpr std::uint64_t get_radix(void) {
    return inner_kernel_type::get_m();
  }

  class buffer_type {

    std::vector<PageMemory<std::uint64_t>> buffers_orig;
    std::vector<std::uint64_t *> buffers;

  public:
    buffer_type(void) {
      constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
      const std::uint64_t length{(inner_m + block_padding_elements) *
                                     block_rows -
                                 block_padding_elements};
      /*
       * Align the length to 4096 bytes to avoid false Store Fetch Interlocks
       * (SFIs) on A64FX.
       */
      const std::uint64_t length_aligned{
          (length % 512 == 0) ? length : length + 512 - length % 512};

      /* TODO: Take thread affinity as an argument. */
      const std::uint64_t num_domains{4};
      const std::uint64_t num_threads_per_domain{12};
      buffers_orig.resize(num_domains);
      buffers.resize(num_threads_per_domain * num_domains);
      [[omp::directive(parallel, num_threads(num_domains))]] {
        const int thread_num{omp_get_thread_num()};
        const bool allocate_huge_pages{length_aligned *
                                           num_threads_per_domain >=
                                       (std::uint64_t{1} << 18)};

        buffers_orig[thread_num].reset(length_aligned * num_threads_per_domain,
                                       allocate_huge_pages);

        for (std::uint64_t i{}; i < num_threads_per_domain; ++i) {
          buffers[num_threads_per_domain * thread_num + i] =
              &buffers_orig[thread_num][length_aligned * i];
        }
      }
    }

    std::uint64_t *get(const std::uint64_t thread_num) {
      return buffers[thread_num];
    }

    void touch_pages_cyclically(void) {
      const int domain_num_threads{12};

      [[omp::directive(parallel)]] {
        const int domain_num{omp_get_thread_num() / domain_num_threads};
        const int domain_thread_num{omp_get_thread_num() % domain_num_threads};
        const std::uint64_t page_length{
            (buffers_orig[domain_num].size() >= (std::uint64_t{1} << 18))
                ? (std::uint64_t{1} << 18)
                : (std::uint64_t{1} << 13)};
        for (std::uint64_t i = page_length * domain_thread_num;
             i < buffers_orig[domain_num].size();
             i += page_length * domain_num_threads) {
          buffers_orig[domain_num][i] = {};
        }
      }
    }
  };

  template <class vector_type> static void prepare_forward(vector_type &aux) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};

    if (block_rows % cntd != 0) {
      throw std::invalid_argument{"Block size needs to be a multiple of cntd"};
    }
    if (m / inner_m % block_rows != 0) {
      throw std::invalid_argument{"SVE vector length is too large"};
    }

    generic_layer_type::prepare_forward(aux);
  }

  static void compute_forward_without_twiddle(std::uint64_t *const dst,
                                              const std::uint64_t *const src,
                                              const std::byte *&aux_arg,
                                              buffer_type &buffer_arg) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};

    const std::byte *aux;
    {
      const std::uint64_t aux_size{
          pointer_utility::get_and_advance<std::uint64_t>(aux_arg)};
      aux = aux_arg;
      /* Note: This includes the size of twiddles used in twiddle_rows. */
      aux_arg += aux_size;
    }

    std::uint64_t *const buffer{buffer_arg.get(omp_get_thread_num())};

    const std::byte *const aux_orig{aux};
    [[omp::directive(for)]]
    for (std::uint64_t i = 0; i < m / inner_m; i += block_rows) {
      transposition_type::transpose(buffer, &src[i], inner_m, block_rows,
                                    inner_m + block_padding_elements,
                                    m / inner_m);

      for (std::uint64_t ii{}; ii < block_rows; ++ii) {
        aux = aux_orig;
        inner_kernel_type::compute_forward(
            &buffer[(inner_m + block_padding_elements) * ii], aux);
      }

      transposition_type::transpose(&dst[i], buffer, block_rows, inner_m,
                                    m / inner_m,
                                    inner_m + block_padding_elements);
    }
  }

  static void compute_forward_without_twiddle(std::uint64_t *const dst,
                                              const std::byte *&aux,
                                              buffer_type &buffer) {
    compute_forward_without_twiddle(dst, dst, aux, buffer);
  }

  static void twiddle_rows_forward(std::uint64_t *const dst,
                                   const std::uint64_t *const src,
                                   const std::byte *const aux_arg,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride_dst,
                                   const std::uint64_t stride_src) {
    generic_layer_type::twiddle_rows_forward(dst, src, aux_arg, j_start, j_end,
                                             stride_dst, stride_src);
  }

  static void twiddle_rows_forward(std::uint64_t *const dst,
                                   const std::byte *const aux,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride) {
    generic_layer_type::twiddle_rows_forward(dst, aux, j_start, j_end, stride);
  }

  static void compute_forward(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    buffer_type buffer;
    compute_forward_without_twiddle(dst, src, aux, buffer);
    twiddle_rows_forward(dst, aux, 0, inner_m, m / inner_m);
  }

  static void compute_forward(std::uint64_t *const dst, const std::byte *&aux) {
    compute_forward(dst, dst, aux);
  }

  template <class vector_type> static void prepare_inverse(vector_type &aux) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};

    if (block_rows % cntd != 0) {
      throw std::invalid_argument{"Block size needs to be a multiple of cntd"};
    }
    if (m / inner_m % block_rows != 0) {
      throw std::invalid_argument{"SVE vector length is too large"};
    }

    generic_layer_type::prepare_inverse(aux);
  }

  static void twiddle_rows_inverse(std::uint64_t *const dst,
                                   const std::uint64_t *const src,
                                   const std::byte *const aux_arg,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride_dst,
                                   const std::uint64_t stride_src) {
    generic_layer_type::twiddle_rows_inverse(dst, src, aux_arg, j_start, j_end,
                                             stride_dst, stride_src);
  }

  static void twiddle_rows_inverse(std::uint64_t *const dst,
                                   const std::byte *const aux,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride) {
    generic_layer_type::twiddle_rows_inverse(dst, aux, j_start, j_end, stride);
  }

  static void compute_inverse_without_twiddle(std::uint64_t *const dst,
                                              const std::uint64_t *const src,
                                              const std::byte *&aux_arg,
                                              buffer_type &buffer_arg) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};

    aux_arg += 8 * cntd * inner_m;
    const std::byte *aux;
    {
      const std::uint64_t aux_size{
          pointer_utility::get_and_advance<std::uint64_t>(aux_arg)};
      aux = aux_arg;
      aux_arg += aux_size;
    }

    std::uint64_t *const buffer{buffer_arg.get(omp_get_thread_num())};

    const std::byte *const aux_orig{aux};
    [[omp::directive(for)]]
    for (std::uint64_t i = 0; i < m / inner_m; i += block_rows) {
      transposition_type::transpose(buffer, &src[i], inner_m, block_rows,
                                    inner_m + block_padding_elements,
                                    m / inner_m);

      for (std::uint64_t ii{}; ii < block_rows; ++ii) {
        aux = aux_orig;
        inner_kernel_type::compute_inverse(
            &buffer[(inner_m + block_padding_elements) * ii], aux);
      }

      transposition_type::transpose(&dst[i], buffer, block_rows, inner_m,
                                    m / inner_m,
                                    inner_m + block_padding_elements);
    }
  }

  static void compute_inverse_without_twiddle(std::uint64_t *const dst,
                                              const std::uint64_t *const src,
                                              const std::byte *&aux_arg) {
    buffer_type buffer;
    compute_inverse_without_twiddle(dst, src, aux_arg, buffer);
  }

  static void compute_inverse_without_twiddle(std::uint64_t *const dst,
                                              const std::byte *&aux,
                                              buffer_type &buffer) {
    compute_inverse_without_twiddle(dst, dst, aux, buffer);
  }

  static void compute_inverse_without_twiddle(std::uint64_t *const dst,
                                              const std::byte *&aux) {
    compute_inverse_without_twiddle(dst, dst, aux);
  }

  static void compute_inverse(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    twiddle_rows_inverse(dst, src, aux, 0, inner_m, m / inner_m, m / inner_m);
    compute_inverse_without_twiddle(dst, aux);
  }

  static void compute_inverse(std::uint64_t *const dst, const std::byte *&aux) {
    compute_inverse(dst, dst, aux);
  }
};

} // namespace sventt

#endif /* SVENTT_LAYER_SVE_BLOCKED_GENERIC_HPP_INCLUDED */

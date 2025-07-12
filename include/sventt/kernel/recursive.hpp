// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_KERNEL_RECURSIVE_HPP_INCLUDED
#define SVENTT_KERNEL_RECURSIVE_HPP_INCLUDED

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sventt {

template <class modulus_type_, std::uint64_t m, class layer_type_,
          class inner_kernel_type_, bool separate_twiddle>
class RecursiveNTT {

public:
  using modulus_type = modulus_type_;
  using layer_type = layer_type_;
  using inner_kernel_type = inner_kernel_type_;

private:
  static_assert(
      std::is_same_v<modulus_type, typename layer_type::modulus_type>);
  static_assert(
      std::is_same_v<modulus_type, typename inner_kernel_type::modulus_type>);
  static_assert(layer_type::get_m() == m);
  static_assert(inner_kernel_type::get_m() * layer_type::get_radix() == m);

public:
  static constexpr std::uint64_t get_m(void) { return m; }

  template <class vector_type> static void prepare_forward(vector_type &aux) {
    /* aux size, to be corrected later */
    aux.template push_back<std::uint64_t>({});
    const typename vector_type::size_type aux_pos_after_size{aux.size()};

    layer_type::prepare_forward(aux);
    inner_kernel_type::prepare_forward(aux);

    aux.template reinterpret_at<std::uint64_t>(aux_pos_after_size -
                                               sizeof(std::uint64_t)) =
        aux.size() - aux_pos_after_size;
  }

  static void compute_forward(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux_arg) {
    constexpr std::uint64_t radix{layer_type::get_radix()};

    const std::byte *aux;
    {
      const std::uint64_t aux_size{
          pointer_utility::get_and_advance<std::uint64_t>(aux_arg)};
      aux = aux_arg;
      aux_arg += aux_size;
    }

    if constexpr (separate_twiddle) {
      typename layer_type::buffer_type buffer;
      buffer.touch_pages_cyclically();

      [[omp::directive(parallel, firstprivate(aux))]] {
        layer_type::compute_forward_without_twiddle(dst, src, aux, buffer);

        const std::byte *const aux_orig{aux};
        [[omp::directive(for)]]
        for (std::uint64_t i = 0; i < radix; ++i) {
          aux = aux_orig;
          layer_type::twiddle_rows_forward(dst, aux, i, i + 1, m / radix);
          inner_kernel_type::compute_forward(&dst[m / radix * i], aux);
        }
      }
    } else {
      layer_type::compute_forward(dst, src, aux);
      const std::byte *const aux_orig{aux};
      for (std::uint64_t i{}; i < radix; ++i) {
        aux = aux_orig;
        inner_kernel_type::compute_forward(&dst[m / radix * i], aux);
      }
    }
  }

  static void compute_forward(std::uint64_t *const dst, const std::byte *&aux) {
    compute_forward(dst, dst, aux);
  }

  static void
  descramble_forward([[maybe_unused]] std::uint64_t *const dst,
                     [[maybe_unused]] const std::uint64_t *const src) {}

  template <class vector_type> static void prepare_inverse(vector_type &aux) {
    /* aux size, to be corrected later */
    aux.template push_back<std::uint64_t>({});
    const typename vector_type::size_type aux_pos_after_size{aux.size()};

    inner_kernel_type::prepare_inverse(aux);
    layer_type::prepare_inverse(aux);

    aux.template reinterpret_at<std::uint64_t>(aux_pos_after_size -
                                               sizeof(std::uint64_t)) =
        aux.size() - aux_pos_after_size;
  }

  static void compute_inverse(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux_arg) {
    constexpr std::uint64_t radix{layer_type::get_radix()};

    const std::byte *aux;
    {
      const std::uint64_t aux_size{
          pointer_utility::get_and_advance<std::uint64_t>(aux_arg)};
      aux = aux_arg;
      aux_arg += aux_size;
    }

    if constexpr (separate_twiddle) {
      typename layer_type::buffer_type buffer;

      [[omp::directive(parallel, firstprivate(aux))]] {
        const std::byte *const aux_orig{aux};
        [[omp::directive(for)]]
        for (std::uint64_t i = 0; i < radix; ++i) {
          aux = aux_orig;
          inner_kernel_type::compute_inverse(&dst[m / radix * i],
                                             &src[m / radix * i], aux);
          layer_type::twiddle_rows_inverse(dst, aux, i, i + 1, m / radix);
        }

        layer_type::compute_inverse_without_twiddle(dst, aux, buffer);
      }
    } else {
      const std::byte *const aux_orig{aux};
      for (std::uint64_t i{}; i < radix; ++i) {
        aux = aux_orig;
        inner_kernel_type::compute_inverse(&dst[m / radix * i],
                                           &src[m / radix * i], aux);
      }
      layer_type::compute_inverse(dst, aux);
    }
  }

  static void compute_inverse(std::uint64_t *const dst, const std::byte *&aux) {
    compute_inverse(dst, dst, aux);
  }

  static void
  descramble_inverse([[maybe_unused]] std::uint64_t *const dst,
                     [[maybe_unused]] const std::uint64_t *const src) {}
};

} // namespace sventt

#endif /* SVENTT_KERNEL_RECURSIVE_HPP_INCLUDED */

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_KERNEL_ITERATIVE_HPP_INCLUDED
#define SVENTT_KERNEL_ITERATIVE_HPP_INCLUDED

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "sventt/vector.hpp"

namespace sventt {

template <class modulus_type_, std::uint64_t m, class... layer_types>
class IterativeNTT {

public:
  using modulus_type = modulus_type_;

private:
  static_assert((
      std::is_same_v<modulus_type, typename layer_types::modulus_type> && ...));
  static_assert(((layer_types::get_m() == m) && ...));
  static_assert((layer_types::get_radix() * ...) == m);

  template <class... layer_types_, class vector_type>
  static void prepare_forward_helper(vector_type &aux) {
    (layer_types_::prepare_forward(aux), ...);
  }

  template <class layer_type, class... layer_types_, class vector_type>
  static void prepare_inverse_helper(vector_type &aux) {
    if constexpr (sizeof...(layer_types_) >= 1) {
      prepare_inverse_helper<layer_types_...>(aux);
    }
    layer_type::prepare_inverse(aux);
  }

  template <class layer_type, class... layer_types_>
  static void compute_forward_helper(std::uint64_t *const dst,
                                     const std::uint64_t *const src,
                                     const std::byte *&aux) {
    layer_type::compute_forward(dst, src, aux);
    compute_forward_helper<layer_types_...>(dst, aux);
  }

  template <class... layer_types_>
  static void compute_forward_helper(std::uint64_t *const dst,
                                     const std::byte *&aux) {
    (layer_types_::compute_forward(dst, aux), ...);
  }

  template <class layer_type, class... layer_types_>
  static void compute_inverse_helper(std::uint64_t *const dst,
                                     const std::uint64_t *const src,
                                     const std::byte *&aux) {
    if constexpr (sizeof...(layer_types_) >= 1) {
      compute_inverse_helper<layer_types_...>(dst, src, aux);
      layer_type::compute_inverse(dst, aux);
    } else {
      layer_type::compute_inverse(dst, src, aux);
    }
  }

  template <class layer_type, class... layer_types_>
  static void compute_inverse_helper(std::uint64_t *const dst,
                                     const std::byte *&aux) {
    if constexpr (sizeof...(layer_types_) >= 1) {
      compute_inverse_helper<layer_types_...>(dst, aux);
    }
    layer_type::compute_inverse(dst, aux);
  }

public:
  static constexpr std::uint64_t get_m(void) { return m; }

  template <class vector_type> static void prepare_forward(vector_type &aux) {
    prepare_forward_helper<layer_types...>(aux);
  }

  static void compute_forward(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux) {
    compute_forward_helper<layer_types...>(dst, src, aux);
  }

  static void compute_forward(std::uint64_t *const dst, const std::byte *&aux) {
    compute_forward_helper<layer_types...>(dst, aux);
  }

  static void
  descramble_forward([[maybe_unused]] std::uint64_t *const dst,
                     [[maybe_unused]] const std::uint64_t *const src) {}

  template <class vector_type> static void prepare_inverse(vector_type &aux) {
    prepare_inverse_helper<layer_types...>(aux);
  }

  static void compute_inverse(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux) {
    compute_inverse_helper<layer_types...>(dst, src, aux);
  }

  static void compute_inverse(std::uint64_t *const dst, const std::byte *&aux) {
    compute_inverse_helper<layer_types...>(dst, aux);
  }

  static void
  descramble_inverse([[maybe_unused]] std::uint64_t *const dst,
                     [[maybe_unused]] const std::uint64_t *const src) {}
};

} // namespace sventt

#endif /* SVENTT_KERNEL_ITERATIVE_HPP_INCLUDED */

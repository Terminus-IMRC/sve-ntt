// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_LAYER_SCALAR_GENERIC_HPP_INCLUDED
#define SVENTT_LAYER_SCALAR_GENERIC_HPP_INCLUDED

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace sventt {

template <class modmul_type_, std::uint64_t m, class inner_kernel_type_>
class GenericScalarLayer {

public:
  using modmul_type = modmul_type_;
  using modulus_type = modmul_type::modulus_type;
  using inner_kernel_type = inner_kernel_type_;

  static_assert(m % inner_kernel_type::get_m() == 0);

  static constexpr std::uint64_t get_m(void) { return m; }

  static constexpr std::uint64_t get_radix(void) {
    return inner_kernel_type::get_m();
  }

  static void prepare_forward(AuxiliaryVector &aux) {
    inner_kernel_type::prepare_forward(aux);

    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    constexpr std::uint64_t omega_m{modulus_type::get_root_forward(m)};
    for (std::uint64_t j{}; j < inner_m; ++j) {
      aux.push_back(modmul_type::to_montgomery(modulus_type::power(
          omega_m, bitreverse(j) >> (65 - std::bit_width(inner_m)))));
    }
  }

  static void compute_forward_without_twiddle(std::uint64_t *const dst,
                                              const std::uint64_t *const src,
                                              const std::byte *&aux) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};

    const std::byte *const aux_orig{aux};
    for (std::uint64_t i{}; i < m / inner_m; ++i) {
      alignas(64) std::array<std::uint64_t, inner_m> arr;
      for (std::uint64_t j{}; j < inner_m; ++j) {
        arr[j] = src[m / inner_m * j + i];
      }
      aux = aux_orig;
      inner_kernel_type::compute_forward(arr.data(), aux);
      for (std::uint64_t j{}; j < inner_m; ++j) {
        dst[m / inner_m * j + i] = arr[j];
      }
    }

    /* Increment in advance. */
    aux += 8 * inner_m;
  }

  static void compute_forward_without_twiddle(std::uint64_t *const dst,
                                              const std::byte *&aux) {
    compute_forward_without_twiddle(dst, dst, aux);
  }

  static void twiddle_rows_forward(std::uint64_t *const dst,
                                   const std::uint64_t *const src,
                                   const std::byte *const aux,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride_dst,
                                   const std::uint64_t stride_src) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    constexpr std::uint64_t unity{modmul_type::to_montgomery(1)};
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};

    for (std::uint64_t j{j_start}; j < j_end; ++j) {
      const std::uint64_t omega_j{
          pointer_utility::get<std::uint64_t>(aux - 8 * inner_m + 8 * j)};
      const std::uint64_t omega_j_precomp{modmul_type::precompute(omega_j)};
      std::uint64_t omega_ij{unity};
      for (std::uint64_t i{}; i < m / inner_m; ++i) {
        dst[stride_dst * j + i] =
            modmul_type::multiply(src[stride_src * j + i], omega_ij);
        omega_ij = modmul_type::multiply(omega_ij, omega_j, omega_j_precomp);
        omega_ij = std::min(omega_ij, omega_ij - N);
      }
    }
  }

  static void twiddle_rows_forward(std::uint64_t *const dst,
                                   const std::byte *const aux,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride) {
    twiddle_rows_forward(dst, dst, aux, j_start, j_end, stride, stride);
  }

  static void compute_forward(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    compute_forward_without_twiddle(dst, src, aux);
    twiddle_rows_forward(dst, aux, 0, inner_m, m / inner_m);
  }

  static void compute_forward(std::uint64_t *const dst, const std::byte *&aux) {
    compute_forward(dst, dst, aux);
  }

  static void prepare_inverse(AuxiliaryVector &aux) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    constexpr std::uint64_t omega_m{modulus_type::get_root_inverse(m)};
    for (std::uint64_t j{}; j < inner_m; ++j) {
      aux.push_back(modmul_type::to_montgomery(modulus_type::power(
          omega_m, bitreverse(j) >> (65 - std::bit_width(inner_m)))));
    }

    inner_kernel_type::prepare_inverse(aux);
  }

  static void twiddle_rows_inverse(std::uint64_t *const dst,
                                   const std::uint64_t *const src,
                                   const std::byte *const aux,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride_dst,
                                   const std::uint64_t stride_src) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    constexpr std::uint64_t unity{modmul_type::to_montgomery(1)};
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};

    for (std::uint64_t j{j_start}; j < j_end; ++j) {
      const std::uint64_t omega_j{
          pointer_utility::get<std::uint64_t>(aux + 8 * j)};
      const std::uint64_t omega_j_precomp{modmul_type::precompute(omega_j)};
      std::uint64_t omega_ij{unity};
      for (std::uint64_t i{}; i < m / inner_m; ++i) {
        dst[stride_dst * j + i] =
            modmul_type::multiply(src[stride_src * j + i], omega_ij);
        omega_ij = modmul_type::multiply(omega_ij, omega_j, omega_j_precomp);
        omega_ij = std::min(omega_ij, omega_ij - N);
      }
    }
  }

  static void twiddle_rows_inverse(std::uint64_t *const dst,
                                   const std::byte *const aux,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride) {
    twiddle_rows_inverse(dst, dst, aux, j_start, j_end, stride, stride);
  }

  static void compute_inverse_without_twiddle(std::uint64_t *const dst,
                                              const std::uint64_t *const src,
                                              const std::byte *&aux) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};

    aux += 8 * inner_m;
    const std::byte *const aux_orig{aux};
    for (std::uint64_t i{}; i < m / inner_m; ++i) {
      alignas(64) std::array<std::uint64_t, inner_m> arr;
      for (std::uint64_t j{}; j < inner_m; ++j) {
        arr[j] = src[m / inner_m * j + i];
      }
      aux = aux_orig;
      inner_kernel_type::compute_inverse(arr.data(), aux);
      for (std::uint64_t j{}; j < inner_m; ++j) {
        dst[m / inner_m * j + i] = arr[j];
      }
    }
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

#endif /* SVENTT_LAYER_SCALAR_GENERIC_HPP_INCLUDED */

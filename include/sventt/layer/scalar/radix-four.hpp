// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_LAYER_SCALAR_RADIX_FOUR_HPP_INCLUDED
#define SVENTT_LAYER_SCALAR_RADIX_FOUR_HPP_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sventt {

template <class modmul_type_, std::uint64_t m, std::uint64_t n,
          std::uint64_t inverse_factor = 1>
class RadixFourScalarLayer {

public:
  using modmul_type = modmul_type_;
  using modulus_type = modmul_type::modulus_type;

  static constexpr std::uint64_t get_radix(void) { return 4; }

  static constexpr std::uint64_t get_m(void) { return m; }

  static constexpr std::uint64_t get_n(void) { return n; }

  static_assert(n >= get_radix());

  static void prepare_forward(AuxiliaryVector &aux) {
    const std::uint64_t omega_n{modulus_type::get_root_forward(n)};
    aux.push_back(modmul_type::to_montgomery(omega_n));
  }

  static void compute_forward(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    constexpr std::uint64_t unity{modmul_type::to_montgomery(1)};
    constexpr std::uint64_t omega_n4{
        modmul_type::to_montgomery(modulus_type::get_root_forward(4))};
    const std::uint64_t omega_n{
        pointer_utility::get_and_advance<std::uint64_t>(aux)};
    const std::uint64_t omega_n_precomp{modmul_type::precompute(omega_n)};
    std::uint64_t omega_i{unity}, omega_i_n4{omega_n4}, omega_2i{unity};
    for (std::uint64_t i{}; i < n / 4; ++i) {
      const std::uint64_t omega_i_precomp{modmul_type::precompute(omega_i)},
          omega_i_n4_precomp{modmul_type::precompute(omega_i_n4)},
          omega_2i_precomp{modmul_type::precompute(omega_2i)};
      for (std::uint64_t j{}; j < m; j += n) {
        std::uint64_t x0, x1, x2, x3, y0, y1, y2, y3;
        x0 = src[i + j + n / 4 * 0];
        x1 = src[i + j + n / 4 * 1];
        x2 = src[i + j + n / 4 * 2];
        x3 = src[i + j + n / 4 * 3];
        y0 = x0 + x2;
        y1 = x1 + x3;
        y2 = modmul_type::multiply(x0 - x2 + N * 2, omega_i, omega_i_precomp);
        y3 = modmul_type::multiply(x1 - x3 + N * 2, omega_i_n4,
                                   omega_i_n4_precomp);
        y0 = std::min(y0, y0 - N * 2);
        y1 = std::min(y1, y1 - N * 2);
        x0 = y0 + y1;
        x1 = modmul_type::multiply(y0 - y1 + N * 2, omega_2i, omega_2i_precomp);
        x2 = y2 + y3;
        x3 = modmul_type::multiply(y2 - y3 + N * 2, omega_2i, omega_2i_precomp);
        x0 = std::min(x0, x0 - N * 2);
        x2 = std::min(x2, x2 - N * 2);
        dst[i + j + n / 4 * 0] = x0;
        dst[i + j + n / 4 * 1] = x1;
        dst[i + j + n / 4 * 2] = x2;
        dst[i + j + n / 4 * 3] = x3;
      }
      omega_i = modmul_type::multiply(omega_i, omega_n, omega_n_precomp);
      omega_i_n4 = modmul_type::multiply(omega_i_n4, omega_n, omega_n_precomp);
      omega_i = std::min(omega_i, omega_i - N);
      omega_i_n4 = std::min(omega_i_n4, omega_i_n4 - N);
      /* TODO: Update these incrementally. */
      omega_2i = modmul_type::multiply(omega_i, omega_i);
      omega_2i = std::min(omega_2i, omega_2i - N);
    }
  }

  static void compute_forward(std::uint64_t *const dst, const std::byte *&aux) {
    compute_forward(dst, dst, aux);
  }

  static void prepare_inverse(AuxiliaryVector &aux) {
    if constexpr (n == get_radix()) {
      if constexpr (inverse_factor != 1) {
        const std::uint64_t m_inv{modulus_type::invert(inverse_factor)};
        const std::uint64_t omega_n4{modulus_type::get_root_inverse(4)};
        const std::uint64_t omega_n4_m_inv{
            modulus_type::multiply(omega_n4, m_inv)};
        aux.push_back(modmul_type::to_montgomery(m_inv));
        aux.push_back(modmul_type::to_montgomery(omega_n4_m_inv));
      }
    } else {
      static_assert(inverse_factor == 1);
      const std::uint64_t omega_n{modulus_type::get_root_inverse(n)};
      aux.push_back(modmul_type::to_montgomery(omega_n));
    }
  }

  static void compute_inverse(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    constexpr std::uint64_t unity{modmul_type::to_montgomery(1)};
    constexpr std::uint64_t omega_n4{
        modmul_type::to_montgomery(modulus_type::get_root_inverse(4))};
    if constexpr (n == get_radix()) {
      if constexpr (inverse_factor == 1) {
        const std::uint64_t omega_n4_precomp{modmul_type::precompute(omega_n4)};
        for (std::uint64_t j{}; j < m; j += 4) {
          std::uint64_t x0, x1, x2, x3, y0, y1, y2, y3;
          x0 = src[j + 0];
          x1 = src[j + 1];
          x2 = src[j + 2];
          x3 = src[j + 3];
          y0 = x0 + x1;
          y1 = x0 - x1;
          y2 = x2 + x3;
          y3 = modmul_type::multiply(x2 - x3 + N * 2, omega_n4,
                                     omega_n4_precomp);
          y0 = std::min(y0, y0 - N * 2);
          y1 = std::min(y1, y1 + N * 2);
          y2 = std::min(y2, y2 - N * 2);
          x0 = y0 + y2;
          x1 = y1 + y3;
          x2 = y0 - y2;
          x3 = y1 - y3;
          x0 = std::min(x0, x0 - N * 2);
          x1 = std::min(x1, x1 - N * 2);
          x2 = std::min(x2, x2 + N * 2);
          x3 = std::min(x3, x3 + N * 2);
          dst[j + 0] = x0;
          dst[j + 1] = x1;
          dst[j + 2] = x2;
          dst[j + 3] = x3;
        }
      } else {
        const std::uint64_t m_inv{
            pointer_utility::get_and_advance<std::uint64_t>(aux)};
        const std::uint64_t omega_n4_m_inv{
            pointer_utility::get_and_advance<std::uint64_t>(aux)};
        const std::uint64_t m_inv_precomp{modmul_type::precompute(m_inv)};
        const std::uint64_t omega_n4_m_inv_precomp{
            modmul_type::precompute(omega_n4_m_inv)};
        for (std::uint64_t j{}; j < m; j += 4) {
          std::uint64_t x0, x1, x2, x3, y0, y1, y2, y3;
          x0 = src[j + 0];
          x1 = src[j + 1];
          x2 = src[j + 2];
          x3 = src[j + 3];
          y0 = modmul_type::multiply(x0 + x1, m_inv, m_inv_precomp);
          y1 = modmul_type::multiply(x0 - x1 + N * 2, m_inv, m_inv_precomp);
          y2 = modmul_type::multiply(x2 + x3, m_inv, m_inv_precomp);
          y3 = modmul_type::multiply(x2 - x3 + N * 2, omega_n4_m_inv,
                                     omega_n4_m_inv_precomp);
          x0 = y0 + y2;
          x1 = y1 + y3;
          x2 = y0 - y2;
          x3 = y1 - y3;
          x0 = std::min(x0, x0 - N * 2);
          x1 = std::min(x1, x1 - N * 2);
          x2 = std::min(x2, x2 + N * 2);
          x3 = std::min(x3, x3 + N * 2);
          dst[j + 0] = x0;
          dst[j + 1] = x1;
          dst[j + 2] = x2;
          dst[j + 3] = x3;
        }
      }
    } else {
      const std::uint64_t omega_n{
          pointer_utility::get_and_advance<std::uint64_t>(aux)};
      const std::uint64_t omega_n_precomp{modmul_type::precompute(omega_n)};
      std::uint64_t omega_i{unity}, omega_i_n4{omega_n4}, omega_2i{unity};
      for (std::uint64_t i{}; i < n / 4; ++i) {
        const std::uint64_t omega_i_precomp{modmul_type::precompute(omega_i)},
            omega_i_n4_precomp{modmul_type::precompute(omega_i_n4)},
            omega_2i_precomp{modmul_type::precompute(omega_2i)};
        for (std::uint64_t j{}; j < m; j += n) {
          std::uint64_t x0, x1, x2, x3, y0, y1, y2, y3;
          x0 = src[i + j + n / 4 * 0];
          x1 = modmul_type::multiply(src[i + j + n / 4 * 1], omega_2i,
                                     omega_2i_precomp);
          x2 = src[i + j + n / 4 * 2];
          x3 = modmul_type::multiply(src[i + j + n / 4 * 3], omega_2i,
                                     omega_2i_precomp);
          y0 = x0 + x1;
          y1 = x0 - x1;
          y2 = modmul_type::multiply(x2 + x3, omega_i, omega_i_precomp);
          y3 = modmul_type::multiply(x2 - x3 + N * 2, omega_i_n4,
                                     omega_i_n4_precomp);
          y0 = std::min(y0, y0 - N * 2);
          y1 = std::min(y1, y1 + N * 2);
          x0 = y0 + y2;
          x1 = y1 + y3;
          x2 = y0 - y2;
          x3 = y1 - y3;
          x0 = std::min(x0, x0 - N * 2);
          x1 = std::min(x1, x1 - N * 2);
          x2 = std::min(x2, x2 + N * 2);
          x3 = std::min(x3, x3 + N * 2);
          dst[i + j + n / 4 * 0] = x0;
          dst[i + j + n / 4 * 1] = x1;
          dst[i + j + n / 4 * 2] = x2;
          dst[i + j + n / 4 * 3] = x3;
        }
        omega_i = modmul_type::multiply(omega_i, omega_n, omega_n_precomp);
        omega_i_n4 =
            modmul_type::multiply(omega_i_n4, omega_n, omega_n_precomp);
        omega_i = std::min(omega_i, omega_i - N);
        omega_i_n4 = std::min(omega_i_n4, omega_i_n4 - N);
        /* TODO: Update these incrementally. */
        omega_2i = modmul_type::multiply(omega_i, omega_i);
        omega_2i = std::min(omega_2i, omega_2i - N);
      }
    }
  }

  static void compute_inverse(std::uint64_t *const dst, const std::byte *&aux) {
    compute_inverse(dst, dst, aux);
  }
};

} // namespace sventt

#endif /* SVENTT_LAYER_SCALAR_RADIX_FOUR_HPP_INCLUDED */

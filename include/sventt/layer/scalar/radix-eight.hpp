// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_LAYER_SCALAR_RADIX_EIGHT_HPP_INCLUDED
#define SVENTT_LAYER_SCALAR_RADIX_EIGHT_HPP_INCLUDED

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace sventt {

template <class modmul_type_, std::uint64_t m, std::uint64_t n,
          std::uint64_t inverse_factor = 1>
class RadixEightScalarLayer {

public:
  using modmul_type = modmul_type_;
  using modulus_type = modmul_type::modulus_type;

  static constexpr std::uint64_t get_radix(void) { return 8; }

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
    constexpr std::uint64_t omega_n81{
        modmul_type::to_montgomery(modulus_type::get_root_forward(8))};
    constexpr std::uint64_t omega_n82{
        modmul_type::to_montgomery(modulus_type::get_root_forward(4))};
    constexpr std::uint64_t omega_n83{modmul_type::from_montgomery(
        modulus_type::multiply(omega_n81, omega_n82))};
    const std::uint64_t omega_n{
        pointer_utility::get_and_advance<std::uint64_t>(aux)};
    const std::uint64_t omega_n_precomp{modmul_type::precompute(omega_n)};
    std::uint64_t omega_i{unity}, omega_i_n81{omega_n81},
        omega_i_n82{omega_n82}, omega_i_n83{omega_n83}, omega_2i{unity},
        omega_2i_n41{omega_n82}, omega_4i{unity};
    for (std::uint64_t i{}; i < n / 8; ++i) {
      const std::uint64_t omega_n82_precomp{modmul_type::precompute(omega_n82)},
          omega_i_precomp{modmul_type::precompute(omega_i)},
          omega_i_n81_precomp{modmul_type::precompute(omega_i_n81)},
          omega_i_n82_precomp{modmul_type::precompute(omega_i_n82)},
          omega_i_n83_precomp{modmul_type::precompute(omega_i_n83)},
          omega_2i_precomp{modmul_type::precompute(omega_2i)},
          omega_2i_n41_precomp{modmul_type::precompute(omega_2i_n41)},
          omega_4i_precomp{modmul_type::precompute(omega_4i)};
      for (std::uint64_t j{}; j < m; j += n) {
        std::uint64_t x0, x1, x2, x3, x4, x5, x6, x7;
        std::uint64_t y0, y1, y2, y3, y4, y5, y6, y7;
        x0 = src[i + j + n / 8 * 0];
        x1 = src[i + j + n / 8 * 1];
        x2 = src[i + j + n / 8 * 2];
        x3 = src[i + j + n / 8 * 3];
        x4 = src[i + j + n / 8 * 4];
        x5 = src[i + j + n / 8 * 5];
        x6 = src[i + j + n / 8 * 6];
        x7 = src[i + j + n / 8 * 7];
        y0 = x0 + x4;
        y1 = x1 + x5;
        y2 = x2 + x6;
        y3 = x3 + x7;
        y4 = modmul_type::multiply(x0 - x4 + N * 2, omega_i, omega_i_precomp);
        y5 = modmul_type::multiply(x1 - x5 + N * 2, omega_i_n81,
                                   omega_i_n81_precomp);
        y6 = modmul_type::multiply(x2 - x6 + N * 2, omega_i_n82,
                                   omega_i_n82_precomp);
        y7 = modmul_type::multiply(x3 - x7 + N * 2, omega_i_n83,
                                   omega_i_n83_precomp);
        y0 = std::min(y0, y0 - N * 2);
        y1 = std::min(y1, y1 - N * 2);
        y2 = std::min(y2, y2 - N * 2);
        y3 = std::min(y3, y3 - N * 2);
        x0 = y0 + y2;
        x1 = y1 + y3;
        x2 = modmul_type::multiply(y0 - y2 + N * 2, omega_2i, omega_2i_precomp);
        x3 = modmul_type::multiply(y1 - y3 + N * 2, omega_2i_n41,
                                   omega_2i_n41_precomp);
        x4 = y4 + y6;
        x5 = y5 + y7;
        x6 = modmul_type::multiply(y4 - y6 + N * 2, omega_2i, omega_2i_precomp);
        x7 = modmul_type::multiply(y5 - y7 + N * 2, omega_2i_n41,
                                   omega_2i_n41_precomp);
        x0 = std::min(x0, x0 - N * 2);
        x1 = std::min(x1, x1 - N * 2);
        x4 = std::min(x4, x4 - N * 2);
        x5 = std::min(x5, x5 - N * 2);
        y0 = x0 + x1;
        y1 = modmul_type::multiply(x0 - x1 + N * 2, omega_4i, omega_4i_precomp);
        y2 = x2 + x3;
        y3 = modmul_type::multiply(x2 - x3 + N * 2, omega_4i, omega_4i_precomp);
        y4 = x4 + x5;
        y5 = modmul_type::multiply(x4 - x5 + N * 2, omega_4i, omega_4i_precomp);
        y6 = x6 + x7;
        y7 = modmul_type::multiply(x6 - x7 + N * 2, omega_4i, omega_4i_precomp);
        y0 = std::min(y0, y0 - N * 2);
        y2 = std::min(y2, y2 - N * 2);
        y4 = std::min(y4, y4 - N * 2);
        y6 = std::min(y6, y6 - N * 2);
        dst[i + j + n / 8 * 0] = y0;
        dst[i + j + n / 8 * 1] = y1;
        dst[i + j + n / 8 * 2] = y2;
        dst[i + j + n / 8 * 3] = y3;
        dst[i + j + n / 8 * 4] = y4;
        dst[i + j + n / 8 * 5] = y5;
        dst[i + j + n / 8 * 6] = y6;
        dst[i + j + n / 8 * 7] = y7;
      }
      omega_i = modmul_type::multiply(omega_i, omega_n, omega_n_precomp);
      omega_i_n81 =
          modmul_type::multiply(omega_i_n81, omega_n, omega_n_precomp);
      omega_i_n82 =
          modmul_type::multiply(omega_i_n82, omega_n, omega_n_precomp);
      omega_i_n83 =
          modmul_type::multiply(omega_i_n83, omega_n, omega_n_precomp);
      omega_i = std::min(omega_i, omega_i - N);
      omega_i_n81 = std::min(omega_i_n81, omega_i_n81 - N);
      omega_i_n82 = std::min(omega_i_n82, omega_i_n82 - N);
      omega_i_n83 = std::min(omega_i_n83, omega_i_n83 - N);
      /* TODO: Update these incrementally. */
      omega_2i = modmul_type::multiply(omega_i, omega_i);
      omega_2i_n41 =
          modmul_type::multiply(omega_2i, omega_n82, omega_n82_precomp);
      omega_2i_n41 = std::min(omega_2i_n41, omega_2i_n41 - N);
      omega_2i = std::min(omega_2i, omega_2i - N);
      omega_4i = modmul_type::multiply(omega_2i, omega_2i);
      omega_4i = std::min(omega_4i, omega_4i - N);
    }
  }

  static void compute_forward(std::uint64_t *const dst, const std::byte *&aux) {
    compute_forward(dst, dst, aux);
  }

  static void prepare_inverse(AuxiliaryVector &aux) {
    if constexpr (n == get_radix()) {
      if constexpr (inverse_factor != 1) {
        constexpr std::uint64_t m_inv{modulus_type::invert(inverse_factor)};
        constexpr std::uint64_t omega_n81{modulus_type::get_root_inverse(8)};
        constexpr std::uint64_t omega_n82{modulus_type::get_root_inverse(4)};
        constexpr std::uint64_t omega_n83{
            modulus_type::multiply(omega_n81, omega_n82)};
        constexpr std::uint64_t omega_n81_m_inv{
            modulus_type::multiply(omega_n81, m_inv)};
        constexpr std::uint64_t omega_n82_m_inv{
            modulus_type::multiply(omega_n82, m_inv)};
        constexpr std::uint64_t omega_n83_m_inv{
            modulus_type::multiply(omega_n83, m_inv)};
        aux.push_back(modmul_type::to_montgomery(m_inv));
        aux.push_back(modmul_type::to_montgomery(omega_n81_m_inv));
        aux.push_back(modmul_type::to_montgomery(omega_n82_m_inv));
        aux.push_back(modmul_type::to_montgomery(omega_n83_m_inv));
      }
    } else {
      static_assert(inverse_factor == 1);
      constexpr std::uint64_t omega_n{modulus_type::get_root_inverse(n)};
      aux.push_back(modmul_type::to_montgomery(omega_n));
    }
  }

  static void compute_inverse(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    constexpr std::uint64_t unity{modmul_type::to_montgomery(1)};
    constexpr std::uint64_t omega_n81{
        modmul_type::to_montgomery(modulus_type::get_root_inverse(8))};
    constexpr std::uint64_t omega_n82{
        modmul_type::to_montgomery(modulus_type::get_root_inverse(4))};
    constexpr std::uint64_t omega_n83{modmul_type::from_montgomery(
        modulus_type::multiply(omega_n81, omega_n82))};
    const std::uint64_t omega_n82_precomp{modmul_type::precompute(omega_n82)};
    if constexpr (n == get_radix()) {
      if constexpr (inverse_factor == 1) {
        const std::uint64_t omega_n81_precomp{
            modmul_type::precompute(omega_n81)},
            omega_n83_precomp{modmul_type::precompute(omega_n83)};
        for (std::uint64_t j{}; j < m; j += 8) {
          std::uint64_t x0, x1, x2, x3, x4, x5, x6, x7;
          std::uint64_t y0, y1, y2, y3, y4, y5, y6, y7;
          x0 = src[j + 0];
          x1 = src[j + 1];
          x2 = src[j + 2];
          x3 = src[j + 3];
          x4 = src[j + 4];
          x5 = src[j + 5];
          x6 = src[j + 6];
          x7 = src[j + 7];
          y0 = x0 + x1;
          y1 = x0 - x1;
          y2 = x2 + x3;
          y3 = modmul_type::multiply(x2 - x3 + N * 2, omega_n82,
                                     omega_n82_precomp);
          y4 = x4 + x5;
          y5 = x4 - x5;
          y6 = x6 + x7;
          y7 = modmul_type::multiply(x6 - x7 + N * 2, omega_n82,
                                     omega_n82_precomp);
          y0 = std::min(y0, y0 - N * 2);
          y1 = std::min(y1, y1 + N * 2);
          y2 = std::min(y2, y2 - N * 2);
          y4 = std::min(y4, y4 - N * 2);
          y5 = std::min(y5, y5 + N * 2);
          y6 = std::min(y6, y6 - N * 2);
          x0 = y0 + y2;
          x1 = y1 + y3;
          x2 = y0 - y2;
          x3 = y1 - y3;
          x4 = y4 + y6;
          x5 = modmul_type::multiply(y5 + y7, omega_n81, omega_n81_precomp);
          x6 = modmul_type::multiply(y4 - y6 + N * 2, omega_n82,
                                     omega_n82_precomp);
          x7 = modmul_type::multiply(y5 - y7 + N * 2, omega_n83,
                                     omega_n83_precomp);
          x0 = std::min(x0, x0 - N * 2);
          x1 = std::min(x1, x1 - N * 2);
          x2 = std::min(x2, x2 + N * 2);
          x3 = std::min(x3, x3 + N * 2);
          x4 = std::min(x4, x4 - N * 2);
          y0 = x0 + x4;
          y1 = x1 + x5;
          y2 = x2 + x6;
          y3 = x3 + x7;
          y4 = x0 - x4;
          y5 = x1 - x5;
          y6 = x2 - x6;
          y7 = x3 - x7;
          y0 = std::min(y0, y0 - N * 2);
          y1 = std::min(y1, y1 - N * 2);
          y2 = std::min(y2, y2 - N * 2);
          y3 = std::min(y3, y3 - N * 2);
          y4 = std::min(y4, y4 + N * 2);
          y5 = std::min(y5, y5 + N * 2);
          y6 = std::min(y6, y6 + N * 2);
          y7 = std::min(y7, y7 + N * 2);
          dst[j + 0] = y0;
          dst[j + 1] = y1;
          dst[j + 2] = y2;
          dst[j + 3] = y3;
          dst[j + 4] = y4;
          dst[j + 5] = y5;
          dst[j + 6] = y6;
          dst[j + 7] = y7;
        }
      } else {
        const std::uint64_t m_inv{
            pointer_utility::get_and_advance<std::uint64_t>(aux)};
        const std::uint64_t omega_n81_m_inv{
            pointer_utility::get_and_advance<std::uint64_t>(aux)};
        const std::uint64_t omega_n82_m_inv{
            pointer_utility::get_and_advance<std::uint64_t>(aux)};
        const std::uint64_t omega_n83_m_inv{
            pointer_utility::get_and_advance<std::uint64_t>(aux)};
        const std::uint64_t m_inv_precomp{modmul_type::precompute(m_inv)};
        const std::uint64_t omega_n81_m_inv_precomp{
            modmul_type::precompute(omega_n81_m_inv)},
            omega_n82_m_inv_precomp{modmul_type::precompute(omega_n82_m_inv)},
            omega_n83_m_inv_precomp{modmul_type::precompute(omega_n83_m_inv)};
        for (std::uint64_t j{}; j < m; j += 8) {
          std::uint64_t x0, x1, x2, x3, x4, x5, x6, x7;
          std::uint64_t y0, y1, y2, y3, y4, y5, y6, y7;
          x0 = src[j + 0];
          x1 = src[j + 1];
          x2 = src[j + 2];
          x3 = src[j + 3];
          x4 = src[j + 4];
          x5 = src[j + 5];
          x6 = src[j + 6];
          x7 = src[j + 7];
          y0 = modmul_type::multiply(x0 + x1, m_inv, m_inv_precomp);
          y1 = modmul_type::multiply(x0 - x1 + N * 2, m_inv, m_inv_precomp);
          y2 = modmul_type::multiply(x2 + x3, m_inv, m_inv_precomp);
          y3 = modmul_type::multiply(x2 - x3 + N * 2, omega_n82_m_inv,
                                     omega_n82_m_inv_precomp);
          y4 = x4 + x5;
          y5 = x4 - x5;
          y6 = x6 + x7;
          y7 = modmul_type::multiply(x6 - x7 + N * 2, omega_n82,
                                     omega_n82_precomp);
          y4 = std::min(y4, y4 - N * 2);
          y5 = std::min(y5, y5 + N * 2);
          y6 = std::min(y6, y6 - N * 2);
          x0 = y0 + y2;
          x1 = y1 + y3;
          x2 = y0 - y2;
          x3 = y1 - y3;
          x4 = modmul_type::multiply(y4 + y6, m_inv, m_inv_precomp);
          x5 = modmul_type::multiply(y5 + y7, omega_n81_m_inv,
                                     omega_n81_m_inv_precomp);
          x6 = modmul_type::multiply(y4 - y6 + N * 2, omega_n82_m_inv,
                                     omega_n82_m_inv_precomp);
          x7 = modmul_type::multiply(y5 - y7 + N * 2, omega_n83_m_inv,
                                     omega_n83_m_inv_precomp);
          x0 = std::min(x0, x0 - N * 2);
          x1 = std::min(x1, x1 - N * 2);
          x2 = std::min(x2, x2 + N * 2);
          x3 = std::min(x3, x3 + N * 2);
          y0 = x0 + x4;
          y1 = x1 + x5;
          y2 = x2 + x6;
          y3 = x3 + x7;
          y4 = x0 - x4;
          y5 = x1 - x5;
          y6 = x2 - x6;
          y7 = x3 - x7;
          y0 = std::min(y0, y0 - N * 2);
          y1 = std::min(y1, y1 - N * 2);
          y2 = std::min(y2, y2 - N * 2);
          y3 = std::min(y3, y3 - N * 2);
          y4 = std::min(y4, y4 + N * 2);
          y5 = std::min(y5, y5 + N * 2);
          y6 = std::min(y6, y6 + N * 2);
          y7 = std::min(y7, y7 + N * 2);
          dst[j + 0] = y0;
          dst[j + 1] = y1;
          dst[j + 2] = y2;
          dst[j + 3] = y3;
          dst[j + 4] = y4;
          dst[j + 5] = y5;
          dst[j + 6] = y6;
          dst[j + 7] = y7;
        }
      }
    } else {
      const std::uint64_t omega_n{
          pointer_utility::get_and_advance<std::uint64_t>(aux)};
      const std::uint64_t omega_n_precomp{modmul_type::precompute(omega_n)};
      std::uint64_t omega_i{unity}, omega_i_n81{omega_n81},
          omega_i_n82{omega_n82}, omega_i_n83{omega_n83}, omega_2i{unity},
          omega_2i_n41{omega_n82}, omega_4i{unity};
      for (std::uint64_t i{}; i < n / 8; ++i) {
        const std::uint64_t omega_i_precomp{modmul_type::precompute(omega_i)},
            omega_i_n81_precomp{modmul_type::precompute(omega_i_n81)},
            omega_i_n82_precomp{modmul_type::precompute(omega_i_n82)},
            omega_i_n83_precomp{modmul_type::precompute(omega_i_n83)},
            omega_2i_precomp{modmul_type::precompute(omega_2i)},
            omega_2i_n41_precomp{modmul_type::precompute(omega_2i_n41)},
            omega_4i_precomp{modmul_type::precompute(omega_4i)};
        for (std::uint64_t j{}; j < m; j += n) {
          std::uint64_t x0, x1, x2, x3, x4, x5, x6, x7;
          std::uint64_t y0, y1, y2, y3, y4, y5, y6, y7;
          x0 = src[i + j + n / 8 * 0];
          x1 = modmul_type::multiply(src[i + j + n / 8 * 1], omega_4i,
                                     omega_4i_precomp);
          x2 = src[i + j + n / 8 * 2];
          x3 = modmul_type::multiply(src[i + j + n / 8 * 3], omega_4i,
                                     omega_4i_precomp);
          x4 = src[i + j + n / 8 * 4];
          x5 = modmul_type::multiply(src[i + j + n / 8 * 5], omega_4i,
                                     omega_4i_precomp);
          x6 = src[i + j + n / 8 * 6];
          x7 = modmul_type::multiply(src[i + j + n / 8 * 7], omega_4i,
                                     omega_4i_precomp);
          y0 = x0 + x1;
          y1 = x0 - x1;
          y2 = modmul_type::multiply(x2 + x3, omega_2i, omega_2i_precomp);
          y3 = modmul_type::multiply(x2 - x3 + N * 2, omega_2i_n41,
                                     omega_2i_n41_precomp);
          y4 = x4 + x5;
          y5 = x4 - x5;
          y6 = modmul_type::multiply(x6 + x7, omega_2i, omega_2i_precomp);
          y7 = modmul_type::multiply(x6 - x7 + N * 2, omega_2i_n41,
                                     omega_2i_n41_precomp);
          y0 = std::min(y0, y0 - N * 2);
          y1 = std::min(y1, y1 + N * 2);
          y4 = std::min(y4, y4 - N * 2);
          y5 = std::min(y5, y5 + N * 2);
          x0 = y0 + y2;
          x1 = y1 + y3;
          x2 = y0 - y2;
          x3 = y1 - y3;
          x4 = modmul_type::multiply(y4 + y6, omega_i, omega_i_precomp);
          x5 = modmul_type::multiply(y5 + y7, omega_i_n81, omega_i_n81_precomp);
          x6 = modmul_type::multiply(y4 - y6 + N * 2, omega_i_n82,
                                     omega_i_n82_precomp);
          x7 = modmul_type::multiply(y5 - y7 + N * 2, omega_i_n83,
                                     omega_i_n83_precomp);
          x0 = std::min(x0, x0 - N * 2);
          x1 = std::min(x1, x1 - N * 2);
          x2 = std::min(x2, x2 + N * 2);
          x3 = std::min(x3, x3 + N * 2);
          y0 = x0 + x4;
          y1 = x1 + x5;
          y2 = x2 + x6;
          y3 = x3 + x7;
          y4 = x0 - x4;
          y5 = x1 - x5;
          y6 = x2 - x6;
          y7 = x3 - x7;
          y0 = std::min(y0, y0 - N * 2);
          y1 = std::min(y1, y1 - N * 2);
          y2 = std::min(y2, y2 - N * 2);
          y3 = std::min(y3, y3 - N * 2);
          y4 = std::min(y4, y4 + N * 2);
          y5 = std::min(y5, y5 + N * 2);
          y6 = std::min(y6, y6 + N * 2);
          y7 = std::min(y7, y7 + N * 2);
          dst[i + j + n / 8 * 0] = y0;
          dst[i + j + n / 8 * 1] = y1;
          dst[i + j + n / 8 * 2] = y2;
          dst[i + j + n / 8 * 3] = y3;
          dst[i + j + n / 8 * 4] = y4;
          dst[i + j + n / 8 * 5] = y5;
          dst[i + j + n / 8 * 6] = y6;
          dst[i + j + n / 8 * 7] = y7;
        }
        omega_i = modmul_type::multiply(omega_i, omega_n, omega_n_precomp);
        omega_i_n81 =
            modmul_type::multiply(omega_i_n81, omega_n, omega_n_precomp);
        omega_i_n82 =
            modmul_type::multiply(omega_i_n82, omega_n, omega_n_precomp);
        omega_i_n83 =
            modmul_type::multiply(omega_i_n83, omega_n, omega_n_precomp);
        omega_i = std::min(omega_i, omega_i - N);
        omega_i_n81 = std::min(omega_i_n81, omega_i_n81 - N);
        omega_i_n82 = std::min(omega_i_n82, omega_i_n82 - N);
        omega_i_n83 = std::min(omega_i_n83, omega_i_n83 - N);
        /* TODO: Update these incrementally. */
        omega_2i = modmul_type::multiply(omega_i, omega_i);
        omega_2i = std::min(omega_2i, omega_2i - N);
        omega_2i_n41 =
            modmul_type::multiply(omega_2i, omega_n82, omega_n82_precomp);
        omega_2i_n41 = std::min(omega_2i_n41, omega_2i_n41 - N);
        omega_4i = modmul_type::multiply(omega_2i, omega_2i);
        omega_4i = std::min(omega_4i, omega_4i - N);
      }
    }
  }

  static void compute_inverse(std::uint64_t *const dst, const std::byte *&aux) {
    compute_inverse(dst, dst, aux);
  }
};

} // namespace sventt

#endif /* SVENTT_LAYER_SCALAR_RADIX_EIGHT_HPP_INCLUDED */

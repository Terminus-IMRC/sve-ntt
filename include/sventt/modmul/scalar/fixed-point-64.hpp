// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_MODMUL_SCALAR_FIXED_POINT_64_HPP_INCLUDED
#define SVENTT_MODMUL_SCALAR_FIXED_POINT_64_HPP_INCLUDED

#include <cstdint>

namespace sventt {

template <class modulus_type_> class FixedPoint64Scalar {

public:
  using modulus_type = modulus_type_;

  static constexpr std::uint64_t to_montgomery(const std::uint64_t b) {
    return b;
  }

  static constexpr std::uint64_t from_montgomery(const std::uint64_t b) {
    return b;
  }

  template <bool do_correction = true>
  static std::uint64_t precompute(const std::uint64_t b) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    constexpr typename modulus_type::shoup_inverse_type N_inv{
        modulus_type::get_shoup_inverse()};

    std::uint64_t bp;
    bp = b * (N_inv.modulus_inverse_lo |
              (static_cast<unsigned __int128>(N_inv.modulus_inverse_hi)
               << 64)) >>
         64;
    if constexpr (do_correction) {
      /* Increment bp if (bp N mod 2^e) + N <= 0. */
      bp += (bp * N + N - 1) >> 63;
    }
    return bp;
  }

  /* Note: The optional correction is only performed for precomputed value. */
  template <bool do_correction = true>
  static std::uint64_t multiply(const std::uint64_t a, const std::uint64_t b) {
    return multiply(a, b, precompute<do_correction>(b));
  }

  static std::uint64_t multiply(const std::uint64_t a, const std::uint64_t b,
                                const std::uint64_t bp) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const std::uint64_t q{static_cast<std::uint64_t>(
        a * static_cast<unsigned __int128>(bp) >> 64)};
    const std::uint64_t c{a * b - q * N};
    return c;
  }
};

} // namespace sventt

#endif /* SVENTT_MODMUL_SCALAR_FIXED_POINT_64_HPP_INCLUDED */

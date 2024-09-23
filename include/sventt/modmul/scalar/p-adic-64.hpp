// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_MODMUL_SCALAR_P_ADIC_64_HPP_INCLUDED
#define SVENTT_MODMUL_SCALAR_P_ADIC_64_HPP_INCLUDED

#include <cstdint>

namespace sventt {

template <class modulus_type_> class PAdic64Scalar {

public:
  using modulus_type = modulus_type_;

  static constexpr std::uint64_t to_montgomery(const std::uint64_t b) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    return modulus_type::multiply(b, -N);
  }

  static constexpr std::uint64_t from_montgomery(const std::uint64_t b) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    return modulus_type::multiply(b, modulus_type::invert(-N));
  }

  static std::uint64_t precompute(const std::uint64_t b) {
    constexpr std::uint64_t N_inv{modulus_type::get_montgomery_inverse()};
    return b * N_inv;
  }

  static std::uint64_t multiply(const std::uint64_t a, const std::uint64_t b) {
    return multiply(a, b, precompute(b));
  }

  static std::uint64_t multiply(const std::uint64_t a, const std::uint64_t b,
                                const std::uint64_t bp) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const std::uint64_t q{a * bp};
    const std::uint64_t ab1{static_cast<std::uint64_t>(
        a * static_cast<unsigned __int128>(b) >> 64)};
    const std::uint64_t qN1{static_cast<std::uint64_t>(
        q * static_cast<unsigned __int128>(N) >> 64)};
    const std::uint64_t c{ab1 - qN1 + N};
    return c;
  }
};

} // namespace sventt

#endif /* SVENTT_MODMUL_SCALAR_P_ADIC_64_HPP_INCLUDED */

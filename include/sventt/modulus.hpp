// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_MODULUS_HPP_INCLUDED
#define SVENTT_MODULUS_HPP_INCLUDED

#include <bit>
#include <cstdint>
#include <stdexcept>

namespace sventt {

/* Note: Assuming the modulus is prime. */
template <std::uint64_t modulus, std::uint64_t generator = 0> class Modulus {

public:
  struct shoup_inverse_type {
    std::uint64_t modulus_inverse_lo, modulus_inverse_hi;
  };

  static constexpr std::uint64_t get_modulus(void) { return modulus; }

  static constexpr std::uint64_t get_generator(void) { return generator; }

  static constexpr shoup_inverse_type get_shoup_inverse(void) {
    const unsigned __int128 modulus_inverse{
        std::has_single_bit(modulus)
            ? (unsigned __int128){1} << (128 - std::countr_zero(modulus))
            : ~(unsigned __int128){} / modulus};
    return {
        .modulus_inverse_lo = static_cast<std::uint64_t>(modulus_inverse),
        .modulus_inverse_hi = static_cast<std::uint64_t>(modulus_inverse >> 64),
    };
  }

  static constexpr std::uint64_t get_montgomery_inverse(void) {
    if constexpr (0) {
      std::uint64_t modulus_inverse{(modulus * 3) ^ 2}; /* 5 */
      modulus_inverse *= 2 - modulus * modulus_inverse; /* 10 */
      modulus_inverse *= 2 - modulus * modulus_inverse; /* 20 */
      modulus_inverse *= 2 - modulus * modulus_inverse; /* 40 */
      modulus_inverse *= 2 - modulus * modulus_inverse; /* 64 */
      return modulus_inverse;
    } else {
      /*
       * Another method obtained by reusing the product of modulus and its
       * inverse in the above Newton-Raphson method, which can be seen as a
       * variant of the Goldschmidt division algorithm.
       * While the number of arithmetic operations keeps the same, it is more
       * likely to be pipelined in exchange for an increased register usage.
       */
      std::uint64_t num, den, t;
      num = (modulus * 3) ^ 2; /* 5 */
      den = num * modulus;
      t = 2 - den;
      num *= t; /* 10 */
      den *= t;
      t = 2 - den;
      num *= t; /* 20 */
      den *= t;
      t = 2 - den;
      num *= t; /* 40 */
      den *= t;
      t = 2 - den;
      num *= t; /* 64 */
      return num;
    }
  }

  static constexpr std::uint64_t add(const std::uint64_t a,
                                     const std::uint64_t b) {
    return (a < modulus - b) ? (a + b) : (a + b - modulus);
  }

  static constexpr std::uint64_t subtract(const std::uint64_t a,
                                          const std::uint64_t b) {
    return (a >= b) ? (a - b) : (a - b + modulus);
  }

  static constexpr std::uint64_t multiply(const std::uint64_t a,
                                          const std::uint64_t b) {
    return a * static_cast<unsigned __int128>(b) % modulus;
  }

  static constexpr std::uint64_t power(std::uint64_t a, std::uint64_t e) {
    std::uint64_t b{1};
    for (; e; e >>= 1) {
      if (e & 1) {
        b = multiply(b, a);
      }
      a = multiply(a, a);
    }
    return b;
  }

  static constexpr std::uint64_t invert(const std::uint64_t a) {
    return power(a, modulus - 2);
  }

  static constexpr std::uint64_t get_root_forward(const std::uint64_t order)
    requires(generator != 0)
  {
    if ((modulus - 1) % order != 0) {
      throw std::invalid_argument{"the field has no such root"};
    }
    return power(generator, (modulus - 1) / order);
  }

  static constexpr std::uint64_t get_root_inverse(const std::uint64_t order)
    requires(generator != 0)
  {
    if ((modulus - 1) % order != 0) {
      throw std::invalid_argument{"the field has no such root"};
    }
    return power(generator, Modulus<modulus - 1>::multiply(
                                (modulus - 1) / order, modulus - 2));
  }
};

} // namespace sventt

#endif /* SVENTT_MODULUS_HPP_INCLUDED */

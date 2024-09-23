// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_MODMUL_SVE_P_ADIC_64_HPP_INCLUDED
#define SVENTT_MODMUL_SVE_P_ADIC_64_HPP_INCLUDED

#include <bit>
#include <cstdint>

#include <arm_sve.h>

namespace sventt {

template <class modulus_type_> class PAdic64SVE {

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

  template <bool do_correction = true>
  static svuint64_t precompute(const svuint64_t b) {
    constexpr std::uint64_t N_inv{modulus_type::get_montgomery_inverse()};
    const svbool_t ptrue{svptrue_b8()};
    return svmul_x(ptrue, b, N_inv);
  }

  static svuint64_t multiply(const svuint64_t a, const svuint64_t b) {
    return multiply(a, b, precompute(b));
  }

  static svuint64_t multiply(const svuint64_t a, const svuint64_t b,
                             const svuint64_t bp) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svbool_t ptrue{svptrue_b8()};
    const svuint64_t q{svmul_x(ptrue, a, bp)};
    const svuint64_t ab1{svmulh_x(ptrue, a, b)};
    const svuint64_t qN1{svmulh_x(ptrue, q, N)};
    svuint64_t c;
    if constexpr (std::bit_width(N) <= 63) {
      c = svsub_x(ptrue, svadd_x(ptrue, ab1, N), qN1);
    } else {
      c = svsub_x(ptrue, ab1, qN1);
      c = svadd_m(svcmplt(ptrue, ab1, qN1), c, N);
    }
    return c;
  }

  static svuint64_t multiply_normalize(const svuint64_t a, const svuint64_t b,
                                       const svuint64_t bp) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svbool_t ptrue{svptrue_b8()};
    const svuint64_t q{svmul_x(ptrue, a, bp)};
    const svuint64_t ab1{svmulh_x(ptrue, a, b)};
    const svuint64_t qN1{svmulh_x(ptrue, q, N)};
    svuint64_t c{svsub_x(ptrue, ab1, qN1)};
    if constexpr (std::bit_width(N) <= 63) {
      c = svmin_x(ptrue, c, svadd_x(ptrue, c, N));
    } else {
      c = svadd_m(svcmplt(ptrue, ab1, qN1), c, N);
    }
    return c;
  }

  static void butterfly_forward(svuint64_t &x0, svuint64_t &x1) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svbool_t ptrue{svptrue_b8()};

    svuint64_t y0;
    if constexpr (std::bit_width(N) <= 62) {
      y0 = svadd_x(ptrue, x0, x1);
      y0 = svmin_x(ptrue, y0, svsub_x(ptrue, y0, N * 2));
    } else if constexpr (std::bit_width(N) == 63) {
      y0 = svadd_x(ptrue, x0, x1);
      y0 = svmin_x(ptrue, y0, svsub_x(ptrue, y0, N));
    } else {
      const svuint64_t x1n{svsubr_x(ptrue, x1, N)};
      y0 = svsub_x(ptrue, x0, x1n);
      y0 = svadd_m(svcmplt(ptrue, x0, x1n), y0, N);
    }

    svuint64_t y1{svsub_x(ptrue, x0, x1)};
    if constexpr (std::bit_width(N) <= 62) {
      y1 = svmin_x(ptrue, y1, svadd_x(ptrue, y1, N * 2));
    } else if constexpr (std::bit_width(N) == 63) {
      y1 = svmin_x(ptrue, y1, svadd_x(ptrue, y1, N));
    } else {
      y1 = svadd_m(svcmplt(ptrue, x0, x1), y1, N);
    }

    x0 = y0;
    x1 = y1;
  }

  static void butterfly_forward(svuint64_t &x0, svuint64_t &x1,
                                const svuint64_t w, const svuint64_t wp) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svbool_t ptrue{svptrue_b8()};

    svuint64_t y0;
    if constexpr (std::bit_width(N) <= 62) {
      y0 = svadd_x(ptrue, x0, x1);
      y0 = svmin_x(ptrue, y0, svsub_x(ptrue, y0, N * 2));
    } else if constexpr (std::bit_width(N) == 63) {
      y0 = svadd_x(ptrue, x0, x1);
      y0 = svmin_x(ptrue, y0, svsub_x(ptrue, y0, N));
    } else {
      const svuint64_t x1n{svsubr_x(ptrue, x1, N)};
      y0 = svsub_x(ptrue, x0, x1n);
      y0 = svadd_m(svcmplt(ptrue, x0, x1n), y0, N);
    }

    svuint64_t y1;
    if constexpr (std::bit_width(N) <= 62) {
      y1 = svsub_x(ptrue, svadd_x(ptrue, x0, N * 2), x1);
    } else if constexpr (std::bit_width(N) == 63) {
      y1 = svsub_x(ptrue, svadd_x(ptrue, x0, N), x1);
    } else {
      y1 = svsub_x(ptrue, x0, x1);
      y1 = svadd_m(svcmplt(ptrue, x0, x1), y1, N);
    }

    const svuint64_t q{svmul_x(ptrue, y1, wp)};
    const svuint64_t y1w_high{svmulh_x(ptrue, y1, w)};
    const svuint64_t qN_high{svmulh_x(ptrue, q, N)};
    if constexpr (std::bit_width(N) <= 62) {
      y1 = svsub_x(ptrue, svadd_x(ptrue, y1w_high, N), qN_high);
    } else if constexpr (std::bit_width(N) == 63) {
      y1 = svsub_x(ptrue, y1w_high, qN_high);
      y1 = svmin_x(ptrue, y1, svadd_x(ptrue, y1, N));
    } else {
      y1 = svsub_x(ptrue, y1w_high, qN_high);
      y1 = svadd_m(svcmplt(ptrue, y1w_high, qN_high), y1, N);
    }

    x0 = y0;
    x1 = y1;
  }

  static void butterfly_forward(svuint64_t &x0, svuint64_t &x1,
                                const svuint64_t w0, const svuint64_t w0p,
                                const svuint64_t w1, const svuint64_t w1p) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svbool_t ptrue{svptrue_b8()};

    svuint64_t y0{svadd_x(ptrue, x0, x1)};
    if constexpr (std::bit_width(N) == 64) {
      y0 = svsub_m(svcmplt(ptrue, y0, x0), y0, N);
    }

    const svuint64_t q0{svmul_x(ptrue, y0, w0p)};
    const svuint64_t y0w0_high{svmulh_x(ptrue, y0, w0)};
    const svuint64_t q0N_high{svmulh_x(ptrue, q0, N)};
    if constexpr (std::bit_width(N) <= 62) {
      y0 = svsub_x(ptrue, svadd_x(ptrue, y0w0_high, N), q0N_high);
    } else if constexpr (std::bit_width(N) == 63) {
      y0 = svsub_x(ptrue, y0w0_high, q0N_high);
      y0 = svmin_x(ptrue, y0, svadd_x(ptrue, y0, N));
    } else {
      y0 = svsub_x(ptrue, y0w0_high, q0N_high);
      y0 = svadd_m(svcmplt(ptrue, y0w0_high, q0N_high), y0, N);
    }

    svuint64_t y1;
    if constexpr (std::bit_width(N) <= 62) {
      y1 = svsub_x(ptrue, svadd_x(ptrue, x0, N * 2), x1);
    } else if constexpr (std::bit_width(N) == 63) {
      y1 = svsub_x(ptrue, svadd_x(ptrue, x0, N), x1);
    } else {
      y1 = svsub_x(ptrue, x0, x1);
      y1 = svadd_m(svcmplt(ptrue, x0, x1), y1, N);
    }

    const svuint64_t q1{svmul_x(ptrue, y1, w1p)};
    const svuint64_t y1w1_high{svmulh_x(ptrue, y1, w1)};
    const svuint64_t q1N_high{svmulh_x(ptrue, q1, N)};
    if constexpr (std::bit_width(N) <= 62) {
      y1 = svsub_x(ptrue, svadd_x(ptrue, y1w1_high, N), q1N_high);
    } else if constexpr (std::bit_width(N) == 63) {
      y1 = svsub_x(ptrue, y1w1_high, q1N_high);
      y1 = svmin_x(ptrue, y1, svadd_x(ptrue, y1, N));
    } else {
      y1 = svsub_x(ptrue, y1w1_high, q1N_high);
      y1 = svadd_m(svcmplt(ptrue, y1w1_high, q1N_high), y1, N);
    }

    x0 = y0;
    x1 = y1;
  }

  static void butterfly_inverse(svuint64_t &x0, svuint64_t &x1) {
    butterfly_forward(x0, x1);
  }

  static void butterfly_inverse(svuint64_t &x0, svuint64_t &x1,
                                const svuint64_t w, const svuint64_t wp) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svbool_t ptrue{svptrue_b8()};

    const svuint64_t q{svmul_x(ptrue, x1, wp)};
    const svuint64_t x1w_high{svmulh_x(ptrue, x1, w)};
    const svuint64_t qN_high{svmulh_x(ptrue, q, N)};
    if constexpr (std::bit_width(N) <= 62) {
      x1 = svsub_x(ptrue, svadd_x(ptrue, x1w_high, N), qN_high);
    } else if constexpr (std::bit_width(N) == 63) {
      x1 = svsub_x(ptrue, x1w_high, qN_high);
      x1 = svmin_x(ptrue, x1, svadd_x(ptrue, x1, N));
    } else {
      x1 = svsub_x(ptrue, x1w_high, qN_high);
      x1 = svadd_m(svcmplt(ptrue, x1w_high, qN_high), x1, N);
    }

    butterfly_inverse(x0, x1);
  }
};

} // namespace sventt

#endif /* SVENTT_MODMUL_SVE_P_ADIC_64_HPP_INCLUDED */

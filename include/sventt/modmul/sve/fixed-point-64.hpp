// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_MODMUL_SVE_FIXED_POINT_64_HPP_INCLUDED
#define SVENTT_MODMUL_SVE_FIXED_POINT_64_HPP_INCLUDED

#include <cstdint>

#include <arm_sve.h>

namespace sventt {

template <class modulus_type_> class FixedPoint64SVE {

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
    const svbool_t pfirst{svptrue_pat_b64(SV_VL1)};
    return svlastb(pfirst, precompute<do_correction>(svdup_u64(b)));
  }

  template <bool do_correction = true>
  static svuint64_t precompute(const svuint64_t b) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    constexpr typename modulus_type::shoup_inverse_type N_inv{
        modulus_type::get_shoup_inverse()};
    const svbool_t ptrue{svptrue_b8()};

    svuint64_t bp;
    bp = svmulh_x(ptrue, b, N_inv.modulus_inverse_lo);
    bp = svmla_x(ptrue, bp, b, N_inv.modulus_inverse_hi);
    if constexpr (do_correction) {
      /* Increment bp if (bp N mod 2^e) + N <= 0. */
#if defined(__ARM_FEATURE_SVE2) && 0
      /* Disabled due to less throughput on Neoverse. */
      bp = svsra(bp, svmad_x(ptrue, bp, svdup_u64(N), N - 1), 63);
#else
      bp = svadd_x(ptrue, bp,
                   svlsr_x(ptrue, svmad_x(ptrue, bp, svdup_u64(N), N - 1), 63));
#endif
    }
    return bp;
  }

  template <bool do_correction = true>
  static svuint64_t multiply(const svuint64_t a, const svuint64_t b) {
    return multiply(a, b, precompute<do_correction>(b));
  }

  static svuint64_t multiply(const svuint64_t a, const svuint64_t b,
                             const svuint64_t bp) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    const svbool_t ptrue{svptrue_b8()};
    const svuint64_t q{svmulh_x(ptrue, a, bp)};
    const svuint64_t ab{svmul_x(ptrue, a, b)};
    const svuint64_t c{svmls_x(ptrue, ab, q, N)};
    return c;
  }
};

} // namespace sventt

#endif /* SVENTT_MODMUL_SVE_FIXED_POINT_64_HPP_INCLUDED */

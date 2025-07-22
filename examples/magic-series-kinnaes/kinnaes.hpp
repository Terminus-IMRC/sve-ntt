// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef EXAMPLES_MAGIC_SERIES_KINNAES_HPP_INCLUDED
#define EXAMPLES_MAGIC_SERIES_KINNAES_HPP_INCLUDED

#include <bit>
#include <cstdint>

#include <sventt/sventt.hpp>

template <std::uint64_t m, class modmul_type_, std::uint64_t n>
class MagicSeriesKinnaes {

public:
  using modmul_type = modmul_type_;
  using modulus_type = modmul_type::modulus_type;

  static constexpr std::uint64_t r{m * (m - 1) / 2 * m};

  static constexpr std::uint64_t get_m(void) { return m; }

  static constexpr std::uint64_t get_n(void) { return n; }

  static constexpr std::uint64_t get_r(void) { return r; }

  static std::uint64_t compute(void) {
    std::uint64_t sum;
    sum = compute_sum();
    sum = modulus_type::add(sum, sum);
    sum = modulus_type::add(sum, compute_comb(m * m, m));
    sum = modulus_type::divide(sum, n);
    return sum;
  }

  static std::uint64_t compute_comb(const std::uint64_t a,
                                    const std::uint64_t b) {
    std::uint64_t num{a};
    for (std::uint64_t i{1}; i < b; ++i) {
      num = modulus_type::multiply(num, a - i);
    }
    std::uint64_t den{b};
    for (std::uint64_t i{2}; i < b; ++i) {
      den = modulus_type::multiply(den, i);
    }
    return modulus_type::divide(num, den);
  }

  static std::uint64_t compute_sum(void) { return compute_sum(0, n / 2); }

  static std::uint64_t compute_sum(const std::uint64_t j_begin,
                                   const std::uint64_t j_end) {
    const svbool_t ptrue{svptrue_b8()};
    constexpr std::uint64_t cntd{sventt::cntd};

    constexpr std::uint64_t omega{modulus_type::get_root_forward(n)};
    constexpr std::uint64_t unity{modmul_type::to_montgomery(1)};

    /* omega^{j (m (m - 1) + 1)} */
    svuint64_t num_term_init;
    {
      const std::uint64_t step{modulus_type::power(omega, m * m - m + 1)};
      std::uint64_t val{modulus_type::power(step, j_begin + 1)};
      std::uint64_t vals[cntd];
      for (std::uint64_t i{0}; i < cntd; ++i) {
        vals[i] = val;
        val = modulus_type::multiply(val, step);
      }
      num_term_init = svld1_u64(ptrue, vals);
    }
    /* omega^{cntd (m (m - 1) + 1)} */
    const svuint64_t num_term_init_step{svdup_u64(modmul_type::to_montgomery(
        modulus_type::power(omega, cntd * (m * m - m + 1))))};

    /* omega^j */
    svuint64_t den_term_init;
    {
      const std::uint64_t step{omega};
      std::uint64_t val{modulus_type::power(step, j_begin + 1)};
      std::uint64_t vals[cntd];
      for (std::uint64_t i{0}; i < cntd; ++i) {
        vals[i] = modmul_type::to_montgomery(val);
        val = modulus_type::multiply(val, step);
      }
      den_term_init = svld1_u64(ptrue, vals);
    }
    /* omega^cntd */
    const svuint64_t den_term_init_step{svdup_u64(
        modmul_type::to_montgomery(modulus_type::power(omega, cntd)))};

    /* omega^{j r} */
    svuint64_t den_prod_init;
    {
      const std::uint64_t step{modulus_type::power(omega, r)};
      std::uint64_t val{modulus_type::power(step, j_begin + 1)};
      std::uint64_t vals[cntd];
      for (std::uint64_t i{0}; i < cntd; ++i) {
        vals[i] = val;
        val = modulus_type::multiply(val, step);
      }
      den_prod_init = svld1_u64(ptrue, vals);
    }
    /* omega^{cntd r} */
    const svuint64_t den_prod_init_step{svdup_u64(
        modmul_type::to_montgomery(modulus_type::power(omega, cntd * r)))};

    svuint64_t num_sum{svdup_u64(0)};
    svuint64_t den_sum{svdup_u64(unity)};
    for (std::uint64_t j{j_begin}; j < j_end; j += cntd) {
      svuint64_t num_prod{svdup_u64(1)}, den_prod{den_prod_init};
      svuint64_t num_term{num_term_init},
          den_term{modmul_type::from_montgomery(den_term_init)};
      const svuint64_t den_term_init_precomp{
          modmul_type::precompute(den_term_init)};
      for (std::uint64_t l{}; l < m; ++l) {
        num_prod = modmul_type::multiply(svsub_x(ptrue, num_term, 1), num_prod);
        den_prod = modmul_type::multiply(svsub_x(ptrue, den_term, 1), den_prod);
        if constexpr (std::bit_width(modulus_type::get_modulus()) == 63) {
          /* N <= 2^62 is required for inputs both being lazy. */
          num_term = modmul_type::multiply_normalize(num_term, den_term_init,
                                                     den_term_init_precomp);
          den_term = modmul_type::multiply_normalize(den_term, den_term_init,
                                                     den_term_init_precomp);
        } else {
          num_term = modmul_type::multiply(num_term, den_term_init,
                                           den_term_init_precomp);
          den_term = modmul_type::multiply(den_term, den_term_init,
                                           den_term_init_precomp);
        }
      }

      const svbool_t valid{svwhilelt_b64(j, j_end)};
      const svuint64_t tmp0{modmul_type::multiply_normalize(den_sum, num_prod)};
      const svuint64_t tmp1{modmul_type::multiply_normalize(num_sum, den_prod)};
      num_sum = svsel(valid, modmul_type::add(tmp0, tmp1), num_sum);
      den_sum = svsel(valid, modmul_type::multiply_normalize(den_sum, den_prod),
                      den_sum);

      num_term_init =
          modmul_type::multiply_normalize(num_term_init, num_term_init_step);
      den_term_init =
          modmul_type::multiply_normalize(den_term_init, den_term_init_step);
      den_prod_init = modmul_type::multiply(den_prod_init, den_prod_init_step);
    }

    std::uint64_t nums[cntd], dens[cntd];
    svst1_u64(ptrue, nums, num_sum);
    svst1_u64(ptrue, dens, den_sum);
    std::uint64_t num{nums[0]}, den{dens[0]};
    for (std::uint64_t i{1}; i < cntd; ++i) {
      const std::uint64_t tmp0{modulus_type::multiply(den, nums[i])};
      const std::uint64_t tmp1{modulus_type::multiply(num, dens[i])};
      num = modulus_type::add(tmp0, tmp1);
      den = modulus_type::multiply(den, dens[i]);
    }
    return modulus_type::divide(num, den);
  }
};

#endif /* EXAMPLES_MAGIC_SERIES_KINNAES_HPP_INCLUDED */

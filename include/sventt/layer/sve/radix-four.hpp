// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_LAYER_SVE_RADIX_FOUR_HPP_INCLUDED
#define SVENTT_LAYER_SVE_RADIX_FOUR_HPP_INCLUDED

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <arm_sve.h>

#include "sventt/common/sve.hpp"

namespace sventt {

template <class modmul_type_, std::uint64_t m, std::uint64_t n,
          std::uint64_t inverse_factor = 1, bool store_precomputation = true>
class RadixFourSVELayer {

public:
  using modmul_type = modmul_type_;
  using modulus_type = modmul_type::modulus_type;

  static constexpr std::uint64_t get_radix(void) { return 4; }

  static constexpr std::uint64_t get_m(void) { return m; }

  static constexpr std::uint64_t get_n(void) { return n; }

  static_assert(n >= get_radix());

  template <class vector_type> static void prepare_forward(vector_type &aux) {
    const auto precompute{[&](const std::uint64_t num_elements,
                              std::uint64_t &omega, const std::uint64_t factor,
                              const bool precompute_precompute) {
      std::vector<std::uint64_t> v;
      for (std::uint64_t i{}; i < num_elements; ++i) {
        v.push_back(modmul_type::to_montgomery(omega));
        aux.push_back(v.back());
        omega = modulus_type::multiply(omega, factor);
      }
      if (precompute_precompute) {
        for (std::uint64_t i{}; i < num_elements; ++i) {
          aux.push_back(modmul_type::precompute(v.at(i)));
        }
      }
    }};

    const std::uint64_t omega_n41{modulus_type::get_root_forward(4)};
    const std::uint64_t omega_n_1{modulus_type::get_root_forward(n)},
        omega_n_2{modulus_type::get_root_forward(n / 2)};
    if (n / 4 >= cntd) {
      if (!store_precomputation) {
        for (std::uint64_t omega : {
                 modulus_type::get_root_forward(n / cntd),
                 modulus_type::get_root_forward(n / 2 / cntd),
             }) {
          precompute(1, omega, {}, true);
        }
      }

      std::uint64_t omega_n_1i_n40{1}, omega_n_1i_n41{omega_n41},
          omega_n_2i_n20{1};
      for (std::uint64_t i{}; i < n / 4; i += cntd) {
        if (i == 0 || store_precomputation) {
          precompute(cntd, omega_n_1i_n40, omega_n_1, store_precomputation);
          precompute(cntd, omega_n_1i_n41, omega_n_1, store_precomputation);
          precompute(cntd, omega_n_2i_n20, omega_n_2, store_precomputation);
        }
      }
    } else if (n == 4) {
      if (m < cntd * 4) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      std::uint64_t omega_n_1i{omega_n_1};
      precompute(1, omega_n_1i, {}, true);
    } else if (n == 16) {
      if (m < cntd * 4) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      std::uint64_t omega_n_1i_n40{1};
      precompute(4, omega_n_1i_n40, omega_n_1, true);
      std::uint64_t omega_n_1i_n41{omega_n41};
      precompute(4, omega_n_1i_n41, omega_n_1, true);
      std::uint64_t omega_2i_n20{1};
      precompute(4, omega_2i_n20, omega_n_2, true);
    } else {
      throw std::invalid_argument{"Unsupported SVE vector length"};
    }
  }

  static void compute_forward(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux_arg) {
    const svbool_t ptrue{svptrue_b8()};
    const std::uint64_t *aux{reinterpret_cast<const std::uint64_t *>(aux_arg)};

    const auto load_or_precompute{
        [&](svuint64_t &w, svuint64_t &wp, const bool do_initialize,
            const svuint64_t factor, const svuint64_t factor_precomp) {
          if constexpr (store_precomputation) {
            w = svld1_vnum(ptrue, aux, 0);
            wp = svld1_vnum(ptrue, aux, 1);
            aux += cntd * 2;
          } else {
            if (do_initialize) {
              w = svld1_vnum(ptrue, aux, 0);
              aux += cntd * 1;
            } else {
              w = modmul_type::multiply_normalize(w, factor, factor_precomp);
            }
            wp = modmul_type::precompute(w);
          }
        }};

    if constexpr (n / 4 >= cntd) {
      svuint64_t omega_n_cntd, omega_n_cntd_precomp, omega_n_2cntd,
          omega_n_2cntd_precomp;
      if constexpr (!store_precomputation) {
        broadcast_and_advance(omega_n_cntd, omega_n_cntd_precomp, omega_n_2cntd,
                              omega_n_2cntd_precomp, aux);
      }

      const std::uint64_t *const aux_orig{aux};
      for (std::uint64_t j{}; j < m; j += n) {
        aux = aux_orig;

        svuint64_t omega_n_1i_n40, omega_n_1i_n40_precomp, omega_n_1i_n41,
            omega_n_1i_n41_precomp, omega_n_2i_n20, omega_n_2i_n20_precomp;
        for (std::uint64_t i{}; i < n / 4; i += cntd) {
          svuint64_t x0, x1, x2, x3;
          x0 = svld1(ptrue, &src[i + j + n / 4 * 0]);
          x1 = svld1(ptrue, &src[i + j + n / 4 * 1]);
          x2 = svld1(ptrue, &src[i + j + n / 4 * 2]);
          x3 = svld1(ptrue, &src[i + j + n / 4 * 3]);

          load_or_precompute(omega_n_1i_n40, omega_n_1i_n40_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);
          load_or_precompute(omega_n_1i_n41, omega_n_1i_n41_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);

          modmul_type::butterfly_forward(x0, x2, omega_n_1i_n40,
                                         omega_n_1i_n40_precomp);
          modmul_type::butterfly_forward(x1, x3, omega_n_1i_n41,
                                         omega_n_1i_n41_precomp);

          load_or_precompute(omega_n_2i_n20, omega_n_2i_n20_precomp, i == 0,
                             omega_n_2cntd, omega_n_2cntd_precomp);

          modmul_type::butterfly_forward(x0, x1, omega_n_2i_n20,
                                         omega_n_2i_n20_precomp);
          modmul_type::butterfly_forward(x2, x3, omega_n_2i_n20,
                                         omega_n_2i_n20_precomp);

          svst1(ptrue, &dst[i + j + n / 4 * 0], x0);
          svst1(ptrue, &dst[i + j + n / 4 * 1], x1);
          svst1(ptrue, &dst[i + j + n / 4 * 2], x2);
          svst1(ptrue, &dst[i + j + n / 4 * 3], x3);
        }
      }
    } else if constexpr (n == 4) {
      svuint64_t omega41, omega41_precomp;
      broadcast_and_advance(omega41, omega41_precomp, aux);

      for (std::uint64_t j{}; j < m; j += cntd * 4) {
        svuint64_t x0, x1, x2, x3;
        load_and_deinterleave<1>(x0, x1, x2, x3, &src[j]);

        modmul_type::butterfly_forward(x0, x2);
        modmul_type::butterfly_forward(x1, x3, omega41, omega41_precomp);

        modmul_type::butterfly_forward(x0, x1);
        modmul_type::butterfly_forward(x2, x3);

        interleave_and_store<2>(&dst[j], x0, x1, x2, x3);
      }
    } else if constexpr (n == 16) {
      svuint64_t omega_i, omega_i_precomp, omega_i_n4, omega_i_n4_precomp,
          omega_2i, omega_2i_precomp;
      if constexpr (cntd == 8) {
        omega_i = svld1(ptrue, aux);
        aux += cntd;
        omega_i_precomp = svzip2(omega_i, omega_i);
        omega_i = svzip1(omega_i, omega_i);

        omega_i_n4 = svld1(ptrue, aux);
        aux += cntd;
        omega_i_n4_precomp = svzip2(omega_i_n4, omega_i_n4);
        omega_i_n4 = svzip1(omega_i_n4, omega_i_n4);

        omega_2i = svld1(ptrue, aux);
        aux += cntd;
        omega_2i_precomp = svzip2(omega_2i, omega_2i);
        omega_2i = svzip1(omega_2i, omega_2i);
      } else {
        throw std::invalid_argument{"Unsupported SVE vector length"};
      }

      for (std::uint64_t j{}; j < m; j += cntd * 4) {
        svuint64_t x0, x1, x2, x3;
        svuint64_t y0, y1, y2, y3;

        /*
         * 0:  0  1  2  3  4  5  6  7
         * 1:  8  9 10 11 12 13 14 15
         * 2: 16 17 18 19 20 21 22 23
         * 3: 24 25 26 27 28 29 30 31
         */
        x0 = svld1_vnum(ptrue, &src[j], 0);
        x1 = svld1_vnum(ptrue, &src[j], 1);
        x2 = svld1_vnum(ptrue, &src[j], 2);
        x3 = svld1_vnum(ptrue, &src[j], 3);

        /*
         * 0:  0 16  1 17  2 18  3 19
         * 1:  8 24  9 25 10 26 11 27
         * 2:  4 20  5 21  6 22  7 23
         * 3: 12 28 13 29 14 30 15 31
         */
        y0 = svzip1(x0, x2);
        y1 = svzip1(x1, x3);
        y2 = svzip2(x0, x2);
        y3 = svzip2(x1, x3);

        modmul_type::butterfly_forward(y0, y1, omega_i, omega_i_precomp);
        modmul_type::butterfly_forward(y2, y3, omega_i_n4, omega_i_n4_precomp);

        modmul_type::butterfly_forward(y0, y2, omega_2i, omega_2i_precomp);
        modmul_type::butterfly_forward(y1, y3, omega_2i, omega_2i_precomp);

        /*
         * 0:  0  1  2  3  4  5  6  7
         * 1:  8  9 10 11 12 13 14 15
         * 2: 16 17 18 19 20 21 22 23
         * 3: 24 25 26 27 28 29 30 31
         */
        x0 = svuzp1(y0, y2);
        x1 = svuzp1(y1, y3);
        x2 = svuzp2(y0, y2);
        x3 = svuzp2(y1, y3);

        svst1_vnum(ptrue, &dst[j], 0, x0);
        svst1_vnum(ptrue, &dst[j], 1, x1);
        svst1_vnum(ptrue, &dst[j], 2, x2);
        svst1_vnum(ptrue, &dst[j], 3, x3);
      }
    } else {
      throw std::invalid_argument{"Unsupported SVE vector length"};
    }

    aux_arg = reinterpret_cast<const std::byte *>(aux);
  }

  static void compute_forward(std::uint64_t *const dst, const std::byte *&aux) {
    compute_forward(dst, dst, aux);
  }

  template <class vector_type> static void prepare_inverse(vector_type &aux) {
    const auto precompute{[&](const std::uint64_t num_elements,
                              std::uint64_t &omega, const std::uint64_t factor,
                              const bool precompute_precompute) {
      std::vector<std::uint64_t> v;
      for (std::uint64_t i{}; i < num_elements; ++i) {
        v.push_back(modmul_type::to_montgomery(omega));
        aux.push_back(v.back());
        omega = modulus_type::multiply(omega, factor);
      }
      if (precompute_precompute) {
        for (std::uint64_t i{}; i < num_elements; ++i) {
          aux.push_back(modmul_type::precompute(v.at(i)));
        }
      }
    }};

    const std::uint64_t omega_n41{modulus_type::get_root_inverse(4)};
    const std::uint64_t omega_n_1{modulus_type::get_root_inverse(n)},
        omega_n_2{modulus_type::get_root_inverse(n / 2)};
    if (n / 4 >= cntd) {
      if (!store_precomputation) {
        for (std::uint64_t omega : {
                 modulus_type::get_root_inverse(n / cntd),
                 modulus_type::get_root_inverse(n / 2 / cntd),
             }) {
          precompute(1, omega, {}, true);
        }
      }

      std::uint64_t omega_n_1i_n40{1}, omega_n_1i_n41{omega_n41},
          omega_n_2i_n20{1};
      for (std::uint64_t i{}; i < n / 4; i += cntd) {
        if (i == 0 || store_precomputation) {
          precompute(cntd, omega_n_2i_n20, omega_n_2, store_precomputation);
          precompute(cntd, omega_n_1i_n40, omega_n_1, store_precomputation);
          precompute(cntd, omega_n_1i_n41, omega_n_1, store_precomputation);
        }
      }
    } else if (n == 4) {
      if (m < cntd * 4) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      if constexpr (inverse_factor == 1) {
        std::uint64_t omega_n_1i{omega_n_1};
        precompute(1, omega_n_1i, {}, true);
      } else {
        const std::uint64_t m_inv{modulus_type::invert(inverse_factor)};
        std::uint64_t m_inv_i{m_inv};
        precompute(1, m_inv_i, {}, true);
        std::uint64_t omega_n_1i_m_inv{
            modulus_type::multiply(omega_n_1, m_inv)};
        precompute(1, omega_n_1i_m_inv, {}, true);
      }
    } else if (n == 16) {
      if (m < cntd * 4) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      std::uint64_t omega_n_1i_n40{1};
      precompute(4, omega_n_1i_n40, omega_n_1, true);
      std::uint64_t omega_n_1i_n41{omega_n41};
      precompute(4, omega_n_1i_n41, omega_n_1, true);
      std::uint64_t omega_2i_n20{1};
      precompute(4, omega_2i_n20, omega_n_2, true);
    } else {
      throw std::invalid_argument{"Unsupported SVE vector length"};
    }
  }

  static void compute_inverse(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux_arg) {
    const svbool_t ptrue{svptrue_b8()};
    const std::uint64_t *aux{reinterpret_cast<const std::uint64_t *>(aux_arg)};

    const auto load_or_precompute{
        [&](svuint64_t &w, svuint64_t &wp, const bool do_initialize,
            const svuint64_t factor, const svuint64_t factor_precomp) {
          if constexpr (store_precomputation) {
            w = svld1_vnum(ptrue, aux, 0);
            wp = svld1_vnum(ptrue, aux, 1);
            aux += cntd * 2;
          } else {
            if (do_initialize) {
              w = svld1_vnum(ptrue, aux, 0);
              aux += cntd * 1;
            } else {
              w = modmul_type::multiply_normalize(w, factor, factor_precomp);
            }
            wp = modmul_type::precompute(w);
          }
        }};

    if constexpr (n / 4 >= cntd) {
      svuint64_t omega_n_cntd, omega_n_cntd_precomp, omega_n_2cntd,
          omega_n_2cntd_precomp;
      if constexpr (!store_precomputation) {
        broadcast_and_advance(omega_n_cntd, omega_n_cntd_precomp, omega_n_2cntd,
                              omega_n_2cntd_precomp, aux);
      }

      const std::uint64_t *const aux_orig{aux};
      for (std::uint64_t j{}; j < m; j += n) {
        aux = aux_orig;

        svuint64_t omega_n_1i_n40, omega_n_1i_n40_precomp, omega_n_1i_n41,
            omega_n_1i_n41_precomp, omega_n_2i_n20, omega_n_2i_n20_precomp;
        for (std::uint64_t i{}; i < n / 4; i += cntd) {
          svuint64_t x0, x1, x2, x3;
          x0 = svld1(ptrue, &src[i + j + n / 4 * 0]);
          x1 = svld1(ptrue, &src[i + j + n / 4 * 1]);
          x2 = svld1(ptrue, &src[i + j + n / 4 * 2]);
          x3 = svld1(ptrue, &src[i + j + n / 4 * 3]);

          load_or_precompute(omega_n_2i_n20, omega_n_2i_n20_precomp, i == 0,
                             omega_n_2cntd, omega_n_2cntd_precomp);

          modmul_type::butterfly_inverse(x0, x1, omega_n_2i_n20,
                                         omega_n_2i_n20_precomp);
          modmul_type::butterfly_inverse(x2, x3, omega_n_2i_n20,
                                         omega_n_2i_n20_precomp);

          load_or_precompute(omega_n_1i_n40, omega_n_1i_n40_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);
          load_or_precompute(omega_n_1i_n41, omega_n_1i_n41_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);

          modmul_type::butterfly_inverse(x0, x2, omega_n_1i_n40,
                                         omega_n_1i_n40_precomp);
          modmul_type::butterfly_inverse(x1, x3, omega_n_1i_n41,
                                         omega_n_1i_n41_precomp);

          svst1(ptrue, &dst[i + j + n / 4 * 0], x0);
          svst1(ptrue, &dst[i + j + n / 4 * 1], x1);
          svst1(ptrue, &dst[i + j + n / 4 * 2], x2);
          svst1(ptrue, &dst[i + j + n / 4 * 3], x3);
        }
      }
    } else if constexpr (n == 4) {
      if constexpr (inverse_factor == 1) {
        svuint64_t omega41, omega41_precomp;
        broadcast_and_advance(omega41, omega41_precomp, aux);

        for (std::uint64_t j{}; j < m; j += cntd * 4) {
          svuint64_t x0, x1, x2, x3;
          load_and_deinterleave<1>(x0, x1, x2, x3, &src[j]);

          modmul_type::butterfly_forward(x0, x1);
          modmul_type::butterfly_forward(x2, x3, omega41, omega41_precomp);

          modmul_type::butterfly_forward(x0, x2);
          modmul_type::butterfly_forward(x1, x3);

          interleave_and_store<2>(&dst[j], x0, x1, x2, x3);
        }
      } else {
        svuint64_t m_inv, m_inv_precomp, omega41_m_inv, omega41_m_inv_precomp;
        broadcast_and_advance(m_inv, m_inv_precomp, omega41_m_inv,
                              omega41_m_inv_precomp, aux);

        for (std::uint64_t j{}; j < m; j += cntd * 4) {
          svuint64_t x0, x1, x2, x3;
          load_and_deinterleave<1>(x0, x1, x2, x3, &src[j]);

          modmul_type::butterfly_forward(x0, x1, m_inv, m_inv_precomp);
          modmul_type::butterfly_forward(x2, x3, omega41_m_inv,
                                         omega41_m_inv_precomp);

          modmul_type::butterfly_forward(x0, x2, m_inv, m_inv_precomp, m_inv,
                                         m_inv_precomp);
          modmul_type::butterfly_forward(x1, x3);

          interleave_and_store<2>(&dst[j], x0, x1, x2, x3);
        }
      }
    } else if constexpr (n == 16) {
      svuint64_t omega_i, omega_i_precomp, omega_i_n4, omega_i_n4_precomp,
          omega_2i, omega_2i_precomp;
      if constexpr (cntd == 8) {
        omega_i = svld1(ptrue, aux);
        aux += cntd;
        omega_i_precomp = svzip2(omega_i, omega_i);
        omega_i = svzip1(omega_i, omega_i);

        omega_i_n4 = svld1(ptrue, aux);
        aux += cntd;
        omega_i_n4_precomp = svzip2(omega_i_n4, omega_i_n4);
        omega_i_n4 = svzip1(omega_i_n4, omega_i_n4);

        omega_2i = svld1(ptrue, aux);
        aux += cntd;
        omega_2i_precomp = svzip2(omega_2i, omega_2i);
        omega_2i = svzip1(omega_2i, omega_2i);
      } else {
        throw std::invalid_argument{"Unsupported SVE vector length"};
      }

      for (std::uint64_t j{}; j < m; j += cntd * 4) {
        svuint64_t x0, x1, x2, x3;
        svuint64_t y0, y1, y2, y3;

        /*
         * 0:  0  1  2  3  4  5  6  7
         * 1:  8  9 10 11 12 13 14 15
         * 2: 16 17 18 19 20 21 22 23
         * 3: 24 25 26 27 28 29 30 31
         */
        x0 = svld1_vnum(ptrue, &src[j], 0);
        x1 = svld1_vnum(ptrue, &src[j], 1);
        x2 = svld1_vnum(ptrue, &src[j], 2);
        x3 = svld1_vnum(ptrue, &src[j], 3);

        /*
         * 0:  0 16  1 17  2 18  3 19
         * 1:  8 24  9 25 10 26 11 27
         * 2:  4 20  5 21  6 22  7 23
         * 3: 12 28 13 29 14 30 15 31
         */
        y0 = svzip1(x0, x2);
        y1 = svzip1(x1, x3);
        y2 = svzip2(x0, x2);
        y3 = svzip2(x1, x3);

        modmul_type::butterfly_inverse(y0, y2, omega_2i, omega_2i_precomp);
        modmul_type::butterfly_inverse(y1, y3, omega_2i, omega_2i_precomp);

        modmul_type::butterfly_inverse(y0, y1, omega_i, omega_i_precomp);
        modmul_type::butterfly_inverse(y2, y3, omega_i_n4, omega_i_n4_precomp);

        /*
         * 0:  0  1  2  3  4  5  6  7
         * 1:  8  9 10 11 12 13 14 15
         * 2: 16 17 18 19 20 21 22 23
         * 3: 24 25 26 27 28 29 30 31
         */
        x0 = svuzp1(y0, y2);
        x1 = svuzp1(y1, y3);
        x2 = svuzp2(y0, y2);
        x3 = svuzp2(y1, y3);

        svst1_vnum(ptrue, &dst[j], 0, x0);
        svst1_vnum(ptrue, &dst[j], 1, x1);
        svst1_vnum(ptrue, &dst[j], 2, x2);
        svst1_vnum(ptrue, &dst[j], 3, x3);
      }
    } else {
      throw std::invalid_argument{"Unsupported SVE vector length"};
    }

    aux_arg = reinterpret_cast<const std::byte *>(aux);
  }

  static void compute_inverse(std::uint64_t *const dst, const std::byte *&aux) {
    compute_inverse(dst, dst, aux);
  }
};

} // namespace sventt

#endif /* SVENTT_LAYER_SVE_RADIX_FOUR_HPP_INCLUDED */

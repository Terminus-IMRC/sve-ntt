// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_LAYER_SVE_RADIX_EIGHT_HPP_INCLUDED
#define SVENTT_LAYER_SVE_RADIX_EIGHT_HPP_INCLUDED

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <arm_sve.h>

#include "sventt/common/sve.hpp"

namespace sventt {

template <class modmul_type_, std::uint64_t m, std::uint64_t n,
          std::uint64_t inverse_factor = 1, bool store_precomputation = true>
class RadixEightSVELayer {

public:
  using modmul_type = modmul_type_;
  using modulus_type = modmul_type::modulus_type;

  static constexpr std::uint64_t get_radix(void) { return 8; }

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

    const std::uint64_t omega81{modulus_type::get_root_forward(8)},
        omega82{modulus_type::get_root_forward(4)},
        omega83{modulus_type::multiply(omega81, omega82)};
    if (n / 8 >= cntd) {
      if (!store_precomputation) {
        for (std::uint64_t omega : {
                 modulus_type::get_root_forward(n / cntd),
                 modulus_type::get_root_forward(n / 2 / cntd),
                 modulus_type::get_root_forward(n / 4 / cntd),
             }) {
          precompute(1, omega, {}, true);
        }
      }

      const std::uint64_t omega_n_1{modulus_type::get_root_forward(n)},
          omega_n_2{modulus_type::get_root_forward(n / 2)},
          omega_n_4{modulus_type::get_root_forward(n / 4)};
      std::uint64_t omega_n_1i_n80{1}, omega_n_1i_n81{omega81},
          omega_n_1i_n82{omega82}, omega_n_1i_n83{omega83}, omega_n_2i_n40{1},
          omega_n_2i_n41{omega82}, omega_n_4i_n20{1};
      for (std::uint64_t i{}; i < n / 8; i += cntd) {
        if (i == 0 || store_precomputation) {
          precompute(cntd, omega_n_1i_n80, omega_n_1, store_precomputation);
          precompute(cntd, omega_n_1i_n81, omega_n_1, store_precomputation);
          precompute(cntd, omega_n_1i_n82, omega_n_1, store_precomputation);
          precompute(cntd, omega_n_1i_n83, omega_n_1, store_precomputation);
          precompute(cntd, omega_n_2i_n40, omega_n_2, store_precomputation);
          precompute(cntd, omega_n_2i_n41, omega_n_2, store_precomputation);
          precompute(cntd, omega_n_4i_n20, omega_n_4, store_precomputation);
        }
      }
    } else if (n == 8) {
      if (m < cntd * 8) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      std::uint64_t omega{omega81};
      precompute(1, omega, omega81, true);
      precompute(1, omega, omega81, true);
      precompute(1, omega, {}, true);
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

    if constexpr (n / 8 >= cntd) {
      svuint64_t omega_n_cntd, omega_n_cntd_precomp, omega_n_2cntd,
          omega_n_2cntd_precomp, omega_n_4cntd, omega_n_4cntd_precomp;
      if constexpr (!store_precomputation) {
        broadcast_and_advance(omega_n_cntd, omega_n_cntd_precomp, omega_n_2cntd,
                              omega_n_2cntd_precomp, omega_n_4cntd,
                              omega_n_4cntd_precomp, aux);
      }

      const std::uint64_t *const aux_orig{aux};
      for (std::uint64_t j{}; j < m; j += n) {
        aux = aux_orig;

        svuint64_t omega_n_1i_n80, omega_n_1i_n80_precomp, omega_n_1i_n81,
            omega_n_1i_n81_precomp, omega_n_1i_n82, omega_n_1i_n82_precomp,
            omega_n_1i_n83, omega_n_1i_n83_precomp, omega_n_2i_n40,
            omega_n_2i_n40_precomp, omega_n_2i_n41, omega_n_2i_n41_precomp,
            omega_n_4i_n20, omega_n_4i_n20_precomp;
        for (std::uint64_t i{}; i < n / 8; i += cntd) {
          svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
          x0 = svld1(ptrue, &src[i + j + n / 8 * 0]);
          x1 = svld1(ptrue, &src[i + j + n / 8 * 1]);
          x2 = svld1(ptrue, &src[i + j + n / 8 * 2]);
          x3 = svld1(ptrue, &src[i + j + n / 8 * 3]);
          x4 = svld1(ptrue, &src[i + j + n / 8 * 4]);
          x5 = svld1(ptrue, &src[i + j + n / 8 * 5]);
          x6 = svld1(ptrue, &src[i + j + n / 8 * 6]);
          x7 = svld1(ptrue, &src[i + j + n / 8 * 7]);

          load_or_precompute(omega_n_1i_n80, omega_n_1i_n80_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);
          load_or_precompute(omega_n_1i_n81, omega_n_1i_n81_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);
          load_or_precompute(omega_n_1i_n82, omega_n_1i_n82_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);
          load_or_precompute(omega_n_1i_n83, omega_n_1i_n83_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);

          modmul_type::butterfly_forward(x0, x4, omega_n_1i_n80,
                                         omega_n_1i_n80_precomp);
          modmul_type::butterfly_forward(x1, x5, omega_n_1i_n81,
                                         omega_n_1i_n81_precomp);
          modmul_type::butterfly_forward(x2, x6, omega_n_1i_n82,
                                         omega_n_1i_n82_precomp);
          modmul_type::butterfly_forward(x3, x7, omega_n_1i_n83,
                                         omega_n_1i_n83_precomp);

          load_or_precompute(omega_n_2i_n40, omega_n_2i_n40_precomp, i == 0,
                             omega_n_2cntd, omega_n_2cntd_precomp);
          load_or_precompute(omega_n_2i_n41, omega_n_2i_n41_precomp, i == 0,
                             omega_n_2cntd, omega_n_2cntd_precomp);

          modmul_type::butterfly_forward(x0, x2, omega_n_2i_n40,
                                         omega_n_2i_n40_precomp);
          modmul_type::butterfly_forward(x1, x3, omega_n_2i_n41,
                                         omega_n_2i_n41_precomp);
          modmul_type::butterfly_forward(x4, x6, omega_n_2i_n40,
                                         omega_n_2i_n40_precomp);
          modmul_type::butterfly_forward(x5, x7, omega_n_2i_n41,
                                         omega_n_2i_n41_precomp);

          load_or_precompute(omega_n_4i_n20, omega_n_4i_n20_precomp, i == 0,
                             omega_n_4cntd, omega_n_4cntd_precomp);

          modmul_type::butterfly_forward(x0, x1, omega_n_4i_n20,
                                         omega_n_4i_n20_precomp);
          modmul_type::butterfly_forward(x2, x3, omega_n_4i_n20,
                                         omega_n_4i_n20_precomp);
          modmul_type::butterfly_forward(x4, x5, omega_n_4i_n20,
                                         omega_n_4i_n20_precomp);
          modmul_type::butterfly_forward(x6, x7, omega_n_4i_n20,
                                         omega_n_4i_n20_precomp);

          svst1(ptrue, &dst[i + j + n / 8 * 0], x0);
          svst1(ptrue, &dst[i + j + n / 8 * 1], x1);
          svst1(ptrue, &dst[i + j + n / 8 * 2], x2);
          svst1(ptrue, &dst[i + j + n / 8 * 3], x3);
          svst1(ptrue, &dst[i + j + n / 8 * 4], x4);
          svst1(ptrue, &dst[i + j + n / 8 * 5], x5);
          svst1(ptrue, &dst[i + j + n / 8 * 6], x6);
          svst1(ptrue, &dst[i + j + n / 8 * 7], x7);
        }
      }
    } else if constexpr (n == 8) {
      svuint64_t omega81, omega81_precomp, omega82, omega82_precomp, omega83,
          omega83_precomp;
      broadcast_and_advance(omega81, omega81_precomp, omega82, omega82_precomp,
                            omega83, omega83_precomp, aux);

      for (std::uint64_t j{}; j < m; j += cntd * 8) {
        svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
        load_and_deinterleave<2>(x0, x1, x2, x3, x4, x5, x6, x7, &src[j]);

        modmul_type::butterfly_forward(x0, x4);
        modmul_type::butterfly_forward(x1, x5, omega81, omega81_precomp);
        modmul_type::butterfly_forward(x2, x6, omega82, omega82_precomp);
        modmul_type::butterfly_forward(x3, x7, omega83, omega83_precomp);

        modmul_type::butterfly_forward(x0, x2);
        modmul_type::butterfly_forward(x1, x3, omega82, omega82_precomp);
        modmul_type::butterfly_forward(x4, x6);
        modmul_type::butterfly_forward(x5, x7, omega82, omega82_precomp);

        modmul_type::butterfly_forward(x0, x1);
        modmul_type::butterfly_forward(x2, x3);
        modmul_type::butterfly_forward(x4, x5);
        modmul_type::butterfly_forward(x6, x7);

        interleave_and_store<3>(&dst[j], x0, x1, x2, x3, x4, x5, x6, x7);
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

    const std::uint64_t omega81{modulus_type::get_root_inverse(8)},
        omega82{modulus_type::get_root_inverse(4)},
        omega83{modulus_type::multiply(omega81, omega82)};
    if (n / 8 >= cntd) {
      if (!store_precomputation) {
        for (std::uint64_t omega : {
                 modulus_type::get_root_inverse(n / 4 / cntd),
                 modulus_type::get_root_inverse(n / 2 / cntd),
                 modulus_type::get_root_inverse(n / cntd),
             }) {
          precompute(1, omega, {}, true);
        }
      }

      const std::uint64_t omega_n_1{modulus_type::get_root_inverse(n)},
          omega_n_2{modulus_type::get_root_inverse(n / 2)},
          omega_n_4{modulus_type::get_root_inverse(n / 4)};
      std::uint64_t omega_n_1i_n80{1}, omega_n_1i_n81{omega81},
          omega_n_1i_n82{omega82}, omega_n_1i_n83{omega83}, omega_n_2i_n40{1},
          omega_n_2i_n41{omega82}, omega_n_4i_n20{1};
      for (std::uint64_t i{}; i < n / 8; i += cntd) {
        if (i == 0 || store_precomputation) {
          precompute(cntd, omega_n_4i_n20, omega_n_4, store_precomputation);
          precompute(cntd, omega_n_2i_n40, omega_n_2, store_precomputation);
          precompute(cntd, omega_n_2i_n41, omega_n_2, store_precomputation);
          precompute(cntd, omega_n_1i_n80, omega_n_1, store_precomputation);
          precompute(cntd, omega_n_1i_n81, omega_n_1, store_precomputation);
          precompute(cntd, omega_n_1i_n82, omega_n_1, store_precomputation);
          precompute(cntd, omega_n_1i_n83, omega_n_1, store_precomputation);
        }
      }
    } else if (n == 8) {
      if (m < cntd * 8) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      if (inverse_factor == 1) {
        for (std::uint64_t omega : {omega81, omega82, omega83}) {
          precompute(1, omega, {}, true);
        }
      } else {
        const std::uint64_t m_inv{modulus_type::invert(inverse_factor)};
        for (std::uint64_t omega :
             {m_inv, omega81, modulus_type::multiply(omega82, m_inv),
              omega83}) {
          precompute(1, omega, {}, true);
        }
      }
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

    if constexpr (n / 8 >= cntd) {
      svuint64_t omega_n_cntd, omega_n_cntd_precomp, omega_n_2cntd,
          omega_n_2cntd_precomp, omega_n_4cntd, omega_n_4cntd_precomp;
      if constexpr (!store_precomputation) {
        broadcast_and_advance(omega_n_4cntd, omega_n_4cntd_precomp,
                              omega_n_2cntd, omega_n_2cntd_precomp,
                              omega_n_cntd, omega_n_cntd_precomp, aux);
      }

      const std::uint64_t *const aux_orig{aux};
      for (std::uint64_t j{}; j < m; j += n) {
        aux = aux_orig;

        svuint64_t omega_n_1i_n80, omega_n_1i_n80_precomp, omega_n_1i_n81,
            omega_n_1i_n81_precomp, omega_n_1i_n82, omega_n_1i_n82_precomp,
            omega_n_1i_n83, omega_n_1i_n83_precomp, omega_n_2i_n40,
            omega_n_2i_n40_precomp, omega_n_2i_n41, omega_n_2i_n41_precomp,
            omega_n_4i_n20, omega_n_4i_n20_precomp;
        for (std::uint64_t i{}; i < n / 8; i += cntd) {
          svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
          x0 = svld1(ptrue, &src[i + j + n / 8 * 0]);
          x1 = svld1(ptrue, &src[i + j + n / 8 * 1]);
          x2 = svld1(ptrue, &src[i + j + n / 8 * 2]);
          x3 = svld1(ptrue, &src[i + j + n / 8 * 3]);
          x4 = svld1(ptrue, &src[i + j + n / 8 * 4]);
          x5 = svld1(ptrue, &src[i + j + n / 8 * 5]);
          x6 = svld1(ptrue, &src[i + j + n / 8 * 6]);
          x7 = svld1(ptrue, &src[i + j + n / 8 * 7]);

          load_or_precompute(omega_n_4i_n20, omega_n_4i_n20_precomp, i == 0,
                             omega_n_4cntd, omega_n_4cntd_precomp);

          modmul_type::butterfly_inverse(x0, x1, omega_n_4i_n20,
                                         omega_n_4i_n20_precomp);
          modmul_type::butterfly_inverse(x2, x3, omega_n_4i_n20,
                                         omega_n_4i_n20_precomp);
          modmul_type::butterfly_inverse(x4, x5, omega_n_4i_n20,
                                         omega_n_4i_n20_precomp);
          modmul_type::butterfly_inverse(x6, x7, omega_n_4i_n20,
                                         omega_n_4i_n20_precomp);

          load_or_precompute(omega_n_2i_n40, omega_n_2i_n40_precomp, i == 0,
                             omega_n_2cntd, omega_n_2cntd_precomp);
          load_or_precompute(omega_n_2i_n41, omega_n_2i_n41_precomp, i == 0,
                             omega_n_2cntd, omega_n_2cntd_precomp);

          modmul_type::butterfly_inverse(x0, x2, omega_n_2i_n40,
                                         omega_n_2i_n40_precomp);
          modmul_type::butterfly_inverse(x1, x3, omega_n_2i_n41,
                                         omega_n_2i_n41_precomp);
          modmul_type::butterfly_inverse(x4, x6, omega_n_2i_n40,
                                         omega_n_2i_n40_precomp);
          modmul_type::butterfly_inverse(x5, x7, omega_n_2i_n41,
                                         omega_n_2i_n41_precomp);

          load_or_precompute(omega_n_1i_n80, omega_n_1i_n80_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);
          load_or_precompute(omega_n_1i_n81, omega_n_1i_n81_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);
          load_or_precompute(omega_n_1i_n82, omega_n_1i_n82_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);
          load_or_precompute(omega_n_1i_n83, omega_n_1i_n83_precomp, i == 0,
                             omega_n_cntd, omega_n_cntd_precomp);

          modmul_type::butterfly_inverse(x0, x4, omega_n_1i_n80,
                                         omega_n_1i_n80_precomp);
          modmul_type::butterfly_inverse(x1, x5, omega_n_1i_n81,
                                         omega_n_1i_n81_precomp);
          modmul_type::butterfly_inverse(x2, x6, omega_n_1i_n82,
                                         omega_n_1i_n82_precomp);
          modmul_type::butterfly_inverse(x3, x7, omega_n_1i_n83,
                                         omega_n_1i_n83_precomp);

          svst1(ptrue, &dst[i + j + n / 8 * 0], x0);
          svst1(ptrue, &dst[i + j + n / 8 * 1], x1);
          svst1(ptrue, &dst[i + j + n / 8 * 2], x2);
          svst1(ptrue, &dst[i + j + n / 8 * 3], x3);
          svst1(ptrue, &dst[i + j + n / 8 * 4], x4);
          svst1(ptrue, &dst[i + j + n / 8 * 5], x5);
          svst1(ptrue, &dst[i + j + n / 8 * 6], x6);
          svst1(ptrue, &dst[i + j + n / 8 * 7], x7);
        }
      }
    } else if constexpr (n == 8) {
      if constexpr (inverse_factor == 1) {
        svuint64_t omega81, omega81_precomp, omega82, omega82_precomp, omega83,
            omega83_precomp;
        broadcast_and_advance(omega81, omega81_precomp, omega82,
                              omega82_precomp, omega83, omega83_precomp, aux);

        for (std::uint64_t j{}; j < m; j += cntd * 8) {
          svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
          load_and_deinterleave<2>(x0, x1, x2, x3, x4, x5, x6, x7, &src[j]);

          modmul_type::butterfly_forward(x0, x1);
          modmul_type::butterfly_forward(x2, x3, omega82, omega82_precomp);
          modmul_type::butterfly_forward(x4, x5);
          modmul_type::butterfly_forward(x6, x7, omega82, omega82_precomp);

          modmul_type::butterfly_forward(x0, x2);
          modmul_type::butterfly_forward(x1, x3);
          modmul_type::butterfly_forward(x4, x6, omega82, omega82_precomp);
          modmul_type::butterfly_forward(x5, x7, omega81, omega81_precomp,
                                         omega83, omega83_precomp);

          modmul_type::butterfly_forward(x0, x4);
          modmul_type::butterfly_forward(x1, x5, omega81, omega81_precomp);
          modmul_type::butterfly_forward(x2, x6, omega82, omega82_precomp);
          modmul_type::butterfly_forward(x3, x7, omega83, omega83_precomp);

          interleave_and_store<3>(&dst[j], x0, x1, x2, x3, x4, x5, x6, x7);
        }
      } else {
        svuint64_t m_inv, m_inv_precomp, omega81, omega81_precomp,
            omega82_m_inv, omega82_m_inv_precomp, omega83, omega83_precomp;
        broadcast_and_advance(m_inv, m_inv_precomp, omega81, omega81_precomp,
                              omega82_m_inv, omega82_m_inv_precomp, omega83,
                              omega83_precomp, aux);

        for (std::uint64_t j{}; j < m; j += cntd * 8) {
          svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
          load_and_deinterleave<2>(x0, x1, x2, x3, x4, x5, x6, x7, &src[j]);

          modmul_type::butterfly_forward(x0, x1, m_inv, m_inv_precomp);
          modmul_type::butterfly_forward(x2, x3, omega82_m_inv,
                                         omega82_m_inv_precomp);
          modmul_type::butterfly_forward(x4, x5, m_inv, m_inv_precomp);
          modmul_type::butterfly_forward(x6, x7, omega82_m_inv,
                                         omega82_m_inv_precomp);

          modmul_type::butterfly_forward(x0, x2, m_inv, m_inv_precomp, m_inv,
                                         m_inv_precomp);
          modmul_type::butterfly_forward(x1, x3);
          modmul_type::butterfly_forward(x4, x6, m_inv, m_inv_precomp,
                                         omega82_m_inv, omega82_m_inv_precomp);
          modmul_type::butterfly_forward(x5, x7, omega81, omega81_precomp,
                                         omega83, omega83_precomp);

          modmul_type::butterfly_forward(x0, x4);
          modmul_type::butterfly_forward(x1, x5);
          modmul_type::butterfly_forward(x2, x6);
          modmul_type::butterfly_forward(x3, x7);

          interleave_and_store<3>(&dst[j], x0, x1, x2, x3, x4, x5, x6, x7);
        }
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

#endif /* SVENTT_LAYER_SVE_RADIX_EIGHT_HPP_INCLUDED */

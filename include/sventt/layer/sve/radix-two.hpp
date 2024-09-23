// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_LAYER_SVE_RADIX_TWO_HPP_INCLUDED
#define SVENTT_LAYER_SVE_RADIX_TWO_HPP_INCLUDED

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <arm_sve.h>

#include "sventt/common/sve.hpp"

namespace sventt {

template <class modmul_type_, std::uint64_t m, std::uint64_t n,
          std::uint64_t inverse_factor = 1, bool store_precomputation = true>
class RadixTwoSVELayer {

public:
  using modmul_type = modmul_type_;
  using modulus_type = modmul_type::modulus_type;

  static constexpr std::uint64_t get_radix(void) { return 2; }

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

    if (n / 2 >= cntd) {
      if (!store_precomputation) {
        for (std::uint64_t omega : {modulus_type::get_root_forward(n / cntd)}) {
          precompute(1, omega, {}, true);
        }
      }

      const std::uint64_t omega_n{modulus_type::get_root_forward(n)};
      std::uint64_t omega_n_i{1};
      for (std::uint64_t i{}; i < n / 2; i += cntd) {
        if (i == 0 || store_precomputation) {
          precompute(cntd, omega_n_i, omega_n, store_precomputation);
        }
      }
    } else if (n == 2) {
      if (m < cntd * 2) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
    } else if (n == 4) {
      if (m < cntd * 4) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      const std::uint64_t omega_n{modulus_type::get_root_forward(n)};
      std::uint64_t omega_n_i{omega_n};
      precompute(1, omega_n_i, {}, true);
    } else if (n == 8) {
      if (m < cntd * 8) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      const std::uint64_t omega_n{modulus_type::get_root_forward(n)};
      std::uint64_t omega_n_i{omega_n};
      precompute(1, omega_n_i, omega_n, true);
      precompute(1, omega_n_i, omega_n, true);
      precompute(1, omega_n_i, {}, true);
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

    /* If the stride of butterflies is large enough for vector length... */
    if constexpr (n / 2 >= cntd) {
      svuint64_t omega_n_cntd, omega_n_cntd_precomp;
      if constexpr (!store_precomputation) {
        broadcast_and_advance(omega_n_cntd, omega_n_cntd_precomp, aux);
      }

      const std::uint64_t *const aux_orig{aux};
      for (std::uint64_t j{}; j < m; j += n) {
        aux = aux_orig;

        svuint64_t omega_n_i, omega_n_i_precomp;
        for (std::uint64_t i{}; i < n / 2; i += cntd) {
          svuint64_t x0, x1;
          x0 = svld1(ptrue, &src[i + j + n / 2 * 0]);
          x1 = svld1(ptrue, &src[i + j + n / 2 * 1]);

          load_or_precompute(omega_n_i, omega_n_i_precomp, i == 0, omega_n_cntd,
                             omega_n_cntd_precomp);

          modmul_type::butterfly_forward(x0, x1, omega_n_i, omega_n_i_precomp);

          svst1(ptrue, &dst[i + j + n / 2 * 0], x0);
          svst1(ptrue, &dst[i + j + n / 2 * 1], x1);
        }
      }
    } else if constexpr (n == 2) {
      for (std::uint64_t j{}; j < m; j += cntd * 2) {
        svuint64_t x0, x1;
        load_and_deinterleave<0>(x0, x1, &src[j]);

        modmul_type::butterfly_forward(x0, x1);

        interleave_and_store<1>(&dst[j], x0, x1);
      }
    } else if constexpr (n == 4) {
      svuint64_t omega_n_1, omega_n_1_precomp;
      broadcast_and_advance(omega_n_1, omega_n_1_precomp, aux);

      for (std::uint64_t j{}; j < m; j += cntd * 4) {
        svuint64_t x0, x1, x2, x3;
        load_and_deinterleave<1>(x0, x1, x2, x3, &src[j]);

        modmul_type::butterfly_forward(x0, x2);
        modmul_type::butterfly_forward(x1, x3, omega_n_1, omega_n_1_precomp);

        interleave_and_store<2>(&dst[j], x0, x1, x2, x3);
      }
    } else if constexpr (n == 8) {
      svuint64_t omega_n_1, omega_n_1_precomp, omega_n_2, omega_n_2_precomp,
          omega_n_3, omega_n_3_precomp;
      broadcast_and_advance(omega_n_1, omega_n_1_precomp, omega_n_2,
                            omega_n_2_precomp, omega_n_3, omega_n_3_precomp,
                            aux);

      for (std::uint64_t j{}; j < m; j += cntd * 8) {
        svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
        load_and_deinterleave<1>(x0, x1, x2, x3, x4, x5, x6, x7, &src[j]);

        modmul_type::butterfly_forward(x0, x4);
        modmul_type::butterfly_forward(x1, x5, omega_n_1, omega_n_1_precomp);
        modmul_type::butterfly_forward(x2, x6, omega_n_2, omega_n_2_precomp);
        modmul_type::butterfly_forward(x3, x7, omega_n_3, omega_n_3_precomp);

        interleave_and_store<2>(&dst[j], x0, x1, x2, x3, x4, x5, x6, x7);
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

    if (n != get_radix() && inverse_factor != 1) {
      throw std::invalid_argument{"Multiplying by inverse factor in "
                                  "intermediate layer is not supported"};
    }
    if (n / 2 >= cntd) {
      if (!store_precomputation) {
        for (std::uint64_t omega : {
                 modulus_type::get_root_inverse(n / cntd),
             }) {
          precompute(1, omega, {}, true);
        }
      }

      const std::uint64_t omega_n{modulus_type::get_root_inverse(n)};
      std::uint64_t omega_n_i{1};
      for (std::uint64_t i{}; i < n / 2; i += cntd) {
        if (i == 0 || store_precomputation) {
          precompute(cntd, omega_n_i, omega_n, store_precomputation);
        }
      }
    } else if (n == 2) {
      if (m < cntd * 2) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      if constexpr (inverse_factor != 1) {
        std::uint64_t m_inv{modulus_type::invert(inverse_factor)};
        precompute(1, m_inv, {}, true);
      }
    } else if (n == 4) {
      if (m < cntd * 4) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      const std::uint64_t omega_n{modulus_type::get_root_inverse(n)};
      std::uint64_t omega_n_i{omega_n};
      precompute(1, omega_n_i, {}, true);
    } else if (n == 8) {
      if (m < cntd * 8) {
        throw std::invalid_argument{"SVE vector length is too large"};
      }
      const std::uint64_t omega_n{modulus_type::get_root_inverse(n)};
      std::uint64_t omega_n_i{omega_n};
      precompute(1, omega_n_i, omega_n, true);
      precompute(1, omega_n_i, omega_n, true);
      precompute(1, omega_n_i, {}, true);
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

    if constexpr (n / 2 >= cntd) {
      svuint64_t omega_n_cntd, omega_n_cntd_precomp;
      if constexpr (!store_precomputation) {
        broadcast_and_advance(omega_n_cntd, omega_n_cntd_precomp, aux);
      }

      const std::uint64_t *const aux_orig{aux};
      for (std::uint64_t j{}; j < m; j += n) {
        aux = aux_orig;

        svuint64_t omega_n_i, omega_n_i_precomp;
        for (std::uint64_t i{}; i < n / 2; i += cntd) {
          svuint64_t x0, x1;
          x0 = svld1(ptrue, &src[i + j + n / 2 * 0]);
          x1 = svld1(ptrue, &src[i + j + n / 2 * 1]);

          load_or_precompute(omega_n_i, omega_n_i_precomp, i == 0, omega_n_cntd,
                             omega_n_cntd_precomp);

          modmul_type::butterfly_inverse(x0, x1, omega_n_i, omega_n_i_precomp);

          svst1(ptrue, &dst[i + j + n / 2 * 0], x0);
          svst1(ptrue, &dst[i + j + n / 2 * 1], x1);
        }
      }
    } else if constexpr (n == 2) {
      if constexpr (inverse_factor == 1) {
        for (std::uint64_t j{}; j < m; j += cntd * 2) {
          svuint64_t x0, x1;
          load_and_deinterleave<0>(x0, x1, &src[j]);

          modmul_type::butterfly_inverse(x0, x1);

          interleave_and_store<1>(&dst[j], x0, x1);
        }
      } else {
        svuint64_t m_inv, m_inv_precomp;
        broadcast_and_advance(m_inv, m_inv_precomp, aux);

        for (std::uint64_t j{}; j < m; j += cntd * 2) {
          svuint64_t x0, x1;
          load_and_deinterleave<0>(x0, x1, &src[j]);

          modmul_type::butterfly_forward(x0, x1, m_inv, m_inv_precomp, m_inv,
                                         m_inv_precomp);

          interleave_and_store<1>(&dst[j], x0, x1);
        }
      }
    } else if constexpr (n == 4) {
      svuint64_t omega_n_1, omega_n_1_precomp;
      broadcast_and_advance(omega_n_1, omega_n_1_precomp, aux);

      for (std::uint64_t j{}; j < m; j += cntd * 4) {
        svuint64_t x0, x1, x2, x3;
        load_and_deinterleave<1>(x0, x1, x2, x3, &src[j]);

        modmul_type::butterfly_inverse(x0, x2);
        modmul_type::butterfly_inverse(x1, x3, omega_n_1, omega_n_1_precomp);

        interleave_and_store<2>(&dst[j], x0, x1, x2, x3);
      }
    } else if constexpr (n == 8) {
      svuint64_t omega_n_1, omega_n_1_precomp, omega_n_2, omega_n_2_precomp,
          omega_n_3, omega_n_3_precomp;
      broadcast_and_advance(omega_n_1, omega_n_1_precomp, omega_n_2,
                            omega_n_2_precomp, omega_n_3, omega_n_3_precomp,
                            aux);

      for (std::uint64_t j{}; j < m; j += cntd * 8) {
        svuint64_t x0, x1, x2, x3, x4, x5, x6, x7;
        load_and_deinterleave<1>(x0, x1, x2, x3, x4, x5, x6, x7, &src[j]);

        modmul_type::butterfly_inverse(x0, x4);
        modmul_type::butterfly_inverse(x1, x5, omega_n_1, omega_n_1_precomp);
        modmul_type::butterfly_inverse(x2, x6, omega_n_2, omega_n_2_precomp);
        modmul_type::butterfly_inverse(x3, x7, omega_n_3, omega_n_3_precomp);

        interleave_and_store<2>(&dst[j], x0, x1, x2, x3, x4, x5, x6, x7);
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

#endif /* SVENTT_LAYER_SVE_RADIX_TWO_HPP_INCLUDED */

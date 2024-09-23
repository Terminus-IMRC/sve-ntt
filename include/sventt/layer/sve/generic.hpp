// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_LAYER_SVE_GENERIC_HPP_INCLUDED
#define SVENTT_LAYER_SVE_GENERIC_HPP_INCLUDED

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <arm_sve.h>

#include "sventt/common/sve.hpp"
#include "sventt/vector.hpp"

namespace sventt {

template <class modmul_type_, std::uint64_t m, class inner_kernel_type_,
          std::uint64_t buffer_padding_elements,
          std::uint64_t twiddle_unroll_count, class transposition_type,
          bool transpose_in_place = false>
class GenericSVELayer {

public:
  using modmul_type = modmul_type_;
  using modulus_type = modmul_type::modulus_type;
  using inner_kernel_type = inner_kernel_type_;

  static_assert(m % inner_kernel_type::get_m() == 0);
  static_assert(!transpose_in_place ||
                m / inner_kernel_type::get_m() == inner_kernel_type::get_m());

  static constexpr std::uint64_t get_m(void) { return m; }

  static constexpr std::uint64_t get_radix(void) {
    return inner_kernel_type::get_m();
  }

  class buffer_type {

    PageMemory<std::uint64_t> buffer{
        transpose_in_place
            ? 0
            : ((inner_kernel_type::get_m() + buffer_padding_elements) *
                   (m / inner_kernel_type::get_m()) -
               buffer_padding_elements),
        m >= (std::uint64_t{1} << 18)};

  public:
    decltype(buffer) &get(void) { return buffer; }

    void touch_pages_cyclically(void) {
      const std::uint64_t page_size{(m >= (std::uint64_t{1} << 18))
                                        ? (std::uint64_t{1} << 21)
                                        : (std::uint64_t{1} << 16)};

      /* TODO: Take the number of NUMA domains as an argument. */
      [[omp::directive(parallel)]] {
        sventt::touch_pages_cyclically(
            buffer.data(),
            sizeof(typename decltype(buffer)::value_type) * buffer.size(),
            page_size, 4, 12, omp_get_thread_num() / 12,
            omp_get_thread_num() % 12);
      }
    }
  };

  template <class vector_type>
  static void prepare_forward(vector_type &aux) {
    const auto precompute{
        [&aux](const std::uint64_t num_elements, std::uint64_t &omega,
               const std::uint64_t factor, const bool precompute_precompute) {
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

    /* aux size, to be corrected later */
    aux.template push_back<std::uint64_t>({});
    const typename vector_type::size_type aux_pos_after_size{aux.size()};

    inner_kernel_type::prepare_forward(aux);

    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    constexpr std::uint64_t omega_m{modulus_type::get_root_forward(m)};
    for (std::uint64_t j{}; j < inner_m; ++j) {
      const std::uint64_t omega_j{modulus_type::power(
          omega_m, bitreverse(j) >> (65 - std::bit_width(inner_m)))};
      std::uint64_t omega_ij;
      omega_ij = modulus_type::power(omega_j, cntd);
      precompute(1, omega_ij, {}, false);
      omega_ij = omega_j;
      precompute(cntd - 1, omega_ij, omega_j, false);
    }

    aux.template reinterpret_at<std::uint64_t>(aux_pos_after_size -
                                      sizeof(std::uint64_t)) =
        aux.size() - aux_pos_after_size;
  }

  static void compute_forward_without_twiddle(std::uint64_t *const dst,
                                              const std::uint64_t *const src,
                                              const std::byte *&aux_arg,
                                              buffer_type &buffer_arg) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};

    const std::byte *aux;
    {
      const std::uint64_t aux_size{
          pointer_utility::get_and_advance<std::uint64_t>(aux_arg)};
      aux = aux_arg;
      /* Note: This includes the size of twiddles used in twiddle_rows. */
      aux_arg += aux_size;
    }

    auto &buffer{buffer_arg.get()};

    if constexpr (transpose_in_place) {
      if (src == dst) {
        transposition_type::transpose(dst, inner_m);
      } else {
        transposition_type::transpose(dst, src, inner_m, m / inner_m, inner_m,
                                      m / inner_m);
      }
    } else {
      transposition_type::transpose(&buffer[0], src, inner_m, m / inner_m,
                                    inner_m + buffer_padding_elements,
                                    m / inner_m);
    }

    const std::byte *const aux_orig{aux};
    [[omp::directive(for)]]
    for (std::uint64_t i = 0; i < m / inner_m; ++i) {
      aux = aux_orig;
      if constexpr (transpose_in_place) {
        inner_kernel_type::compute_forward(&dst[inner_m * i], aux);
      } else {
        inner_kernel_type::compute_forward(
            &buffer[(inner_m + buffer_padding_elements) * i], aux);
      }
    }

    if constexpr (transpose_in_place) {
      transposition_type::transpose(dst, inner_m);
    } else {
      transposition_type::transpose(dst, &buffer[0], m / inner_m, inner_m,
                                    m / inner_m,
                                    inner_m + buffer_padding_elements);
    }
  }

  static void compute_forward_without_twiddle(std::uint64_t *const dst,
                                              const std::byte *&aux,
                                              buffer_type &buffer) {
    compute_forward_without_twiddle(dst, dst, aux, buffer);
  }

  static void twiddle_rows_forward(std::uint64_t *const dst,
                                   const std::uint64_t *const src,
                                   const std::byte *const aux_arg,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride_dst,
                                   const std::uint64_t stride_src) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    constexpr std::uint64_t unity{modmul_type::to_montgomery(1)};
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    const svbool_t ptrue{svptrue_b8()};
    const svbool_t pfirst{svptrue_pat_b8(SV_VL1)};
    const std::uint64_t *aux{reinterpret_cast<const std::uint64_t *>(aux_arg) -
                             cntd * inner_m};

    for (std::uint64_t j{j_start}; j < j_end; ++j) {
      if constexpr (twiddle_unroll_count == 1) {
        svuint64_t omega_ij{svld1(ptrue, &aux[cntd * j])};
        const svuint64_t omega_j_cntd{svdup_lane(omega_ij, 0)};
        const svuint64_t omega_j_cntd_precomp{
            modmul_type::precompute(omega_j_cntd)};
        omega_ij = svdup_u64_m(omega_ij, pfirst, unity);
        for (std::uint64_t i{}; i < m / inner_m; i += cntd) {
          svuint64_t t;
          t = svld1(ptrue, &src[stride_src * j + i]);
          t = modmul_type::multiply(t, omega_ij);
          svst1(ptrue, &dst[stride_dst * j + i], t);
          omega_ij = modmul_type::multiply(omega_ij, omega_j_cntd,
                                           omega_j_cntd_precomp);
          omega_ij = svmin_x(ptrue, omega_ij, svsub_x(ptrue, omega_ij, N));
        }
      } else if constexpr (twiddle_unroll_count == 2) {
        svuint64_t omega_ij_0, omega_ij_1;
        svuint64_t omega_j_cntd, omega_j_cntd_precomp;
        omega_ij_0 = svld1(ptrue, &aux[cntd * j]);
        omega_j_cntd = svdup_lane(omega_ij_0, 0);
        omega_j_cntd_precomp = modmul_type::precompute(omega_j_cntd);
        omega_ij_0 = svdup_u64_m(omega_ij_0, pfirst, unity);
        omega_ij_1 = modmul_type::multiply(omega_ij_0, omega_j_cntd,
                                           omega_j_cntd_precomp);
        omega_ij_1 = svmin_x(ptrue, omega_ij_1, svsub_x(ptrue, omega_ij_1, N));
        omega_j_cntd = modmul_type::multiply(omega_ij_1, omega_j_cntd,
                                             omega_j_cntd_precomp);
        omega_j_cntd =
            svmin_x(ptrue, omega_j_cntd, svsub_x(ptrue, omega_j_cntd, N));
        omega_j_cntd = svdup_lane(omega_j_cntd, 0);
        omega_j_cntd_precomp = modmul_type::precompute(omega_j_cntd);
        for (std::uint64_t i{}; i < m / inner_m; i += cntd * 2) {
          svuint64_t t0, t1;
          t0 = svld1(ptrue, &src[stride_src * j + i + cntd * 0]);
          t1 = svld1(ptrue, &src[stride_src * j + i + cntd * 1]);
          t0 = modmul_type::multiply(t0, omega_ij_0);
          t1 = modmul_type::multiply(t1, omega_ij_1);
          svst1(ptrue, &dst[stride_dst * j + i + cntd * 0], t0);
          svst1(ptrue, &dst[stride_dst * j + i + cntd * 1], t1);
          omega_ij_0 = modmul_type::multiply(omega_ij_0, omega_j_cntd,
                                             omega_j_cntd_precomp);
          omega_ij_0 =
              svmin_x(ptrue, omega_ij_0, svsub_x(ptrue, omega_ij_0, N));
          omega_ij_1 = modmul_type::multiply(omega_ij_1, omega_j_cntd,
                                             omega_j_cntd_precomp);
          omega_ij_1 =
              svmin_x(ptrue, omega_ij_1, svsub_x(ptrue, omega_ij_1, N));
        }
      } else if constexpr (twiddle_unroll_count == 4) {
        svuint64_t omega_ij_0, omega_ij_1, omega_ij_2, omega_ij_3;
        svuint64_t omega_j_cntd, omega_j_cntd_precomp;
        omega_ij_0 = svld1(ptrue, &aux[cntd * j]);
        omega_j_cntd = svdup_lane(omega_ij_0, 0);
        omega_j_cntd_precomp = modmul_type::precompute(omega_j_cntd);
        omega_ij_0 = svdup_u64_m(omega_ij_0, pfirst, unity);
        omega_ij_1 = modmul_type::multiply(omega_ij_0, omega_j_cntd,
                                           omega_j_cntd_precomp);
        omega_ij_1 = svmin_x(ptrue, omega_ij_1, svsub_x(ptrue, omega_ij_1, N));
        omega_ij_2 = modmul_type::multiply(omega_ij_1, omega_j_cntd,
                                           omega_j_cntd_precomp);
        omega_ij_2 = svmin_x(ptrue, omega_ij_2, svsub_x(ptrue, omega_ij_2, N));
        omega_ij_3 = modmul_type::multiply(omega_ij_2, omega_j_cntd,
                                           omega_j_cntd_precomp);
        omega_ij_3 = svmin_x(ptrue, omega_ij_3, svsub_x(ptrue, omega_ij_3, N));
        omega_j_cntd = modmul_type::multiply(omega_ij_3, omega_j_cntd,
                                             omega_j_cntd_precomp);
        omega_j_cntd =
            svmin_x(ptrue, omega_j_cntd, svsub_x(ptrue, omega_j_cntd, N));
        omega_j_cntd = svdup_lane(omega_j_cntd, 0);
        omega_j_cntd_precomp = modmul_type::precompute(omega_j_cntd);
        for (std::uint64_t i{}; i < m / inner_m; i += cntd * 4) {
          svuint64_t t0, t1, t2, t3;
          t0 = svld1(ptrue, &src[stride_src * j + i + cntd * 0]);
          t1 = svld1(ptrue, &src[stride_src * j + i + cntd * 1]);
          t2 = svld1(ptrue, &src[stride_src * j + i + cntd * 2]);
          t3 = svld1(ptrue, &src[stride_src * j + i + cntd * 3]);
          t0 = modmul_type::multiply(t0, omega_ij_0);
          t1 = modmul_type::multiply(t1, omega_ij_1);
          t2 = modmul_type::multiply(t2, omega_ij_2);
          t3 = modmul_type::multiply(t3, omega_ij_3);
          svst1(ptrue, &dst[stride_dst * j + i + cntd * 0], t0);
          svst1(ptrue, &dst[stride_dst * j + i + cntd * 1], t1);
          svst1(ptrue, &dst[stride_dst * j + i + cntd * 2], t2);
          svst1(ptrue, &dst[stride_dst * j + i + cntd * 3], t3);
          omega_ij_0 = modmul_type::multiply(omega_ij_0, omega_j_cntd,
                                             omega_j_cntd_precomp);
          omega_ij_0 =
              svmin_x(ptrue, omega_ij_0, svsub_x(ptrue, omega_ij_0, N));
          omega_ij_1 = modmul_type::multiply(omega_ij_1, omega_j_cntd,
                                             omega_j_cntd_precomp);
          omega_ij_1 =
              svmin_x(ptrue, omega_ij_1, svsub_x(ptrue, omega_ij_1, N));
          omega_ij_2 = modmul_type::multiply(omega_ij_2, omega_j_cntd,
                                             omega_j_cntd_precomp);
          omega_ij_2 =
              svmin_x(ptrue, omega_ij_2, svsub_x(ptrue, omega_ij_2, N));
          omega_ij_3 = modmul_type::multiply(omega_ij_3, omega_j_cntd,
                                             omega_j_cntd_precomp);
          omega_ij_3 =
              svmin_x(ptrue, omega_ij_3, svsub_x(ptrue, omega_ij_3, N));
        }
      } else {
        throw std::invalid_argument{"Unsupported twiddle unroll count"};
      }
    }
  }

  static void twiddle_rows_forward(std::uint64_t *const dst,
                                   const std::byte *const aux,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride) {
    twiddle_rows_forward(dst, dst, aux, j_start, j_end, stride, stride);
  }

  static void compute_forward(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    buffer_type buffer;
    compute_forward_without_twiddle(dst, src, aux, buffer);
    twiddle_rows_forward(dst, aux, 0, inner_m, m / inner_m);
  }

  static void compute_forward(std::uint64_t *const dst, const std::byte *&aux) {
    compute_forward(dst, dst, aux);
  }

  template <class vector_type>
  static void prepare_inverse(vector_type &aux) {
    const auto precompute{
        [&aux](const std::uint64_t num_elements, std::uint64_t &omega,
               const std::uint64_t factor, const bool precompute_precompute) {
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

    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    constexpr std::uint64_t omega_m{modulus_type::get_root_inverse(m)};
    if (m / inner_m % cntd != 0) {
      throw std::invalid_argument{"SVE vector length is too large"};
    }
    for (std::uint64_t j{}; j < inner_m; ++j) {
      const std::uint64_t omega_j{modulus_type::power(
          omega_m, bitreverse(j) >> (65 - std::bit_width(inner_m)))};
      std::uint64_t omega_ij;
      omega_ij = modulus_type::power(omega_j, cntd);
      precompute(1, omega_ij, {}, false);
      omega_ij = omega_j;
      precompute(cntd - 1, omega_ij, omega_j, false);
    }

    /* aux size, to be corrected later */
    aux.template push_back<std::uint64_t>({});
    const typename vector_type::size_type aux_pos_after_size{aux.size()};

    inner_kernel_type::prepare_inverse(aux);

    aux.template reinterpret_at<std::uint64_t>(aux_pos_after_size -
                                      sizeof(std::uint64_t)) =
        aux.size() - aux_pos_after_size;
  }

  static void twiddle_rows_inverse(std::uint64_t *const dst,
                                   const std::uint64_t *const src,
                                   const std::byte *const aux_arg,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride_dst,
                                   const std::uint64_t stride_src) {
    constexpr std::uint64_t N{modulus_type::get_modulus()};
    constexpr std::uint64_t unity{modmul_type::to_montgomery(1)};
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    const svbool_t ptrue{svptrue_b8()};
    const svbool_t pfirst{svptrue_pat_b8(SV_VL1)};
    const std::uint64_t *aux{reinterpret_cast<const std::uint64_t *>(aux_arg)};

    for (std::uint64_t j{j_start}; j < j_end; ++j) {
      svuint64_t omega_ij{svld1(ptrue, &aux[cntd * j])};
      const svuint64_t omega_j_cntd{svdup_lane(omega_ij, 0)};
      const svuint64_t omega_j_cntd_precomp{
          modmul_type::precompute(omega_j_cntd)};
      omega_ij = svdup_u64_m(omega_ij, pfirst, unity);
      for (std::uint64_t i{}; i < m / inner_m; i += cntd) {
        svuint64_t t;
        t = svld1(ptrue, &src[stride_src * j + i]);
        t = modmul_type::multiply(t, omega_ij);
        svst1(ptrue, &dst[stride_dst * j + i], t);
        omega_ij =
            modmul_type::multiply(omega_ij, omega_j_cntd, omega_j_cntd_precomp);
        omega_ij = svmin_x(ptrue, omega_ij, svsub_x(ptrue, omega_ij, N));
      }
    }
  }

  static void twiddle_rows_inverse(std::uint64_t *const dst,
                                   const std::byte *const aux,
                                   const std::uint64_t j_start,
                                   const std::uint64_t j_end,
                                   const std::uint64_t stride) {
    twiddle_rows_inverse(dst, dst, aux, j_start, j_end, stride, stride);
  }

  static void compute_inverse_without_twiddle(std::uint64_t *const dst,
                                              const std::uint64_t *const src,
                                              const std::byte *&aux_arg,
                                              buffer_type &buffer_arg) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};

    aux_arg += 8 * cntd * inner_m;
    const std::byte *aux;
    {
      const std::uint64_t aux_size{
          pointer_utility::get_and_advance<std::uint64_t>(aux_arg)};
      aux = aux_arg;
      aux_arg += aux_size;
    }

    auto &buffer{buffer_arg.get()};

    if constexpr (transpose_in_place) {
      if (src == dst) {
        transposition_type::transpose(dst, inner_m);
      } else {
        transposition_type::transpose(dst, src, inner_m, m / inner_m, inner_m,
                                      m / inner_m);
      }
    } else {
      transposition_type::transpose(&buffer[0], src, inner_m, m / inner_m,
                                    inner_m + buffer_padding_elements,
                                    m / inner_m);
    }

    const std::byte *const aux_orig{aux};
    [[omp::directive(for)]]
    for (std::uint64_t i = 0; i < m / inner_m; ++i) {
      aux = aux_orig;
      if constexpr (transpose_in_place) {
        inner_kernel_type::compute_inverse(&dst[inner_m * i], aux);
      } else {
        inner_kernel_type::compute_inverse(
            &buffer[(inner_m + buffer_padding_elements) * i], aux);
      }
    }

    if constexpr (transpose_in_place) {
      transposition_type::transpose(dst, inner_m);
    } else {
      transposition_type::transpose(dst, &buffer[0], m / inner_m, inner_m,
                                    m / inner_m,
                                    inner_m + buffer_padding_elements);
    }
  }

  static void compute_inverse_without_twiddle(std::uint64_t *const dst,
                                              const std::byte *&aux,
                                              buffer_type &buffer) {
    compute_inverse_without_twiddle(dst, dst, aux, buffer);
  }

  static void compute_inverse(std::uint64_t *const dst,
                              const std::uint64_t *const src,
                              const std::byte *&aux) {
    constexpr std::uint64_t inner_m{inner_kernel_type::get_m()};
    buffer_type buffer;
    twiddle_rows_inverse(dst, src, aux, 0, inner_m, m / inner_m, m / inner_m);
    compute_inverse_without_twiddle(dst, aux, buffer);
  }

  static void compute_inverse(std::uint64_t *const dst, const std::byte *&aux) {
    compute_inverse(dst, dst, aux);
  }
};

} // namespace sventt

#endif /* SVENTT_LAYER_SVE_GENERIC_HPP_INCLUDED */

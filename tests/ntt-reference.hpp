// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef NTT_REFERENCE_HPP_INCLUDED
#define NTT_REFERENCE_HPP_INCLUDED

#include <bit>
#include <cstdint>
#include <stdexcept>

class NTTReference {
  const std::uint64_t m, log2m;
  const std::uint64_t N;
  const std::uint64_t omega_m, omegainv_m, minv;

  std::uint64_t modmul(const std::uint64_t x, const std::uint64_t y) const {
    return x * static_cast<unsigned __int128>(y) % N;
  }

  std::uint64_t modpow(std::uint64_t x, std::uint64_t e) const {
    std::uint64_t t{1};
    for (; e; e >>= 1) {
      if (e & 1) {
        t = modmul(t, x);
      }
      x = modmul(x, x);
    }
    return t;
  }

public:
  NTTReference(const std::uint64_t m, const std::uint64_t N,
               const std::uint64_t omega)
      : m{m}, log2m{static_cast<std::uint64_t>(std::countr_zero(m))}, N{N},
        omega_m{modpow(omega, (N - 1) >> log2m)},
        omegainv_m{modpow(omega_m, N - 2)}, minv{modpow(m, N - 2)} {
    if (!std::has_single_bit(m)) {
      throw std::invalid_argument{
          "Transform length must be a power of two for now"};
    }
  }

  void compute_forward(std::uint64_t *const dst,
                       const std::uint64_t *src) const {
    std::uint64_t omega_2l{omega_m};
    for (std::uint64_t i{log2m - 1}; i < log2m; --i) {
      const std::uint64_t l{std::uint64_t{1} << i};
      std::uint64_t omega_2l_j{1};
      for (std::uint64_t j{0}; j < l; ++j) {
        for (std::uint64_t k{j}; k < m; k += l * 2) {
          const std::uint64_t x0{src[k]}, x1{src[k + l]};
          dst[k] = (x0 < N - x1) ? (x0 + x1) : (x0 + x1 - N);
          dst[k + l] =
              modmul((x0 >= x1) ? (x0 - x1) : (x0 - x1 + N), omega_2l_j);
        }
        omega_2l_j = modmul(omega_2l_j, omega_2l);
      }
      omega_2l = modmul(omega_2l, omega_2l);
      src = dst;
    }
  }

  void compute_inverse(std::uint64_t *const dst,
                       const std::uint64_t *const src) const {
    for (std::uint64_t i{0}; i < m; ++i) {
      dst[i] = modmul(src[i], minv);
    }

    for (std::uint64_t i{0}; i < log2m; ++i) {
      const std::uint64_t l{std::uint64_t{1} << i};
      const std::uint64_t omegainv_2l{
          modpow(omegainv_m, std::uint64_t{1} << (log2m - i - 1))};
      std::uint64_t omegainv_2l_j{1};
      for (std::uint64_t j{0}; j < l; ++j) {
        for (std::uint64_t k{j}; k < m; k += l * 2) {
          const std::uint64_t x0{dst[k]}, x1{modmul(dst[k + l], omegainv_2l_j)};
          dst[k] = (x0 < N - x1) ? (x0 + x1) : (x0 + x1 - N);
          dst[k + l] = (x0 >= x1) ? (x0 - x1) : (x0 - x1 + N);
        }
        omegainv_2l_j = modmul(omegainv_2l_j, omegainv_2l);
      }
    }
  }
};

#endif /* NTT_REFERENCE_HPP_INCLUDED */

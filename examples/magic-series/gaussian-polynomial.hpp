// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_EXAMPLES_MAGIC_SERIES_GAUSSIAN_POLYNOMIAL_HPP_INCLUDED
#define SVENTT_EXAMPLES_MAGIC_SERIES_GAUSSIAN_POLYNOMIAL_HPP_INCLUDED

#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <omp.h>

#include "../../tests/utility.hpp"

#include "restricted-partition.hpp"

template <class modulus_type, class range_type>
static void calculate_q_pochhammer(range_type &&coefficients,
                                   const std::uint64_t k)
  requires(std::ranges::random_access_range<range_type> &&
           std::same_as<std::ranges::range_value_t<range_type>, std::uint64_t>)
{
  if (std::ranges::size(coefficients) < k * (k + 1) / 2 + 1) {
    throw std::invalid_argument{"coefficient vector is too small"};
  }

  coefficients[0] = 1;
  std::uint64_t length{1};

  for (std::uint64_t i{1}; i <= k; ++i) {
    /* Multiply by 1 - q^i. */
    std::uint64_t j{length - 1};
    for (; j < length && i + j >= length; --j) {
      coefficients[i + j] = modulus_type::negate(coefficients[j]);
    }
    for (; j < length; --j) {
      coefficients[i + j] =
          modulus_type::subtract(coefficients[i + j], coefficients[j]);
    }

    length += i;
  }
}

/*
 * Iteratively generates the non-zero segments in the numerator of the Gaussian
 * polynomial using the Rothe's identity:
 * qbinom(k, j, q) = qbinom(k, j - 1, q) * (1 - q^(k - j + 1)) / (1 - q^j)
 */
template <class modulus_type_> class GaussianPolynomialNumeratorSegment {

public:
  using modulus_type = modulus_type_;

private:
  std::uint64_t k, j;
  std::vector<std::uint64_t> coefficients;

public:
  GaussianPolynomialNumeratorSegment(void) = default;

  GaussianPolynomialNumeratorSegment(const std::uint64_t k) : k{k}, j{} {
    coefficients.reserve(k / 2 * (k - k / 2) + 1);
  }

  std::uint64_t get_k(void) const { return k; }

  std::uint64_t get_j(void) const { return j; }

  const std::vector<std::uint64_t> &get_coefficients(void) const {
    return coefficients;
  }

  void advance(void) {
    if (j == 0) {
      coefficients = {1};
      ++j;
      return;
    } else if (j == k) {
      coefficients = {1};
      return;
    }

    coefficients.resize(j * (k - j) + 1);

    /*
     * Multiply by 1 - q^(k - j + 1).
     */
    for (std::uint64_t l{j * (k - j)}; l >= k - j + 1; --l) {
      coefficients.at(l) = modulus_type::subtract(
          coefficients.at(l), coefficients.at(l - (k - j + 1)));
    }

    /*
     * Divide by 1 - q^j.
     */
    for (std::uint64_t l{j}; l <= j * (k - j); ++l) {
      coefficients.at(l) =
          modulus_type::add(coefficients.at(l), coefficients.at(l - j));
    }

    ++j;
  }
};

template <class modulus_type_> class GaussianPolynomialNumerator {

public:
  using modulus_type = modulus_type_;

private:
  std::uint64_t n, k;
  std::uint64_t j, l, i;
  GaussianPolynomialNumeratorSegment<modulus_type> segment;

public:
  GaussianPolynomialNumerator(void) = default;

  GaussianPolynomialNumerator(const std::uint64_t n, const std::uint64_t k)
      : n{n}, k{k}, j{}, l{}, i{}, segment{k} {}

  void subtract_next(std::uint64_t *const minuend, const std::uint64_t size) {
    std::uint64_t pos{};

    for (; j <= k; ++j, l = 0) {
      if (l == 0) {
        segment.advance();
      }

      const std::uint64_t shift_next{(j + 1) * (n - k + 1) + (j + 1) * j / 2};
      for (; i < shift_next && pos < size; ++i, ++pos, ++l) {
        if (l <= j * (k - j)) {
          minuend[pos] =
              (j % 2 == 1 ? modulus_type::add : modulus_type::subtract)(
                  minuend[pos], segment.get_coefficients().at(l));
        }
      }

      if (pos == size) {
        return;
      }
    }
  }
};

template <class ntt_type>
static std::uint64_t calculate_gaussian_polynomial_coefficient(
    const std::uint64_t n, const std::uint64_t k, const std::uint64_t d,
    const ntt_type &ntt) {
  using modulus_type = ntt_type::modulus_type;
  using modmul_type = sventt::PAdic64SVE<modulus_type>;
  const bool allocate_huge_pages{false};

  const svbool_t ptrue{svptrue_b8()};

  if (d > k * (n - k)) {
    throw std::invalid_argument{"d is out of range"};
  }
  if (n < (k * k + 2 * k + k % 2 + 3) / 4) {
    throw std::invalid_argument{"n is too small; segments will overlap"};
  }

  if (ntt_type::get_m() < (k * (k + 1) / 2 + 1) * 2) {
    throw std::invalid_argument{"NTT length is too small"};
  }
  constexpr std::uint64_t ntt_size{ntt_type::get_m()};
  constexpr std::uint64_t chunk_size{ntt_size / 2};

  GaussianPolynomialNumerator<modulus_type> numerator(n, k);

  sventt::PageMemory<std::uint64_t> denominator(ntt_size, allocate_huge_pages);
  calculate_q_pochhammer<modulus_type>(
      denominator | std::views::drop(chunk_size), k);
  ntt.compute_forward(denominator.data());
  for (std::uint64_t i = 0; i < ntt_size; ++i) {
    denominator[i] = modmul_type::to_montgomery(denominator[i]);
  }

  sventt::PageMemory<std::uint64_t> denominator_inverse(ntt_size,
                                                        allocate_huge_pages);
  {
    RestrictedPartition<modulus_type> partition(k);
    for (std::uint64_t i{}; i < chunk_size; ++i) {
      denominator_inverse[i] = modulus_type::negate(partition());
      partition.advance();
    }
  }
  ntt.compute_forward(denominator_inverse.data());
  for (std::uint64_t i = 0; i < ntt_size; ++i) {
    denominator_inverse[i] = modmul_type::to_montgomery(denominator_inverse[i]);
  }

  sventt::PageMemory<std::uint64_t> coefficients(ntt_size, allocate_huge_pages);
  for (std::uint64_t i{}; i < d; i += chunk_size) {
    numerator.subtract_next(coefficients.data(), chunk_size);

    ntt.compute_forward(coefficients.data());

#pragma omp parallel
    {
      const auto [part_offset, part_size] = partition_static(
          ntt_size, omp_get_num_threads(), omp_get_thread_num(), sventt::cntd);
      for (std::uint64_t j = part_offset; j < part_offset + part_size;
           j += sventt::cntd) {
        svst1(ptrue, &coefficients[j],
              modmul_type::multiply_normalize(
                  svld1(ptrue, &coefficients[j]),
                  svld1(ptrue, &denominator_inverse[j])));
      }
    }

    ntt.compute_inverse(coefficients.data());

    if (d < i + chunk_size) {
      return coefficients.at(d - i);
    }

    memset_parallel(&coefficients[chunk_size], 0,
                    sizeof(std::uint64_t) * (ntt_size - chunk_size));

    ntt.compute_forward(coefficients.data());

#pragma omp parallel
    {
      const auto [part_offset, part_size] = partition_static(
          ntt_size, omp_get_num_threads(), omp_get_thread_num(), sventt::cntd);
      for (std::uint64_t j = part_offset; j < part_offset + part_size;
           j += sventt::cntd) {
        svst1(ptrue, &coefficients[j],
              modmul_type::multiply_normalize(svld1(ptrue, &coefficients[j]),
                                              svld1(ptrue, &denominator[j])));
      }
    }

    ntt.compute_inverse(coefficients.data());

    memset_parallel(&coefficients[chunk_size], 0,
                    sizeof(std::uint64_t) * (ntt_size - chunk_size));
  }

  throw std::runtime_error{"internal error"};
}

template <class ntt_type>
static std::uint64_t calculate_number_of_magic_series(const std::uint64_t m,
                                                      const ntt_type &ntt) {
  return calculate_gaussian_polynomial_coefficient(m * m, m,
                                                   m * m * (m - 1) / 2, ntt);
}

#endif /* SVENTT_EXAMPLES_MAGIC_SERIES_GAUSSIAN_POLYNOMIAL_HPP_INCLUDED */

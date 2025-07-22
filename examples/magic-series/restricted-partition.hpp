// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_EXAMPLES_MAGIC_SERIES_RESTRICTED_PARTITION_HPP_INCLUDED
#define SVENTT_EXAMPLES_MAGIC_SERIES_RESTRICTED_PARTITION_HPP_INCLUDED

#include <algorithm>
#include <cstdint>
#include <vector>

template <class modulus_type_> class RestrictedPartition {

public:
  using modulus_type = modulus_type_;

private:
  std::uint64_t n, k;
  std::vector<std::uint64_t> table;

public:
  RestrictedPartition(void) = default;

  RestrictedPartition(const std::uint64_t k)
      : n{}, k{k}, table((k + 1) * (k + 1)) {
    /* Set p(0, 0) = 0 and p(0, i) = 1 for i = 1, 2, ..., k. */
    std::fill(table.begin() + 1, table.begin() + (k + 1), 1);
  }

  std::uint64_t get_n(void) const { return n; }

  std::uint64_t get_k(void) const { return k; }

  std::uint64_t operator()(void) const {
    return table.at((k + 1) * (n % (k + 1)) + k);
  }

  void advance(void) {
    ++n;

    /*
     * p(n, k) = p(n - k, k) + p(n, k - 1)
     *         = p(n - k, k) + p(n - k + 1, k - 1) + p(n, k - 2)
     *         = p(n - k, k) + p(n - k + 1, k - 1) + ... + p(n, 0)
     */
    for (std::uint64_t i{1}; i <= k; ++i) {
      table.at((k + 1) * (n % (k + 1)) + i) = modulus_type::add(
          table.at((k + 1) * (n % (k + 1)) + (i - 1)),
          table.at((k + 1) * ((n - i + k + 1) % (k + 1)) + i));
    }
  }
};

#endif /* SVENTT_EXAMPLES_MAGIC_SERIES_RESTRICTED_PARTITION_HPP_INCLUDED */

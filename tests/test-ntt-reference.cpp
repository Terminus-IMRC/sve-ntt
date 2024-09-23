// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "ntt-reference.hpp"

TEST(NTTReference, ForwardInverse) {
  const std::vector<std::tuple<std::uint64_t, std::uint64_t>> N_list{
      {UINT64_C(0x0c40'0000'0000'0001), 5},
      {UINT64_C(0x0c60'0000'0000'0001), 7},
      {UINT64_C(0x0003'f000'0000'0001), 11},
      {UINT64_C(0x0002'5800'0000'0001), 11},
      {UINT64_C(0xffff'ffff'0000'0001), 7},
  };

  std::default_random_engine gen;

  for (const auto &[N, omega] : N_list) {
    for (const std::uint64_t log2m : {1, 2, 3, 4, 5, 6, 7}) {
      const std::uint64_t m{std::uint64_t{1} << log2m};
      std::cout << "Testing N = 0x";
      std::ostream{std::cout.rdbuf()} << std::hex << std::setw(64 / 4)
                                      << std::setfill('0') << N;
      std::cout << ", log2m = " << log2m << std::endl;

      std::uniform_int_distribution<std::uint64_t> dist{std::uint64_t{0},
                                                        N - 1};
      std::vector<std::uint64_t> a(m), b(m), c(m);
      std::generate(a.begin(), a.end(), [&gen, &dist] { return dist(gen); });
      std::generate(b.begin(), b.end(), [&gen, &dist] { return dist(gen); });
      std::generate(c.begin(), c.end(), [&gen, &dist] { return dist(gen); });

      const NTTReference ntt{m, N, omega};
      ntt.compute_forward(b.data(), a.data());

      {
        std::uint64_t sum{0};
        for (std::uint64_t i{0}; i < m; ++i) {
          sum = (sum < N - a.at(i)) ? (sum + a.at(i)) : (sum + a.at(i) - N);
        }
        ASSERT_EQ(b.at(0), sum);
      }

      {
        std::uint64_t sum{0};
        for (std::uint64_t i{0}; i < m; ++i) {
          if (i % 2 == 0) {
            sum = (sum < N - a.at(i)) ? (sum + a.at(i)) : (sum + a.at(i) - N);
          } else {
            sum = (sum >= a.at(i)) ? (sum - a.at(i)) : (sum - a.at(i) + N);
          }
        }
        ASSERT_EQ(b.at(1), sum);
      }

      {
        std::uint64_t omega_m{1}, omega_m_i{1}, sum{0};
        for (std::uint64_t e{(N - 1) >> log2m}, t{omega}; e;
             e >>= 1, t = static_cast<unsigned __int128>(t) * t % N) {
          if (e & 1) {
            omega_m = static_cast<unsigned __int128>(omega_m) * t % N;
          }
        }
        for (std::uint64_t i{0}; i < m; ++i) {
          const std::uint64_t t{static_cast<std::uint64_t>(
              static_cast<unsigned __int128>(a.at(i)) * omega_m_i % N)};
          sum = (sum < N - t) ? (sum + t) : (sum + t - N);
          omega_m_i = static_cast<unsigned __int128>(omega_m_i) * omega_m % N;
        }
        ASSERT_EQ(b.at(m / 2), sum);
      }

      ntt.compute_inverse(c.data(), b.data());
      for (std::uint64_t i{0}; i < m; ++i) {
        ASSERT_EQ(c.at(i), a.at(i)) << "i = " << i;
      }
    }
  }
}

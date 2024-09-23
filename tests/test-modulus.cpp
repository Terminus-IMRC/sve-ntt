// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#include <sventt/sventt.hpp>

#include <cstdint>
#include <initializer_list>
#include <iostream>

#include <gtest/gtest.h>

TEST(Modulus, SumOfRoots) {
  using modulus_type = sventt::Modulus<UINT64_C(0xffff'ffff'0000'0001), 7>;
  std::cout << "modulus = " << modulus_type::get_modulus() << std::endl;
  std::cout << "generator = " << modulus_type::get_generator() << std::endl;

  for (const std::uint64_t order : std::initializer_list<std::uint64_t>{
           std::uint64_t{1} << 28, 3, 5, 17, 257, 65537,
           (std::uint64_t{1} << 14) * 5 * 17 * 257}) {
    std::cout << "Testing order = " << order << std::endl;

    {
      std::uint64_t sum{};
      const std::uint64_t root_1{modulus_type::get_root_forward(order)};
      ASSERT_GT(root_1, 1);
      std::cout << "forward_root = " << root_1 << std::endl;
      std::uint64_t root_i{1};
      for (std::uint64_t i{}; i < order; ++i) {
        sum = modulus_type::add(sum, root_i);
        root_i = modulus_type::multiply(root_i, root_1);
      }
      EXPECT_EQ(sum, 0);
    }

    {
      std::uint64_t sum{};
      const std::uint64_t root_1{modulus_type::get_root_inverse(order)};
      ASSERT_GT(root_1, 1);
      std::cout << "inverse_root = " << root_1 << std::endl;
      std::uint64_t root_i{1};
      for (std::uint64_t i{}; i < order; ++i) {
        sum = modulus_type::add(sum, root_i);
        root_i = modulus_type::multiply(root_i, root_1);
      }
      EXPECT_EQ(sum, 0);
    }
  }
}

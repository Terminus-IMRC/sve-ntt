// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_UTILITY_HPP_INCLUDED
#define SVENTT_UTILITY_HPP_INCLUDED

#include <bit>
#include <cstdint>

namespace sventt {

static inline constexpr std::uint64_t bitreverse(std::uint64_t x) {
  const std::uint64_t c0{UINT64_C(0x5555'5555'5555'5555)},
      c1{UINT64_C(0x3333'3333'3333'3333)}, c2{UINT64_C(0x0f0f'0f0f'0f0f'0f0f)},
      c3{UINT64_C(0x00ff'00ff'00ff'00ff)}, c4{UINT64_C(0x0000'ffff'0000'ffff)};
  x = ((x & c0) << 1) | ((x >> 1) & c0);
  x = ((x & c1) << 2) | ((x >> 2) & c1);
  x = ((x & c2) << 4) | ((x >> 4) & c2);
  x = ((x & c3) << 8) | ((x >> 8) & c3);
  x = ((x & c4) << 16) | ((x >> 16) & c4);
  x = std::rotl(x, 32);
  return x;
}

} // namespace sventt

#endif /* SVENTT_UTILITY_HPP_INCLUDED */

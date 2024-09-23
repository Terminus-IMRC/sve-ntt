// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_TESTS_UTILITY_HPP_INCLUDED
#define SVENTT_TESTS_UTILITY_HPP_INCLUDED

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

template <class value_type, value_type... values>
static constexpr std::array<value_type, sizeof...(values)>
integer_sequence_to_array(
    const std::integer_sequence<value_type, values...> &) {
  return {values...};
}

template <std::uint64_t... bounds> class for_each {

  template <std::uint64_t index, std::uint64_t... values, class function_type>
  static constexpr void impl(const function_type &function) {
    if constexpr (index == sizeof...(bounds)) {
      function.template operator()<values...>();
    } else {
      [function]<std::uint64_t... value>(
          const std::integer_sequence<std::uint64_t, value...> &) {
        (impl<index + 1, values..., value>(function), ...);
      }(std::make_integer_sequence<std::uint64_t,
                                   std::array{bounds...}.at(index)>{});
    }
  }

public:
  template <class function_type>
  constexpr for_each(const function_type &function) {
    impl<0>(function);
  }
};

template <class... value_lists> class for_each_element {

public:
  template <class function_type>
  constexpr for_each_element(const function_type &function) {
    for_each<value_lists::size()...>{[function]<std::uint64_t... indices> {
      function.template
      operator()<integer_sequence_to_array(value_lists{}).at(indices)...>();
    }};
  }
};

static constexpr std::tuple<std::uint64_t, std::uint64_t>
partition_static(const std::uint64_t size, const std::uint64_t num_parts,
                 const std::uint64_t part_num,
                 const std::uint64_t part_align = 1) {
  const std::uint64_t part_offset = size / part_align * part_num / num_parts;
  const std::uint64_t next_part_offset =
      size / part_align * (part_num + 1) / num_parts;
  const std::uint64_t part_size = next_part_offset - part_offset;
  return {part_offset * part_align, (part_num == num_parts - 1)
                                        ? size - part_offset * part_align
                                        : part_size * part_align};
}

[[maybe_unused]]
static void memset_parallel(void *const dst, const std::uint8_t value,
                            const std::uint64_t size) {
  [[omp::directive(parallel)]] {
    const std::uint64_t part_align = (size >= 65536) ? 65536 : 256;
    const auto [part_offset, part_size] = partition_static(
        size, omp_get_num_threads(), omp_get_thread_num(), part_align);
    std::memset(static_cast<std::byte *>(dst) + part_offset, value, part_size);
  }
}

[[maybe_unused]]
static void memcpy_parallel(void *const dst, const void *const src,
                            const std::uint64_t size) {
  [[omp::directive(parallel)]] {
    const std::uint64_t part_align = (size >= 65536) ? 65536 : 256;
    const auto [part_offset, part_size] = partition_static(
        size, omp_get_num_threads(), omp_get_thread_num(), part_align);
    std::memcpy(static_cast<std::byte *>(dst) + part_offset,
                static_cast<const std::byte *>(src) + part_offset, part_size);
  }
}

[[maybe_unused]]
static int memcmp_parallel(const void *const src0, const void *const src1,
                           const std::uint64_t size) {
  std::vector<int> results(omp_get_max_threads());
  [[omp::directive(parallel)]] {
    const std::uint64_t part_align = (size >= 65536) ? 65536 : 256;
    const auto [part_offset, part_size] = partition_static(
        size, omp_get_num_threads(), omp_get_thread_num(), part_align);
    results[omp_get_thread_num()] = std::memcmp(
        static_cast<const std::byte *>(src0) + part_offset,
        static_cast<const std::byte *>(src1) + part_offset, part_size);
  }
  const auto it =
      std::find_if(results.cbegin(), results.cend(), std::identity{});
  return (it == results.cend()) ? 0 : *it;
}

template <class iterator_type, class value_type>
[[maybe_unused]]
static bool is_iota_2d_parallel(iterator_type begin, const std::uint64_t rows,
                                const std::uint64_t columns,
                                const std::uint64_t stride,
                                const value_type value) {
  bool is_iota = true;
  [[omp::directive(parallel for, reduction(&& : is_iota))]]
  for (std::uint64_t i = 0; i < rows; ++i) {
    const iterator_type part_begin = std::next(begin, stride * i);
    const iterator_type part_end = std::next(begin, stride * i + columns);
    is_iota =
        is_iota &&
        std::all_of(part_begin, part_end,
                    [begin, columns, value, i,
                     part_begin](std::iter_value_t<iterator_type> &element) {
                      return element == value + columns * i +
                                            std::distance(part_begin, &element);
                    });
  }
  return is_iota;
}

template <class iterator_type, class value_type>
[[maybe_unused]]
static void iota_parallel(iterator_type begin, iterator_type end,
                          const value_type value) {
  [[omp::directive(parallel)]] {
    const auto size = std::distance(begin, end);
    const std::uint64_t part_align = (size >= 65536) ? 65536 : 256;
    const auto [part_offset, part_size] = partition_static(
        size, omp_get_num_threads(), omp_get_thread_num(), part_align);
    std::iota(std::next(begin, part_offset),
              std::next(begin, part_offset + part_size),
              static_cast<std::iter_value_t<iterator_type>>(value) +
                  part_offset);
  }
}

template <class iterator_type, class value_type>
[[maybe_unused]]
static void iota_2d_parallel(iterator_type begin, const std::uint64_t rows,
                             const std::uint64_t columns,
                             const std::uint64_t stride,
                             const value_type value) {
  [[omp::directive(parallel for)]]
  for (std::uint64_t i = 0; i < rows; ++i) {
    const iterator_type part_begin = std::next(begin, stride * i);
    const iterator_type part_end = std::next(begin, stride * i + columns);
    std::iota(part_begin, part_end, value + columns * i);
  }
}

template <class iterator_type>
[[maybe_unused]]
static std::iter_value_t<iterator_type> reduce_serial(iterator_type begin,
                                                      iterator_type end)
  requires std::same_as<std::iter_value_t<iterator_type>, std::uint8_t>
{
  using size_type = std::iter_difference_t<iterator_type>;
  const size_type size = std::distance(begin, end);
  const std::uint8_t *const src = &*begin;

  const svbool_t ptrue = svptrue_b8();
  const size_type cnt = svcntb();
  svuint8_t x0 = svdup_u8(0), x1 = svdup_u8(0), x2 = svdup_u8(0),
            x3 = svdup_u8(0), x4 = svdup_u8(0), x5 = svdup_u8(0),
            x6 = svdup_u8(0), x7 = svdup_u8(0), x8 = svdup_u8(0),
            x9 = svdup_u8(0), x10 = svdup_u8(0), x11 = svdup_u8(0),
            x12 = svdup_u8(0), x13 = svdup_u8(0), x14 = svdup_u8(0),
            x15 = svdup_u8(0);

  size_type i = 0;
  /* TODO: Deal with unaligned address. */
  for (; i + cnt * 16 < size; i += cnt * 16) {
    svuint8_t y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14,
        y15;
    y0 = svld1(ptrue, &src[i + cnt * 0]);
    y1 = svld1(ptrue, &src[i + cnt * 1]);
    y2 = svld1(ptrue, &src[i + cnt * 2]);
    y3 = svld1(ptrue, &src[i + cnt * 3]);
    y4 = svld1(ptrue, &src[i + cnt * 4]);
    y5 = svld1(ptrue, &src[i + cnt * 5]);
    y6 = svld1(ptrue, &src[i + cnt * 6]);
    y7 = svld1(ptrue, &src[i + cnt * 7]);
    y8 = svld1(ptrue, &src[i + cnt * 8]);
    y9 = svld1(ptrue, &src[i + cnt * 9]);
    y10 = svld1(ptrue, &src[i + cnt * 10]);
    y11 = svld1(ptrue, &src[i + cnt * 11]);
    y12 = svld1(ptrue, &src[i + cnt * 12]);
    y13 = svld1(ptrue, &src[i + cnt * 13]);
    y14 = svld1(ptrue, &src[i + cnt * 14]);
    y15 = svld1(ptrue, &src[i + cnt * 15]);
    x0 = svadd_x(ptrue, x0, y0);
    x1 = svadd_x(ptrue, x1, y1);
    x2 = svadd_x(ptrue, x2, y2);
    x3 = svadd_x(ptrue, x3, y3);
    x4 = svadd_x(ptrue, x4, y4);
    x5 = svadd_x(ptrue, x5, y5);
    x6 = svadd_x(ptrue, x6, y6);
    x7 = svadd_x(ptrue, x7, y7);
    x8 = svadd_x(ptrue, x8, y8);
    x9 = svadd_x(ptrue, x9, y9);
    x10 = svadd_x(ptrue, x10, y10);
    x11 = svadd_x(ptrue, x11, y11);
    x12 = svadd_x(ptrue, x12, y12);
    x13 = svadd_x(ptrue, x13, y13);
    x14 = svadd_x(ptrue, x14, y14);
    x15 = svadd_x(ptrue, x15, y15);
  }
  if (i >= cnt * 16) {
    for (i -= cnt * 16; i < size; i += cnt) {
      x0 = svadd_x(ptrue, x0, svld1(svwhilelt_b8(i, size), &src[i]));
    }
  }

  x0 = svadd_x(ptrue, x0, x1);
  x2 = svadd_x(ptrue, x2, x3);
  x4 = svadd_x(ptrue, x4, x5);
  x6 = svadd_x(ptrue, x6, x7);
  x8 = svadd_x(ptrue, x8, x9);
  x10 = svadd_x(ptrue, x10, x11);
  x12 = svadd_x(ptrue, x12, x13);
  x14 = svadd_x(ptrue, x14, x15);

  x0 = svadd_x(ptrue, x0, x2);
  x4 = svadd_x(ptrue, x4, x6);
  x8 = svadd_x(ptrue, x8, x10);
  x12 = svadd_x(ptrue, x12, x14);

  x0 = svadd_x(ptrue, x0, x4);
  x8 = svadd_x(ptrue, x8, x12);

  x0 = svadd_x(ptrue, x0, x8);

  return svaddv(ptrue, x0);
}

template <class iterator_type>
[[maybe_unused]]
static std::iter_value_t<iterator_type> reduce_parallel(iterator_type begin,
                                                        iterator_type end) {
  std::vector<std::iter_value_t<iterator_type>> results(omp_get_max_threads());
  [[omp::directive(parallel)]] {
    const auto size = std::distance(begin, end);
    const std::uint64_t part_align = (size >= 65536) ? 65536 : 256;
    const auto [part_offset, part_size] = partition_static(
        size, omp_get_num_threads(), omp_get_thread_num(), part_align);
    results[omp_get_thread_num()] =
        reduce_serial(std::next(begin, part_offset),
                      std::next(begin, part_offset + part_size));
  }
  return std::reduce(results.cbegin(), results.cend());
}

#endif /* SVENTT_TESTS_UTILITY_HPP_INCLUDED */

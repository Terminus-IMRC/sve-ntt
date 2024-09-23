// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#include <cstdint>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <benchmark/benchmark.h>

#include <sventt/sventt.hpp>

#include "utility.hpp"

template <class transposition_type>
static void benchmark_transpose_out_of_place_parallel(benchmark::State &state) {
  const std::uint64_t rows = state.range(0);
  const std::uint64_t columns = state.range(1);
  const std::uint64_t pad_src = state.range(2);
  const std::uint64_t pad_dst = state.range(3);

  std::default_random_engine gen;

  const bool allocate_huge_pages = (columns * rows >= (std::uint64_t{1} << 18));
  const std::uint64_t page_size =
      allocate_huge_pages ? (std::uint64_t{1} << 21) : (std::uint64_t{1} << 16);
  sventt::PageMemory<std::uint64_t> src((columns + pad_src) * rows - pad_src,
                                        allocate_huge_pages),
      dst((rows + pad_dst) * columns - pad_dst, allocate_huge_pages);

  const std::uint64_t start{std::uniform_int_distribution<std::uint64_t>{
      0, std::numeric_limits<std::uint64_t>::max() - columns * rows + 1}(gen)};
  /* TODO: Take the number of NUMA domains as an argument. */
  [[omp::directive(parallel)]] {
    sventt::touch_pages_cyclically(
        src.data(), sizeof(std::uint64_t) * src.size(), page_size, 4, 12,
        omp_get_thread_num() / 12, omp_get_thread_num() % 12);
    sventt::touch_pages_cyclically(
        dst.data(), sizeof(std::uint64_t) * dst.size(), page_size, 4, 12,
        omp_get_thread_num() / 12, omp_get_thread_num() % 12);
  }
  iota_2d_parallel(src.begin(), rows, columns, columns + pad_src, start);
  memset_parallel(dst.data(), 0x55, dst.size());

  for (auto _ : state) {
    benchmark::DoNotOptimize(src.data());
    benchmark::DoNotOptimize(dst.data());
    [[omp::directive(parallel)]] transposition_type::transpose(
        dst.data(), src.data(), rows, columns, rows + pad_dst,
        columns + pad_src);
    benchmark::ClobberMemory();
  }

  memset_parallel(&src[0], 0xaa, src.size());
  [[omp::directive(parallel)]] sventt::TransposeParallelSVEInRegister<
      32, 32>::transpose(src.data(), dst.data(), columns, rows,
                         columns + pad_src, rows + pad_dst);
  if (!is_iota_2d_parallel(src.cbegin(), rows, columns, columns + pad_src,
                           start)) {
    throw std::runtime_error{"Result mismatched between implementations"};
  }

  state.SetBytesProcessed(sizeof(std::uint64_t) * columns * rows *
                          state.iterations());
  state.counters["rows"] = rows;
  state.counters["columns"] = columns;
  state.counters["pad_src"] = pad_src;
  state.counters["pad_dst"] = pad_dst;
}

template <class transposition_type>
static void benchmark_transpose_in_place_parallel(benchmark::State &state) {
  const std::uint64_t dim = state.range(0);

  std::default_random_engine gen;

  const bool allocate_huge_pages = (dim * dim >= (std::uint64_t{1} << 18));
  sventt::PageMemory<std::uint64_t> dst(dim * dim, allocate_huge_pages);

  const std::uint64_t start{std::uniform_int_distribution<std::uint64_t>{
      0, std::numeric_limits<std::uint64_t>::max() - dim * dim + 1}(gen)};
  iota_2d_parallel(dst.begin(), dim, dim, dim, start);

  for (auto _ : state) {
    benchmark::DoNotOptimize(dst.data());
    [[omp::directive(parallel)]] transposition_type::transpose(dst.data(), dim);
    benchmark::ClobberMemory();
  }

  if (state.iterations() % 2 == 1) {
    [[omp::directive(parallel)]] sventt::TransposeParallelSVEInRegister<
        32, 32>::transpose(dst.data(), dim);
  }
  if (!is_iota_2d_parallel(dst.cbegin(), dim, dim, dim, start)) {
    throw std::runtime_error{"Result mismatched between implementations"};
  }

  state.SetBytesProcessed(sizeof(std::uint64_t) * dim * dim *
                          state.iterations());
  state.counters["dim"] = dim;
}

static void register_out_of_place(void) {
  const auto apply_sizes{[](benchmark::internal::Benchmark *bench) {
    bench
        ->ArgsProduct({
            benchmark::CreateRange(std::size_t{1} << 8, std::size_t{1} << 15,
                                   2),
            benchmark::CreateRange(std::size_t{1} << 8, std::size_t{1} << 15,
                                   2),
            {0, 32},
            {0, 32},
        })
        ->Args({std::size_t{1} << 15, std::size_t{1} << 16, 0, 0})
        ->Args({std::size_t{1} << 15, std::size_t{1} << 16, 32, 0})
        ->Args({std::size_t{1} << 15, std::size_t{1} << 16, 0, 32})
        ->Args({std::size_t{1} << 15, std::size_t{1} << 16, 32, 32})
        ->Args({std::size_t{1} << 16, std::size_t{1} << 15, 0, 0})
        ->Args({std::size_t{1} << 16, std::size_t{1} << 15, 32, 0})
        ->Args({std::size_t{1} << 16, std::size_t{1} << 15, 0, 32})
        ->Args({std::size_t{1} << 16, std::size_t{1} << 15, 32, 32});
  }};

  const auto apply_sizes_2{[](benchmark::internal::Benchmark *bench,
                              const std::int64_t minimum_rows = 0,
                              const std::int64_t minimum_columns = 0) {
    for (std::int64_t rows =
             std::max({std::int64_t{1} << 8, minimum_rows, minimum_columns});
         rows <= (std::int64_t{1} << 15); rows *= 2) {
      for (const std::int64_t columns : {rows, rows * 2}) {
        if (rows * columns >= (std::int64_t{1} << 31)) {
          continue;
        }
        bench->Args({rows, columns, 0, 0});
        bench->Args({rows, columns, 0, 32});
      }
    }
  }};

  {
    using block_rows_list = std::integer_sequence<std::uint64_t, 8, 16, 32, 64>;
    using block_columns_list =
        std::integer_sequence<std::uint64_t, 8, 16, 32, 64>;
    for_each_element<block_rows_list, block_columns_list>{
        [apply_sizes]<std::uint64_t block_rows, std::uint64_t block_columns> {
          benchmark::RegisterBenchmark(
              (std::ostringstream{} << "TransposeParallelSVEInRegister<"
                                    << block_rows << ", " << block_columns
                                    << ">, out-of-place")
                  .str(),
              benchmark_transpose_out_of_place_parallel<
                  sventt::TransposeParallelSVEInRegister<block_rows,
                                                         block_columns>>)
              ->Apply(apply_sizes);
        }};
  }

  {
    using block_rows_list = std::integer_sequence<
        std::uint64_t, std::uint64_t{1} << 3, std::uint64_t{1} << 4,
        std::uint64_t{1} << 5, std::uint64_t{1} << 6, std::uint64_t{1} << 7,
        std::uint64_t{1} << 8, std::uint64_t{1} << 9, std::uint64_t{1} << 10,
        std::uint64_t{1} << 11, std::uint64_t{1} << 12, std::uint64_t{1} << 13,
        std::uint64_t{1} << 14>;
    using block_columns_list = std::integer_sequence<
        std::uint64_t, std::uint64_t{1} << 3, std::uint64_t{1} << 4,
        std::uint64_t{1} << 5, std::uint64_t{1} << 6, std::uint64_t{1} << 7,
        std::uint64_t{1} << 8, std::uint64_t{1} << 9, std::uint64_t{1} << 10,
        std::uint64_t{1} << 11, std::uint64_t{1} << 12, std::uint64_t{1} << 13,
        std::uint64_t{1} << 14>;
    using num_shuffle_stages_list =
        std::integer_sequence<std::uint64_t, 0, 1, 2, 3>;
    for_each_element<block_rows_list, block_columns_list,
                     num_shuffle_stages_list>{
        [apply_sizes_2]<std::uint64_t block_rows, std::uint64_t block_columns,
                        std::uint64_t num_shuffle_stages> {
          if constexpr (block_rows * block_columns <=
                        (std::uint64_t{1} << 17)) {
            benchmark::internal::Benchmark *const bench{
                benchmark::RegisterBenchmark(
                    (std::ostringstream{}
                     << "TransposeParallelSVEInRegisterRowFirst<" << block_rows
                     << ", " << block_columns << ", " << num_shuffle_stages
                     << ">, out-of-place")
                        .str(),
                    benchmark_transpose_out_of_place_parallel<
                        sventt::TransposeParallelSVEInRegisterRowFirst<
                            block_rows, block_columns, num_shuffle_stages>>)};
            apply_sizes_2(bench, block_rows, block_columns);
          }
        }};
  }

  {
    using block_rows_list = std::integer_sequence<
        std::uint64_t, std::uint64_t{1} << 3, std::uint64_t{1} << 4,
        std::uint64_t{1} << 5, std::uint64_t{1} << 6, std::uint64_t{1} << 7,
        std::uint64_t{1} << 8, std::uint64_t{1} << 9, std::uint64_t{1} << 10,
        std::uint64_t{1} << 11, std::uint64_t{1} << 12, std::uint64_t{1} << 13,
        std::uint64_t{1} << 14>;
    using block_columns_list = std::integer_sequence<
        std::uint64_t, std::uint64_t{1} << 3, std::uint64_t{1} << 4,
        std::uint64_t{1} << 5, std::uint64_t{1} << 6, std::uint64_t{1} << 7,
        std::uint64_t{1} << 8, std::uint64_t{1} << 9, std::uint64_t{1} << 10,
        std::uint64_t{1} << 11, std::uint64_t{1} << 12, std::uint64_t{1} << 13,
        std::uint64_t{1} << 14>;
    using block_padding_list = std::integer_sequence<std::uint64_t, 0, 32>;
    using num_shuffle_stages_list =
        std::integer_sequence<std::uint64_t, 0, 1, 2, 3>;
    for_each_element<
        block_rows_list, block_columns_list, block_padding_list,
        num_shuffle_stages_list>{[apply_sizes_2]<
                                     std::uint64_t block_rows,
                                     std::uint64_t block_columns,
                                     std::uint64_t block_padding,
                                     std::uint64_t num_shuffle_stages> {
      if constexpr (block_rows * block_columns <= (std::uint64_t{1} << 17)) {
        benchmark::internal::Benchmark *const bench{
            benchmark::RegisterBenchmark(
                (std::ostringstream{}
                 << "TransposeParallelSVEInRegisterExplicitBlockingRowFirst<"
                 << block_rows << ", " << block_columns << ", "
                 << block_columns + block_padding << ", " << num_shuffle_stages
                 << ">, out-of-place")
                    .str(),
                benchmark_transpose_out_of_place_parallel<
                    sventt::
                        TransposeParallelSVEInRegisterExplicitBlockingRowFirst<
                            block_rows, block_columns,
                            block_columns + block_padding,
                            num_shuffle_stages>>)};
        apply_sizes_2(bench, block_rows, block_columns);
      }
    }};
  }

  {
    using block_rows_list = std::integer_sequence<
        std::uint64_t, std::uint64_t{1} << 3, std::uint64_t{1} << 4,
        std::uint64_t{1} << 5, std::uint64_t{1} << 6, std::uint64_t{1} << 7,
        std::uint64_t{1} << 8, std::uint64_t{1} << 9, std::uint64_t{1} << 10,
        std::uint64_t{1} << 11, std::uint64_t{1} << 12, std::uint64_t{1} << 13,
        std::uint64_t{1} << 14>;
    using block_columns_list = std::integer_sequence<
        std::uint64_t, std::uint64_t{1} << 3, std::uint64_t{1} << 4,
        std::uint64_t{1} << 5, std::uint64_t{1} << 6, std::uint64_t{1} << 7,
        std::uint64_t{1} << 8, std::uint64_t{1} << 9, std::uint64_t{1} << 10,
        std::uint64_t{1} << 11, std::uint64_t{1} << 12, std::uint64_t{1} << 13,
        std::uint64_t{1} << 14>;
    using block_padding_list = std::integer_sequence<std::uint64_t, 0, 32>;
    using num_shuffle_stages_list =
        std::integer_sequence<std::uint64_t, 0, 1, 2, 3>;
    for_each_element<
        block_rows_list, block_columns_list, block_padding_list,
        num_shuffle_stages_list>{[apply_sizes_2]<
                                     std::uint64_t block_rows,
                                     std::uint64_t block_columns,
                                     std::uint64_t block_padding,
                                     std::uint64_t num_shuffle_stages> {
      if constexpr (block_rows * block_columns <= (std::uint64_t{1} << 17)) {
        benchmark::internal::Benchmark *const bench{
            benchmark::RegisterBenchmark(
                (std::ostringstream{}
                 << "TransposeParallelSVEInRegisterFullBlockingRowFirst<"
                 << block_rows << ", " << block_columns << ", "
                 << block_columns + block_padding << ", "
                 << block_rows + block_padding << ", " << num_shuffle_stages
                 << ">, out-of-place")
                    .str(),
                benchmark_transpose_out_of_place_parallel<
                    sventt::TransposeParallelSVEInRegisterFullBlockingRowFirst<
                        block_rows, block_columns,
                        block_columns + block_padding,
                        block_rows + block_padding, num_shuffle_stages>>)};
        apply_sizes_2(bench, block_rows, block_columns);
      }
    }};
  }

  {
    using block_rows_list =
        std::integer_sequence<std::uint64_t, 8, 16, 32, 64, 128>;
    using block_columns_list =
        std::integer_sequence<std::uint64_t, 1, 2, 4, 8, 16, 32>;
    for_each_element<block_rows_list, block_columns_list>{
        [apply_sizes]<std::uint64_t block_rows, std::uint64_t block_columns> {
          benchmark::RegisterBenchmark(
              (std::ostringstream{} << "TransposeParallelSVEGatherRowFirst<"
                                    << block_rows << ", " << block_columns
                                    << ">, out-of-place")
                  .str(),
              benchmark_transpose_out_of_place_parallel<
                  sventt::TransposeParallelSVEGatherRowFirst<block_rows,
                                                             block_columns>>)
              ->Apply(apply_sizes);
        }};
  }

  {
    using block_rows_list =
        std::integer_sequence<std::uint64_t, 8, 16, 32, 64, 128>;
    using block_columns_list =
        std::integer_sequence<std::uint64_t, 1, 2, 4, 8, 16, 32>;
    for_each_element<block_rows_list, block_columns_list>{
        [apply_sizes]<std::uint64_t block_rows, std::uint64_t block_columns> {
          benchmark::RegisterBenchmark(
              (std::ostringstream{}
               << "TransposeParallelSVEGatherVectorIndexRowFirst<" << block_rows
               << ", " << block_columns << ">, out-of-place")
                  .str(),
              benchmark_transpose_out_of_place_parallel<
                  sventt::TransposeParallelSVEGatherVectorIndexRowFirst<
                      block_rows, block_columns>>)
              ->Apply(apply_sizes);
        }};
  }

  {
    using block_rows_list =
        std::integer_sequence<std::uint64_t, 8, 16, 32, 64, 128>;
    using block_columns_list = std::integer_sequence<std::uint64_t, 1, 2, 4, 8>;
    for_each_element<
        block_rows_list,
        block_columns_list>{[apply_sizes]<std::uint64_t block_rows,
                                          std::uint64_t block_columns> {
      benchmark::RegisterBenchmark(
          (std::ostringstream{}
           << "TransposeParallelSVEGatherCombinedColumnVectorIndexRowFirst<"
           << block_rows << ", " << block_columns << ">, out-of-place")
              .str(),
          benchmark_transpose_out_of_place_parallel<
              sventt::
                  TransposeParallelSVEGatherCombinedColumnVectorIndexRowFirst<
                      block_rows, block_columns>>)
          ->Apply(apply_sizes);
    }};
  }

  {
    using block_rows_list =
        std::integer_sequence<std::uint64_t, 8, 16, 32, 64, 128>;
    using block_columns_list =
        std::integer_sequence<std::uint64_t, 1, 2, 4, 8, 16, 32>;
    for_each_element<block_rows_list, block_columns_list>{
        [apply_sizes]<std::uint64_t block_rows, std::uint64_t block_columns> {
          benchmark::RegisterBenchmark(
              (std::ostringstream{} << "TransposeParallelSVEGatherColumnFirst<"
                                    << block_rows << ", " << block_columns
                                    << ">, out-of-place")
                  .str(),
              benchmark_transpose_out_of_place_parallel<
                  sventt::TransposeParallelSVEGatherColumnFirst<block_rows,
                                                                block_columns>>)
              ->Apply(apply_sizes);
        }};
  }

  {
    using block_rows_list =
        std::integer_sequence<std::uint64_t, 8, 16, 32, 64, 128>;
    using block_columns_list =
        std::integer_sequence<std::uint64_t, 1, 2, 4, 8, 16, 32>;
    for_each_element<block_rows_list, block_columns_list>{
        [apply_sizes]<std::uint64_t block_rows, std::uint64_t block_columns> {
          benchmark::RegisterBenchmark(
              (std::ostringstream{}
               << "TransposeParallelSVEGatherVectorIndexColumnFirst<"
               << block_rows << ", " << block_columns << ">, out-of-place")
                  .str(),
              benchmark_transpose_out_of_place_parallel<
                  sventt::TransposeParallelSVEGatherVectorIndexColumnFirst<
                      block_rows, block_columns>>)
              ->Apply(apply_sizes);
        }};
  }
}

static void register_in_place(void) {
  const auto apply_sizes{[](benchmark::internal::Benchmark *bench) {
    bench->RangeMultiplier(2)->Range(std::size_t{1} << 8, std::size_t{1} << 15);
  }};

  {
    using block_dim_list = std::integer_sequence<std::uint64_t, 8, 16, 32, 64>;
    for_each_element<block_dim_list>{[apply_sizes]<std::uint64_t block_dim> {
      benchmark::RegisterBenchmark(
          (std::ostringstream{} << "TransposeParallelSVEInRegister<"
                                << block_dim << ", " << block_dim
                                << ">, in-place")
              .str(),
          benchmark_transpose_in_place_parallel<
              sventt::TransposeParallelSVEInRegister<block_dim, block_dim>>)
          ->Apply(apply_sizes);
    }};
  }

  {
    using block_dim_list =
        std::integer_sequence<std::uint64_t, std::uint64_t{1} << 3,
                              std::uint64_t{1} << 4, std::uint64_t{1} << 5,
                              std::uint64_t{1} << 6, std::uint64_t{1} << 7,
                              std::uint64_t{1} << 8, std::uint64_t{1} << 9>;
    using num_shuffle_stages_list =
        std::integer_sequence<std::uint64_t, 0, 1, 2, 3>;
    for_each_element<block_dim_list, num_shuffle_stages_list>{
        [apply_sizes]<std::uint64_t block_dim,
                      std::uint64_t num_shuffle_stages> {
          benchmark::RegisterBenchmark(
              (std::ostringstream{} << "TransposeParallelSVEInRegisterRowFirst<"
                                    << block_dim << ", " << block_dim << ", "
                                    << num_shuffle_stages << ">, in-place")
                  .str(),
              benchmark_transpose_in_place_parallel<
                  sventt::TransposeParallelSVEInRegisterRowFirst<
                      block_dim, block_dim, num_shuffle_stages>>)
              ->RangeMultiplier(2)
              ->Range(std::max(std::size_t{1} << 8, block_dim),
                      std::size_t{1} << 15);
        }};
  }

  {
    using block_dim_list =
        std::integer_sequence<std::uint64_t, std::uint64_t{1} << 3,
                              std::uint64_t{1} << 4, std::uint64_t{1} << 5,
                              std::uint64_t{1} << 6, std::uint64_t{1} << 7,
                              std::uint64_t{1} << 8, std::uint64_t{1} << 9>;
    using block_padding_list = std::integer_sequence<std::uint64_t, 0, 32>;
    using num_shuffle_stages_list =
        std::integer_sequence<std::uint64_t, 0, 1, 2, 3>;
    for_each_element<block_dim_list, block_padding_list,
                     num_shuffle_stages_list>{
        [apply_sizes]<std::uint64_t block_dim, std::uint64_t block_padding,
                      std::uint64_t num_shuffle_stages> {
          benchmark::RegisterBenchmark(
              (std::ostringstream{}
               << "TransposeParallelSVEInRegisterExplicitBlockingRowFirst<"
               << block_dim << ", " << block_dim << ", "
               << block_dim + block_padding << ", " << num_shuffle_stages
               << ">, in-place")
                  .str(),
              benchmark_transpose_in_place_parallel<
                  sventt::
                      TransposeParallelSVEInRegisterExplicitBlockingRowFirst<
                          block_dim, block_dim, block_dim + block_padding,
                          num_shuffle_stages>>)
              ->RangeMultiplier(2)
              ->Range(std::max(std::size_t{1} << 8, block_dim),
                      std::size_t{1} << 15);
        }};
  }

  {
    using block_dim_list =
        std::integer_sequence<std::uint64_t, std::uint64_t{1} << 3,
                              std::uint64_t{1} << 4, std::uint64_t{1} << 5,
                              std::uint64_t{1} << 6, std::uint64_t{1} << 7,
                              std::uint64_t{1} << 8, std::uint64_t{1} << 9>;
    using block_padding_list = std::integer_sequence<std::uint64_t, 0, 32>;
    using num_shuffle_stages_list =
        std::integer_sequence<std::uint64_t, 0, 1, 2, 3>;
    for_each_element<block_dim_list, block_padding_list,
                     num_shuffle_stages_list>{
        [apply_sizes]<std::uint64_t block_dim, std::uint64_t block_padding,
                      std::uint64_t num_shuffle_stages> {
          benchmark::RegisterBenchmark(
              (std::ostringstream{}
               << "TransposeParallelSVEInRegisterFullBlockingRowFirst<"
               << block_dim << ", " << block_dim << ", "
               << block_dim + block_padding << ", " << block_dim + block_padding
               << ", " << num_shuffle_stages << ">, in-place")
                  .str(),
              benchmark_transpose_in_place_parallel<
                  sventt::TransposeParallelSVEInRegisterFullBlockingRowFirst<
                      block_dim, block_dim, block_dim + block_padding,
                      block_dim + block_padding, num_shuffle_stages>>)
              ->RangeMultiplier(2)
              ->Range(std::max(std::size_t{1} << 8, block_dim),
                      std::size_t{1} << 15);
        }};
  }

  {
    using block_dim_list = std::integer_sequence<std::uint64_t, 8>;
    for_each_element<block_dim_list>{[apply_sizes]<std::uint64_t block_dim> {
      benchmark::RegisterBenchmark(
          (std::ostringstream{}
           << "TransposeParallelSVEGatherVectorIndexRowFirst<" << block_dim
           << ", " << block_dim << ">, in-place")
              .str(),
          benchmark_transpose_in_place_parallel<
              sventt::TransposeParallelSVEGatherVectorIndexRowFirst<block_dim,
                                                                    block_dim>>)
          ->Apply(apply_sizes);
    }};
  }
}

int main(int argc, char *argv[]) {
  register_out_of_place();
  register_in_place();

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}

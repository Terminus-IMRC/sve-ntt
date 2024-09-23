// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#include <cstddef>
#include <cstdint>
#include <vector>

#include <arm_sve.h>

#include <omp.h>

#include <benchmark/benchmark.h>

#include "sventt/sventt.hpp"

#include "utility.hpp"

template <int from_offset>
static void benchmark_reduce_offset(benchmark::State &state) {
  const std::size_t size = state.range(0);
  const std::size_t num_cmgs = omp_get_max_threads();

  sventt::UninitializedVector<std::uint8_t, 65536> buffer(size * num_cmgs);
  std::vector<std::uint8_t *> src(num_cmgs);
#pragma omp parallel
  {
    const std::size_t cmg_num = omp_get_thread_num();
    src[cmg_num] = &buffer.data()[size * cmg_num];
    memset_parallel(src[cmg_num], 0x55, size);
  }

  for (auto _ : state) {
#pragma omp parallel
    {
      const std::size_t cmg_num = omp_get_thread_num();
      const std::size_t from = (cmg_num + num_cmgs + from_offset) % num_cmgs;
      benchmark::DoNotOptimize(src[from]);
      benchmark::DoNotOptimize(
          reduce_parallel(&src[from][0], &src[from][size]));
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(size * num_cmgs * state.iterations());
  state.counters["size"] = size;
}

template <int to_offset>
static void benchmark_memset_offset(benchmark::State &state) {
  const std::size_t size = state.range(0);
  const std::size_t num_cmgs = omp_get_max_threads();

  sventt::UninitializedVector<std::uint8_t, 65536> buffer(size * num_cmgs);
  std::vector<std::uint8_t *> dst(num_cmgs);
#pragma omp parallel
  {
    const std::size_t cmg_num = omp_get_thread_num();
    dst[cmg_num] = &buffer.data()[size * cmg_num];
    memset_parallel(dst[cmg_num], 0x55, size);
  }

  for (auto _ : state) {
#pragma omp parallel
    {
      const std::size_t cmg_num = omp_get_thread_num();
      const std::size_t to = (cmg_num + num_cmgs + to_offset) % num_cmgs;
      benchmark::DoNotOptimize(dst[to]);
      memset_parallel(&dst[to][0], 0xaa, size);
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(size * num_cmgs * state.iterations());
  state.counters["size"] = size;
}

template <int from_offset, int to_offset>
static void benchmark_memcpy_offset(benchmark::State &state) {
  const std::size_t size = state.range(0);
  const std::size_t num_cmgs = omp_get_max_threads();

  sventt::UninitializedVector<std::uint8_t, 65536> buffer(size * 2 * num_cmgs);
  std::vector<std::uint8_t *> src(num_cmgs), dst(num_cmgs);
#pragma omp parallel
  {
    const std::size_t cmg_num = omp_get_thread_num();
    src[cmg_num] = &buffer.data()[size * (cmg_num * 2 + 0)];
    dst[cmg_num] = &buffer.data()[size * (cmg_num * 2 + 1)];
    memset_parallel(src[cmg_num], 0x55, size);
    memset_parallel(dst[cmg_num], 0xaa, size);
  }

  for (auto _ : state) {
#pragma omp parallel
    {
      const std::size_t cmg_num = omp_get_thread_num();
      const std::size_t from = (cmg_num + num_cmgs + from_offset) % num_cmgs;
      const std::size_t to = (cmg_num + num_cmgs + to_offset) % num_cmgs;
      benchmark::DoNotOptimize(src[from]);
      benchmark::DoNotOptimize(dst[to]);
      memcpy_parallel(dst[to], src[from], size);
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(size * num_cmgs * state.iterations());
  state.counters["size"] = size;
}

template <int from0_offset, int from1_offset, int from2_offset,
          int from3_offset, int to0_offset, int to1_offset, int to2_offset,
          int to3_offset>
static void benchmark_memcpy4_offset(benchmark::State &state) {
  const std::size_t size = state.range(0);
  const std::size_t num_cmgs = omp_get_max_threads();

  sventt::UninitializedVector<std::uint8_t, 65536> buffer(size * 2 * num_cmgs);
  std::vector<std::uint8_t *> src(num_cmgs), dst(num_cmgs);
#pragma omp parallel
  {
    const std::size_t cmg_num = omp_get_thread_num();
    src[cmg_num] = &buffer.data()[size * (cmg_num * 2 + 0)];
    dst[cmg_num] = &buffer.data()[size * (cmg_num * 2 + 1)];
    memset_parallel(src[cmg_num], 0x55, size);
    memset_parallel(dst[cmg_num], 0xaa, size);
  }

  for (auto _ : state) {
#pragma omp parallel
    {
      for (std::size_t i = 0; i < num_cmgs; ++i) {
        benchmark::DoNotOptimize(src[i]);
        benchmark::DoNotOptimize(dst[i]);
      }

      const std::size_t cmg_num = omp_get_thread_num();
      const std::size_t from0 = (cmg_num + num_cmgs + from0_offset) % num_cmgs;
      const std::size_t from1 = (cmg_num + num_cmgs + from1_offset) % num_cmgs;
      const std::size_t from2 = (cmg_num + num_cmgs + from2_offset) % num_cmgs;
      const std::size_t from3 = (cmg_num + num_cmgs + from3_offset) % num_cmgs;
      const std::size_t to0 = (cmg_num + num_cmgs + to0_offset) % num_cmgs;
      const std::size_t to1 = (cmg_num + num_cmgs + to1_offset) % num_cmgs;
      const std::size_t to2 = (cmg_num + num_cmgs + to2_offset) % num_cmgs;
      const std::size_t to3 = (cmg_num + num_cmgs + to3_offset) % num_cmgs;

#pragma omp parallel
      {
        const svbool_t ptrue{svptrue_b8()};
        const std::size_t cnt = svcntb();

#pragma omp for simd
        for (std::size_t i = 0; i < size; i += cnt * 2) {
          const svuint8_t x0 = svld1(ptrue, &src[from0][i + cnt * 0]);
          const svuint8_t x1 = svld1(ptrue, &src[from1][i + cnt * 0]);
          const svuint8_t x2 = svld1(ptrue, &src[from2][i + cnt * 0]);
          const svuint8_t x3 = svld1(ptrue, &src[from3][i + cnt * 0]);
          const svuint8_t x4 = svld1(ptrue, &src[from0][i + cnt * 1]);
          const svuint8_t x5 = svld1(ptrue, &src[from1][i + cnt * 1]);
          const svuint8_t x6 = svld1(ptrue, &src[from2][i + cnt * 1]);
          const svuint8_t x7 = svld1(ptrue, &src[from3][i + cnt * 1]);
          svst1(ptrue, &dst[to0][i + cnt * 0], x0);
          svst1(ptrue, &dst[to1][i + cnt * 0], x1);
          svst1(ptrue, &dst[to2][i + cnt * 0], x2);
          svst1(ptrue, &dst[to3][i + cnt * 0], x3);
          svst1(ptrue, &dst[to0][i + cnt * 1], x4);
          svst1(ptrue, &dst[to1][i + cnt * 1], x5);
          svst1(ptrue, &dst[to2][i + cnt * 1], x6);
          svst1(ptrue, &dst[to3][i + cnt * 1], x7);
        }
      }
    }
    benchmark::ClobberMemory();
  }

  state.SetBytesProcessed(size * num_cmgs * num_cmgs * state.iterations());
  state.counters["size"] = size;
}

/*
 * On A64FX, invoke this benchmark with these environment variables:
 * - OMP_NUM_THREADS=4,12
 * - OMP_PLACES=numa_domains
 * - OMP_PROC_BIND=spread,close
 */

BENCHMARK(benchmark_reduce_offset<0>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 32);
BENCHMARK(benchmark_reduce_offset<1>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 32);
BENCHMARK(benchmark_reduce_offset<2>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 32);
BENCHMARK(benchmark_reduce_offset<3>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 32);

BENCHMARK(benchmark_memset_offset<0>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 32);
BENCHMARK(benchmark_memset_offset<1>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 32);
BENCHMARK(benchmark_memset_offset<2>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 32);
BENCHMARK(benchmark_memset_offset<3>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 32);

BENCHMARK(benchmark_memcpy_offset<0, 0>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<0, 1>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<0, 2>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<0, 3>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<1, 0>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<1, 1>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<1, 2>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<1, 3>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<2, 0>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<2, 1>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<2, 2>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<2, 3>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<3, 0>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<3, 1>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<3, 2>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy_offset<3, 3>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);

BENCHMARK(benchmark_memcpy4_offset<0, 0, 0, 0, 0, 0, 0, 0>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy4_offset<0, 0, 0, 0, 1, 1, 1, 1>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy4_offset<1, 1, 1, 1, 0, 0, 0, 0>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy4_offset<0, 0, 0, 0, 0, 1, 2, 3>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy4_offset<0, 1, 2, 3, 0, 0, 0, 0>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy4_offset<0, 1, 2, 3, 1, 2, 3, 0>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy4_offset<1, 2, 3, 0, 0, 1, 2, 3>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy4_offset<0, 1, 2, 3, 2, 3, 0, 1>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);
BENCHMARK(benchmark_memcpy4_offset<2, 3, 0, 1, 0, 1, 2, 3>)
    ->RangeMultiplier(2)
    ->Range(std::size_t{1} << 16, std::size_t{1} << 31);

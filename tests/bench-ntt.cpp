// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#include <sventt/sventt.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "ntt-reference.hpp"
#include "utility.hpp"

template <class kernel_type, bool is_inverse>
static void benchmark_ntt(benchmark::State &state) {
  using ntt_type = sventt::NTT<kernel_type>;
  using modulus_type = ntt_type::modulus_type;
  const std::uint64_t m{ntt_type::get_m()};

  std::default_random_engine gen;

  sventt::PageMemory<std::uint64_t> buffer{m * 3, false};
  std::uint64_t *src{&buffer[m * 0]}, *dst{&buffer[m * 1]},
      *dst_ref{&buffer[m * 2]};
  iota_parallel(&src[0], &src[m],
                std::uniform_int_distribution<std::uint64_t>{
                    std::uint64_t{}, modulus_type::get_modulus() - m - 1}(gen));
  memset_parallel(dst, 0x55, sizeof(std::uint64_t) * m);
  memset_parallel(dst_ref, 0xaa, sizeof(std::uint64_t) * m);

  const NTTReference ntt_ref{m, modulus_type::get_modulus(),
                             modulus_type::get_generator()};
  if constexpr (is_inverse) {
    ntt_ref.compute_inverse(dst_ref, src);
  } else {
    ntt_ref.compute_forward(dst_ref, src);
  }

  const ntt_type ntt{!is_inverse, is_inverse, false};

  for (auto _ : state) {
    benchmark::DoNotOptimize(src);
    benchmark::DoNotOptimize(dst);
    if constexpr (is_inverse) {
      ntt.compute_inverse(dst, src);
    } else {
      ntt.compute_forward(dst, src);
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["m"] = m;

  for (std::uint64_t i{}; i < m; ++i) {
    if (dst[i] % modulus_type::get_modulus() != dst_ref[i]) {
      throw std::runtime_error{"Mismatch at index " + std::to_string(i)};
    }
  }
}

int main(int argc, char *argv[]) {
  {
#include NTT_TEST_CASE_FILE

    benchmark::RegisterBenchmark("Forward, " + name,
                                 benchmark_ntt<kernel_type, false>);
    benchmark::RegisterBenchmark("Inverse, " + name,
                                 benchmark_ntt<kernel_type, true>);
  }

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    throw std::invalid_argument{"Unrecognized arguments found"};
  }
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}

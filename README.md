# [`sve-ntt`](https://github.com/Terminus-IMRC/sve-ntt)

This repository provides a multi-threaded implementation of the **number
theoretic transform (NTT)** using the **Arm Scalable Vector Extension (SVE)**
instruction set.

This library is designed to be flexible and efficient, implementing iterative,
recursive, six-step, and blocked six-step NTT algorithms.
The underlying operations (butterfly operation, inner NTT, etc.) and parameters
(unroll amount, cache block and padding size, etc.) can be customized via C++
template parameters, allowing for different configurations and optimizations.
For example, you can compute NTT using the blocked six-step algorithm with:
```c++
#include <sventt/sventt.hpp>

using namespace sventt;

// Adopt 2^64 - 1827 * 2^32 + 1 as the modulus and 3 as the generator.
using modulus_type = Modulus<UINT64_C(0xffff'fc6e'8000'0001), 3>;

// Adopt the Montgomery algorithm for modular multiplication.
using modmul_type = PAdic64SVE<modulus_type>;

// Adopt the SVE shuffle-based transposition with L1 cache blocking (32-by-128
// with padding) in the six-step NTT.
using transposition_type =
    TransposeParallelSVEInRegisterExplicitBlockingRowFirst<32, 128, 128 + 32, 3>;

// Set the transform size to 2^17, and decompose it as 2^8 * 2^9.
constexpr uint64_t n  = uint64_t(1) << 17,
                   n0 = uint64_t(1) <<  8,
                   n1 = uint64_t(1) <<  9;

// Adopt the mixed-radix (8 * 8 * 2) iterative NTT for the first inner NTT
// stage.
using ntt0_type =
    IterativeNTT<
        modulus_type, n0,
        RadixEightSVELayer<modmul_type, n0, n0     >,
        RadixEightSVELayer<modmul_type, n0, n0 >> 3>,
        RadixFourSVELayer <modmul_type, n0, n0 >> 6>
    >;

// Adopt the radix-8 recursive NTT followed by the radix-8 (8 * 8) iterative NTT
// for the second inner NTT.
using ntt1_type =
    RecursiveNTT<
        modulus_type, n1,
        RadixEightSVELayer<modmul_type, n1, n1>,
        IterativeNTT<
            modulus_type, n1,
            RadixEightSVELayer<modmul_type, n1, n1 >> 3>,
            RadixEightSVELayer<modmul_type, n1, n1 >> 6>
        >,
        false
    >;

// Adopt the blocked six-step NTT using the above inner NTTs and transposition.
using ntt_type =
    RecursiveNTT<
        modulus_type, n,
        BlockedGenericSVELayer<
            modmul_type, n,
            ntt0_type, 32, 2, 128,
            transposition_type
        >,
        ntt1_type, true
    >;

// Instantiate the NTT, which involves precomputations.
ntt_type ntt;

// Prepare the data used for input and output.
// A usual std::vector can also be used, but by the PageMemory class a
// page-aligned memory is allocated, which is preferable for performance.
// The second parameter is whether to use huge pages or not.
PageMemory<std::uint64_t> a(n, true);
initialize_with_some_values(a);

// Compute the forward NTT in-place.
ntt.compute_forward(a.data());
```
The documentation for the parameters is currently not available, but thanks to
the modular design, you can easily refer to each class for the information.

For the optimizations applied in this implementation, refer to the paper *An
Improved Implementation of Multi-Threaded Number Theoretic Transform Using Arm
Scalable Vector Extension Instruction Set* by Yukimasa Sugizaki and Daisuke
Takahashi, presented at the [International Symposium on Parallel and Distributed
Computing (ISPDC) 2025](https://ispdc2025.inria.fr/) (to appear in proceedings).


## Building and running the tests and benchmark

The source code is written in the C++ language (with C++20 features) and is
built using the CMake build system.
Note that the library itself is header-only, so you do not need to build it
unless you want to run the tests or benchmarks.

The tests and benchmarks use the [Google
Test](https://github.com/google/googletest) and
[Benchmark](https://github.com/google/benchmark) libraries.
On Ubuntu, you can install the required packages with the following commands:

```console
$ sudo apt update
$ sudo apt install build-essential cmake libgtest-dev libbenchmark-dev
```

You need first to configure the build system.
Since the SVE is not *completely* scalable, some parameters such as the number
of shuffle stages vary depending on the vector length; thus, the vector length
needs to be fixed at compile time.
This library currently supports 128-bit, 256-bit, and 512-bit SVE units, which
covers all the processors publicly available (as of July 2025).

You can specify this using the `-msve-vector-bits=<bits>` compiler option, where
`<bits>` is the vector length in bits.
To obtain the best performance, you also need to enable optimizations towards
your target processor using the `-mcpu=<uarch>`, `-mtune=<uarch>`, and `-Ofast`
compiler options, where `<uarch>` is the target microarchitecture.
The example combinations of these values are as follows:

| Microarchitecture | Implemented/used in | `<uarch>` | `<bits>` |
| -- | -- | -- | -- |
| Fujitsu A64FX | Fugaku (RIKEN), Ookami (Stony Brook University) | `a64fx` | `512` |
| Arm Neoverse V1 | AWS Graviton3/3E | `neoverse-v1` | `256` |
| Arm Neoverse V2 | AWS Graviton4, Google Axion, NVIDIA Grace | `neoverse-v2` | `128` |
| Arm Neoverse V3 | NVIDIA Thor | `neoverse-v3` | `128` |
| Arm Neoverse N2 | Alibaba Yitian 710, Azure Cobalt 100 | `neoverse-n2` | `128` |

You can specify these options on build configuration.
For example, to build and run the tests and benchmarks for the Fujitsu A64FX
processor, run the following commands:

```console
$ cmake . -D CMAKE_CXX_FLAGS='-msve-vector-bits=512 -mcpu=a64fx -mtune=a64fx -Ofast'
$ cmake --build .
$ cmake --build . --target test
```


## License and contribution

For license and copyright notices, see the SPDX file tags in each file.
Unless otherwise noted, files in this project are licensed under the Apache
License, Version 2.0 (SPDX short-form identifier: Apache-2.0) and copyrighted by
the contributors.

Everyone is encouraged to contribute to this project.
See the [CONTRIBUTING.md](CONTRIBUTING.md) file for instructions.

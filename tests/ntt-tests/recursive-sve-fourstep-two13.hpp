// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

using modulus_type =
    sventt::Modulus<UINT64_C(0x3a00'0000'0000'0001), UINT64_C(3)>;

using modmul_type = sventt::PAdic64SVE<modulus_type>;

constexpr std::uint64_t m{std::uint64_t{1} << 15};

using inner_column_kernel_type = sventt::IterativeNTT<
    modulus_type, std::uint64_t{1} << 9,
    sventt::RadixEightSVELayer<modmul_type, std::uint64_t{1} << 9,
                               std::uint64_t{1} << 9>,
    sventt::RadixFourSVELayer<modmul_type, std::uint64_t{1} << 9,
                              std::uint64_t{1} << 6>,
    sventt::RadixTwoSVELayer<modmul_type, std::uint64_t{1} << 9,
                             std::uint64_t{1} << 4>,
    sventt::RadixEightSVELayer<modmul_type, std::uint64_t{1} << 9,
                               std::uint64_t{1} << 3, m>>;

using inner_row_kernel_type = sventt::IterativeNTT<
    modulus_type, std::uint64_t{1} << 6,
    sventt::RadixEightSVELayer<modmul_type, std::uint64_t{1} << 6,
                               std::uint64_t{1} << 6>,
    sventt::RadixEightSVELayer<modmul_type, std::uint64_t{1} << 6,
                               std::uint64_t{1} << 3>>;

using kernel_type = sventt::RecursiveNTT<
    modulus_type, m,
    sventt::GenericSVELayer<modmul_type, m, inner_column_kernel_type, 8, 2,
                            sventt::TransposeParallelSVEInRegister<8, 64>>,
    inner_row_kernel_type, 256, true>;

const std::string name{"recursive, SVE, four-step"};

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

using modulus_type =
    sventt::Modulus<UINT64_C(0x3a00'0000'0000'0001), UINT64_C(3)>;

using modmul_type = sventt::PAdic64SVE<modulus_type>;

constexpr std::uint64_t m{std::uint64_t{1} << 13};

using inner_inner_kernel_type = sventt::IterativeNTT<
    modulus_type, std::uint64_t{1} << 9,
    sventt::RadixEightSVELayer<modmul_type, std::uint64_t{1} << 9,
                               std::uint64_t{1} << 9, 1, true>,
    sventt::RadixFourSVELayer<modmul_type, std::uint64_t{1} << 9,
                              std::uint64_t{1} << 6, 1, true>,
    sventt::RadixTwoSVELayer<modmul_type, std::uint64_t{1} << 9,
                             std::uint64_t{1} << 4, 1, false>,
    sventt::RadixEightSVELayer<modmul_type, std::uint64_t{1} << 9,
                               std::uint64_t{1} << 3, m, false>>;

using inner_kernel_type = sventt::RecursiveNTT<
    modulus_type, std::uint64_t{1} << 10,
    sventt::RadixTwoSVELayer<modmul_type, std::uint64_t{1} << 10,
                             std::uint64_t{1} << 10, 1, false>,
    inner_inner_kernel_type, 256, false>;

using kernel_type =
    sventt::RecursiveNTT<modulus_type, m,
                         sventt::RadixEightSVELayer<modmul_type, m, m, 1, true>,
                         inner_kernel_type, 256, false>;

const std::string name{"recursive, SVE, radix-2,4,8"};

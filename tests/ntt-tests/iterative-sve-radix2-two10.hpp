// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

using modulus_type =
    sventt::Modulus<UINT64_C(0x3a00'0000'0000'0001), UINT64_C(3)>;

using modmul_type = sventt::PAdic64SVE<modulus_type>;

constexpr std::uint64_t m{std::uint64_t{1} << 10};

using kernel_type = sventt::IterativeNTT<
    modulus_type, m,
    sventt::RadixTwoSVELayer<modmul_type, m, std::uint64_t{1} << 10, 1, true>,
    sventt::RadixTwoSVELayer<modmul_type, m, std::uint64_t{1} << 9, 1, false>,
    sventt::RadixTwoSVELayer<modmul_type, m, std::uint64_t{1} << 8, 1, true>,
    sventt::RadixTwoSVELayer<modmul_type, m, std::uint64_t{1} << 7, 1, false>,
    sventt::RadixTwoSVELayer<modmul_type, m, std::uint64_t{1} << 6, 1, false>,
    sventt::RadixTwoSVELayer<modmul_type, m, std::uint64_t{1} << 5, 1, false>,
    sventt::RadixTwoSVELayer<modmul_type, m, std::uint64_t{1} << 4, 1, true>,
    sventt::RadixTwoSVELayer<modmul_type, m, std::uint64_t{1} << 3, 1, true>,
    sventt::RadixTwoSVELayer<modmul_type, m, std::uint64_t{1} << 2, 1, true>,
    sventt::RadixTwoSVELayer<modmul_type, m, std::uint64_t{1} << 1, m, false>>;

const std::string name{"iterative, SVE, radix-2"};

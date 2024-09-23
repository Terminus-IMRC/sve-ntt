// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

using modulus_type =
    sventt::Modulus<UINT64_C(0x3a00'0000'0000'0001), UINT64_C(3)>;

constexpr std::uint64_t m{std::uint64_t{1} << 13};

using inner_inner_kernel_type = sventt::IterativeNTT<
    modulus_type, std::uint64_t{1} << 9,
    sventt::RadixEightScalarLayer<sventt::PAdic64Scalar<modulus_type>,
                                  std::uint64_t{1} << 9, std::uint64_t{1} << 9>,
    sventt::RadixFourScalarLayer<sventt::FixedPoint64Scalar<modulus_type>,
                                 std::uint64_t{1} << 9, std::uint64_t{1} << 6>,
    sventt::RadixTwoScalarLayer<sventt::PAdic64Scalar<modulus_type>,
                                std::uint64_t{1} << 9, std::uint64_t{1} << 4>,
    sventt::RadixEightScalarLayer<sventt::FixedPoint64Scalar<modulus_type>,
                                  std::uint64_t{1} << 9, std::uint64_t{1} << 3,
                                  m>>;

using inner_kernel_type = sventt::RecursiveNTT<
    modulus_type, std::uint64_t{1} << 10,
    sventt::RadixTwoScalarLayer<sventt::FixedPoint64Scalar<modulus_type>,
                                std::uint64_t{1} << 10, std::uint64_t{1} << 10>,
    inner_inner_kernel_type>;

using kernel_type = sventt::RecursiveNTT<
    modulus_type, m,
    sventt::RadixEightScalarLayer<sventt::PAdic64Scalar<modulus_type>, m, m>,
    inner_kernel_type>;

const std::string name{"recursive, scalar, radix-2,4,8"};

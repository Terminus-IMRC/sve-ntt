// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

using modulus_type =
    sventt::Modulus<UINT64_C(0x3a00'0000'0000'0001), UINT64_C(3)>;

constexpr std::uint64_t m{std::uint64_t{1} << 10};

using kernel_type = sventt::IterativeNTT<
    modulus_type, m,
    sventt::RadixTwoScalarLayer<sventt::FixedPoint64Scalar<modulus_type>, m,
                                std::uint64_t{1} << 10>,
    sventt::RadixTwoScalarLayer<sventt::FixedPoint64Scalar<modulus_type>, m,
                                std::uint64_t{1} << 9>,
    sventt::RadixTwoScalarLayer<sventt::PAdic64Scalar<modulus_type>, m,
                                std::uint64_t{1} << 8>,
    sventt::RadixTwoScalarLayer<sventt::FixedPoint64Scalar<modulus_type>, m,
                                std::uint64_t{1} << 7>,
    sventt::RadixTwoScalarLayer<sventt::PAdic64Scalar<modulus_type>, m,
                                std::uint64_t{1} << 6>,
    sventt::RadixTwoScalarLayer<sventt::PAdic64Scalar<modulus_type>, m,
                                std::uint64_t{1} << 5>,
    sventt::RadixTwoScalarLayer<sventt::PAdic64Scalar<modulus_type>, m,
                                std::uint64_t{1} << 4>,
    sventt::RadixTwoScalarLayer<sventt::FixedPoint64Scalar<modulus_type>, m,
                                std::uint64_t{1} << 3>,
    sventt::RadixTwoScalarLayer<sventt::FixedPoint64Scalar<modulus_type>, m,
                                std::uint64_t{1} << 2>,
    sventt::RadixTwoScalarLayer<sventt::FixedPoint64Scalar<modulus_type>, m,
                                std::uint64_t{1} << 1, m>>;

const std::string name{"iterative, scalar, radix-2"};

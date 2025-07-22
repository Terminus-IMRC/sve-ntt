// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include <sventt/sventt.hpp>

#include <boost/multiprecision/cpp_int.hpp>

#include <gtest/gtest.h>

#include "kinnaes.hpp"

template <typename modulus_type> class ModulusFixture : public testing::Test {};

/* List of (m, (N, g), n). */
using ModulusTypes = ::testing::Types<
    /* Order 100 with 64-bit moduli. */
    std::tuple<std::integral_constant<std::uint64_t, 100>,
               sventt::Modulus<UINT64_C(0xfffffffffeca467f), UINT64_C(5)>,
               std::integral_constant<std::uint64_t, 495017>>,
    std::tuple<std::integral_constant<std::uint64_t, 100>,
               sventt::Modulus<UINT64_C(0xfffffffffe05e355), UINT64_C(6)>,
               std::integral_constant<std::uint64_t, 495017>>,

    /* Order 100 with 63-bit moduli. */
    std::tuple<std::integral_constant<std::uint64_t, 100>,
               sventt::Modulus<UINT64_C(0x7ffffffffed59fb5), UINT64_C(2)>,
               std::integral_constant<std::uint64_t, 495017>>,
    std::tuple<std::integral_constant<std::uint64_t, 100>,
               sventt::Modulus<UINT64_C(0x7ffffffffcc4e37f), UINT64_C(13)>,
               std::integral_constant<std::uint64_t, 495017>>,

    /* Order 100 with 62-bit moduli. */
    std::tuple<std::integral_constant<std::uint64_t, 100>,
               sventt::Modulus<UINT64_C(0x3fffffffff4c9937), UINT64_C(5)>,
               std::integral_constant<std::uint64_t, 495017>>,
    std::tuple<std::integral_constant<std::uint64_t, 100>,
               sventt::Modulus<UINT64_C(0x3fffffffff102bef), UINT64_C(3)>,
               std::integral_constant<std::uint64_t, 495017>>,

    /* Order 100 with 61-bit moduli. */
    std::tuple<std::integral_constant<std::uint64_t, 100>,
               sventt::Modulus<UINT64_C(0x1ffffffffff962df), UINT64_C(7)>,
               std::integral_constant<std::uint64_t, 495017>>,
    std::tuple<std::integral_constant<std::uint64_t, 100>,
               sventt::Modulus<UINT64_C(0x1fffffffffdb2c3b), UINT64_C(2)>,
               std::integral_constant<std::uint64_t, 495017>>,

    /* Order 101 with 64-bit to 61-bit moduli. */
    std::tuple<std::integral_constant<std::uint64_t, 101>,
               sventt::Modulus<UINT64_C(0xfffffffffe023ec1), UINT64_C(11)>,
               std::integral_constant<std::uint64_t, 510053>>,
    std::tuple<std::integral_constant<std::uint64_t, 101>,
               sventt::Modulus<UINT64_C(0x7ffffffffd0f0621), UINT64_C(3)>,
               std::integral_constant<std::uint64_t, 510053>>,
    std::tuple<std::integral_constant<std::uint64_t, 101>,
               sventt::Modulus<UINT64_C(0x3ffffffffec5c639), UINT64_C(21)>,
               std::integral_constant<std::uint64_t, 510053>>,
    std::tuple<std::integral_constant<std::uint64_t, 101>,
               sventt::Modulus<UINT64_C(0x1ffffffffdce2e99), UINT64_C(3)>,
               std::integral_constant<std::uint64_t, 510053>>>;

TYPED_TEST_SUITE(ModulusFixture, ModulusTypes);

TYPED_TEST(ModulusFixture, Kinnaes) {
  constexpr std::uint64_t m{std::tuple_element_t<0, TypeParam>()};
  using modulus_type = std::tuple_element_t<1, TypeParam>;
  constexpr std::uint64_t n{std::tuple_element_t<2, TypeParam>()};
  using modmul_type = sventt::PAdic64SVE<modulus_type>;

  using kinnaes_type = MagicSeriesKinnaes<m, modmul_type, n>;

  const std::unordered_map<std::uint64_t, std::string> expected{
      {10, "78132541528"},
      {25, "140170526450793924490478768121814869629364"},
      {35, "13872534241478210358349096341203128450357241660871429860873721318"},
      {42, "1195452957914568544628242649935060977711193839443701120065551521757"
           "686130217168310"},
      {100, "904300736808894426574793302240693911261234942398748154528052171724"
            "305279045583459861011357813556260746366850646669062169890178280824"
            "885995375485156399921958991796250954308603011799192842071430359668"
            "946052264146938445899732873114858199920"},
      {101, "651742868521150599423217738842736563193389672725617304609189541060"
            "948075348430211017087941851686538398290713576362337481621156854784"
            "148283104866179994202618028615736621185423913319338987817995082551"
            "755913561634157004344784632798600635226832"},
  };

  const std::uint64_t count{kinnaes_type::compute()};
  const std::uint64_t expected_count{
      boost::multiprecision::cpp_int(expected.at(m)) %
      modulus_type::get_modulus()};
  EXPECT_EQ(count, expected_count);
}

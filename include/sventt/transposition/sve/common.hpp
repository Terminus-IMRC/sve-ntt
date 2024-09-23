// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_TRANSPOSITION_SVE_COMMON_HPP_INCLUDED
#define SVENTT_TRANSPOSITION_SVE_COMMON_HPP_INCLUDED

#include <cstdint>
#include <stdexcept>

#include <arm_sve.h>

namespace sventt {

inline void transpose_2x2(svuint64_t &x0, svuint64_t &x1) {
  svuint64_t y0, y1;

  y0 = svtrn1(x0, x1);
  y1 = svtrn2(x0, x1);

  x0 = y0;
  x1 = y1;
}

inline void transpose_4x4_lazy(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2,
                               svuint64_t &x3) {
  const svbool_t phalf{svptrue_pat_b64(SV_VL2)};
  svuint64_t y0, y1, y2, y3;

  transpose_2x2(x0, x1);
  transpose_2x2(x2, x3);

  y0 = svsel(phalf, x0, svdupq_lane(x2, 0));
  y1 = svsel(phalf, x1, svdupq_lane(x3, 0));
  y2 = svsel(phalf, svdupq_lane(x0, 1), x2);
  y3 = svsel(phalf, svdupq_lane(x1, 1), x3);

  x0 = y0;
  x1 = y1;
  x2 = y2;
  x3 = y3;
}

inline void transpose_4x4(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2,
                          svuint64_t &x3) {
  const svbool_t ptrue{svptrue_b8()};
  const svbool_t phalf0{svptrue_pat_b64(SV_VL4)};
  const svbool_t phalf1{svnot_z(ptrue, phalf0)};

  alignas(8) const std::uint8_t table[8]{0, 1, 4, 5, 2, 3, 6, 7};
  const svuint64_t table0{svld1ub_u64(ptrue, table)};
  /* 2, 3, 6, 7, 0, 1, 4, 5 */
  const svuint64_t table1{svreinterpret_u64(svext_u8(
      svreinterpret_u8(table0), svreinterpret_u8(table0), 512 / 8 / 2))};

  svuint64_t y0, y1, y2, y3;

  transpose_2x2(x0, x1);
  transpose_2x2(x2, x3);

  x0 = svtbl(x0, table0);
  x1 = svtbl(x1, table0);
  x2 = svtbl(x2, table1);
  x3 = svtbl(x3, table1);

  y0 = svsel(phalf0, x0, x2);
  y1 = svsel(phalf0, x1, x3);
  y2 = svsel(phalf1, x0, x2);
  y3 = svsel(phalf1, x1, x3);

  x0 = y0;
  x1 = y1;
  x2 = y2;
  x3 = y3;
}

inline void transpose_8x8(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2,
                          svuint64_t &x3, svuint64_t &x4, svuint64_t &x5,
                          svuint64_t &x6, svuint64_t &x7) {
  svuint64_t y0, y1, y2, y3, y4, y5, y6, y7;
  y0 = svuzp1(x0, x1);
  y1 = svuzp2(x0, x1);
  y2 = svuzp1(x2, x3);
  y3 = svuzp2(x2, x3);
  y4 = svuzp1(x4, x5);
  y5 = svuzp2(x4, x5);
  y6 = svuzp1(x6, x7);
  y7 = svuzp2(x6, x7);

  x0 = svuzp1(y0, y2);
  x1 = svuzp1(y1, y3);
  x2 = svuzp2(y0, y2);
  x3 = svuzp2(y1, y3);
  x4 = svuzp1(y4, y6);
  x5 = svuzp1(y5, y7);
  x6 = svuzp2(y4, y6);
  x7 = svuzp2(y5, y7);

  y0 = svuzp1(x0, x4);
  y1 = svuzp1(x1, x5);
  y2 = svuzp1(x2, x6);
  y3 = svuzp1(x3, x7);
  y4 = svuzp2(x0, x4);
  y5 = svuzp2(x1, x5);
  y6 = svuzp2(x2, x6);
  y7 = svuzp2(x3, x7);

  x0 = y0;
  x1 = y1;
  x2 = y2;
  x3 = y3;
  x4 = y4;
  x5 = y5;
  x6 = y6;
  x7 = y7;
}

inline void deinterleave(svuint64_t &x0, svuint64_t &x1) {
  svuint64_t y0, y1;
  y0 = svuzp1(x0, x1);
  y1 = svuzp2(x0, x1);

  x0 = y0;
  x1 = y1;
}

inline void deinterleave(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2,
                         svuint64_t &x3) {
  deinterleave(x0, x1);
  deinterleave(x2, x3);

  deinterleave(x0, x2);
  deinterleave(x1, x3);
}

inline void deinterleave(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2,
                         svuint64_t &x3, svuint64_t &x4, svuint64_t &x5,
                         svuint64_t &x6, svuint64_t &x7) {
  deinterleave(x0, x1, x2, x3);
  deinterleave(x4, x5, x6, x7);

  deinterleave(x0, x4);
  deinterleave(x1, x5);
  deinterleave(x2, x6);
  deinterleave(x3, x7);
}

inline void interleave(svuint64_t &x0, svuint64_t &x1) {
  svuint64_t y0, y1;
  y0 = svzip1(x0, x1);
  y1 = svzip2(x0, x1);

  x0 = y0;
  x1 = y1;
}

inline void interleave(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2,
                       svuint64_t &x3) {
  interleave(x0, x2);
  interleave(x1, x3);

  interleave(x0, x1);
  interleave(x2, x3);
}

inline void interleave(svuint64_t &x0, svuint64_t &x1, svuint64_t &x2,
                       svuint64_t &x3, svuint64_t &x4, svuint64_t &x5,
                       svuint64_t &x6, svuint64_t &x7) {
  interleave(x0, x2, x4, x6);
  interleave(x1, x3, x5, x7);

  interleave(x0, x1);
  interleave(x2, x3);
  interleave(x4, x5);
  interleave(x6, x7);
}

template <std::uint64_t num_shuffle_stages>
inline void load_and_deinterleave(svuint64_t &x0, svuint64_t &x1,
                                  const std::uint64_t *const src) {
  const svbool_t ptrue{svptrue_b8()};

  if constexpr (num_shuffle_stages == 0) {
    svuint64x2_t xx;
    xx = svld2_vnum(ptrue, src, 0);

    x0 = svget2(xx, 0);
    x1 = svget2(xx, 1);
  } else if constexpr (num_shuffle_stages == 1) {
    x0 = svld1_vnum(ptrue, src, 0);
    x1 = svld1_vnum(ptrue, src, 1);

    deinterleave(x0, x1);
  } else {
    throw std::runtime_error{"Unsupported number of shuffle stages"};
  }
}

template <std::uint64_t num_shuffle_stages>
inline void load_and_deinterleave(svuint64_t &x0, svuint64_t &x1,
                                  svuint64_t &x2, svuint64_t &x3,
                                  const std::uint64_t *const src) {
  const svbool_t ptrue{svptrue_b8()};

  if constexpr (num_shuffle_stages == 0) {
    svuint64x4_t xx;
    xx = svld4_vnum(ptrue, src, 0);

    x0 = svget4(xx, 0);
    x1 = svget4(xx, 1);
    x2 = svget4(xx, 2);
    x3 = svget4(xx, 3);
  } else if constexpr (num_shuffle_stages == 1) {
    svuint64x2_t xx0, xx1;
    xx0 = svld2_vnum(ptrue, src, 0);
    xx1 = svld2_vnum(ptrue, src, 2);

    x0 = svget2(xx0, 0);
    x1 = svget2(xx0, 1);
    x2 = svget2(xx1, 0);
    x3 = svget2(xx1, 1);

    deinterleave(x0, x2);
    deinterleave(x1, x3);
  } else if constexpr (num_shuffle_stages == 2) {
    x0 = svld1_vnum(ptrue, src, 0);
    x1 = svld1_vnum(ptrue, src, 1);
    x2 = svld1_vnum(ptrue, src, 2);
    x3 = svld1_vnum(ptrue, src, 3);

    deinterleave(x0, x1, x2, x3);
  } else {
    throw std::runtime_error{"Unsupported number of shuffle stages"};
  }
}

template <std::uint64_t num_shuffle_stages>
inline void load_and_deinterleave(svuint64_t &x0, svuint64_t &x1,
                                  svuint64_t &x2, svuint64_t &x3,
                                  svuint64_t &x4, svuint64_t &x5,
                                  svuint64_t &x6, svuint64_t &x7,
                                  const std::uint64_t *const src) {
  const svbool_t ptrue{svptrue_b8()};

  if constexpr (num_shuffle_stages == 1) {
    svuint64x4_t xx0, xx1;
    xx0 = svld4_vnum(ptrue, src, 0);
    xx1 = svld4_vnum(ptrue, src, 4);

    x0 = svget4(xx0, 0);
    x1 = svget4(xx0, 1);
    x2 = svget4(xx0, 2);
    x3 = svget4(xx0, 3);
    x4 = svget4(xx1, 0);
    x5 = svget4(xx1, 1);
    x6 = svget4(xx1, 2);
    x7 = svget4(xx1, 3);

    deinterleave(x0, x4);
    deinterleave(x1, x5);
    deinterleave(x2, x6);
    deinterleave(x3, x7);
  } else if constexpr (num_shuffle_stages == 2) {
    svuint64x2_t xx0, xx1, xx2, xx3;
    xx0 = svld2_vnum(ptrue, src, 0);
    xx1 = svld2_vnum(ptrue, src, 2);
    xx2 = svld2_vnum(ptrue, src, 4);
    xx3 = svld2_vnum(ptrue, src, 6);

    x0 = svget2(xx0, 0);
    x1 = svget2(xx0, 1);
    x2 = svget2(xx1, 0);
    x3 = svget2(xx1, 1);
    x4 = svget2(xx2, 0);
    x5 = svget2(xx2, 1);
    x6 = svget2(xx3, 0);
    x7 = svget2(xx3, 1);

    deinterleave(x0, x2, x4, x6);
    deinterleave(x1, x3, x5, x7);
  } else if constexpr (num_shuffle_stages == 3) {
    x0 = svld1_vnum(ptrue, src, 0);
    x1 = svld1_vnum(ptrue, src, 1);
    x2 = svld1_vnum(ptrue, src, 2);
    x3 = svld1_vnum(ptrue, src, 3);
    x4 = svld1_vnum(ptrue, src, 4);
    x5 = svld1_vnum(ptrue, src, 5);
    x6 = svld1_vnum(ptrue, src, 6);
    x7 = svld1_vnum(ptrue, src, 7);

    deinterleave(x0, x1, x2, x3, x4, x5, x6, x7);
  } else {
    throw std::runtime_error{"Unsupported number of shuffle stages"};
  }
}

template <std::uint64_t num_shuffle_stages>
inline void interleave_and_store(std::uint64_t *const dst, svuint64_t x0,
                                 svuint64_t x1) {
  const svbool_t ptrue{svptrue_b8()};

  if constexpr (num_shuffle_stages == 0) {
    svst2_vnum(ptrue, dst, 0, svcreate2(x0, x1));
  } else if constexpr (num_shuffle_stages == 1) {
    interleave(x0, x1);

    svst1_vnum(ptrue, dst, 0, x0);
    svst1_vnum(ptrue, dst, 1, x1);
  } else {
    throw std::runtime_error{"Unsupported number of shuffle stages"};
  }
}

template <std::uint64_t num_shuffle_stages>
inline void interleave_and_store(std::uint64_t *const dst, svuint64_t x0,
                                 svuint64_t x1, svuint64_t x2, svuint64_t x3) {
  const svbool_t ptrue{svptrue_b8()};

  if constexpr (num_shuffle_stages == 0) {
    svst4_vnum(ptrue, dst, 0, svcreate4(x0, x1, x2, x3));
  } else if constexpr (num_shuffle_stages == 1) {
    interleave(x0, x2);
    interleave(x1, x3);

    svst2_vnum(ptrue, dst, 0, svcreate2(x0, x1));
    svst2_vnum(ptrue, dst, 2, svcreate2(x2, x3));
  } else if constexpr (num_shuffle_stages == 2) {
    interleave(x0, x1, x2, x3);

    svst1_vnum(ptrue, dst, 0, x0);
    svst1_vnum(ptrue, dst, 1, x1);
    svst1_vnum(ptrue, dst, 2, x2);
    svst1_vnum(ptrue, dst, 3, x3);
  } else {
    throw std::runtime_error{"Unsupported number of shuffle stages"};
  }
}

template <std::uint64_t num_shuffle_stages>
inline void interleave_and_store(std::uint64_t *const dst, svuint64_t x0,
                                 svuint64_t x1, svuint64_t x2, svuint64_t x3,
                                 svuint64_t x4, svuint64_t x5, svuint64_t x6,
                                 svuint64_t x7) {
  const svbool_t ptrue{svptrue_b8()};

  if constexpr (num_shuffle_stages == 1) {
    interleave(x0, x4);
    interleave(x1, x5);
    interleave(x2, x6);
    interleave(x3, x7);

    svst4_vnum(ptrue, dst, 0, svcreate4(x0, x1, x2, x3));
    svst4_vnum(ptrue, dst, 4, svcreate4(x4, x5, x6, x7));
  } else if constexpr (num_shuffle_stages == 2) {
    interleave(x0, x2, x4, x6);
    interleave(x1, x3, x5, x7);

    svst2_vnum(ptrue, dst, 0, svcreate2(x0, x1));
    svst2_vnum(ptrue, dst, 2, svcreate2(x2, x3));
    svst2_vnum(ptrue, dst, 4, svcreate2(x4, x5));
    svst2_vnum(ptrue, dst, 6, svcreate2(x6, x7));
  } else if constexpr (num_shuffle_stages == 3) {
    interleave(x0, x1, x2, x3, x4, x5, x6, x7);

    svst1_vnum(ptrue, dst, 0, x0);
    svst1_vnum(ptrue, dst, 1, x1);
    svst1_vnum(ptrue, dst, 2, x2);
    svst1_vnum(ptrue, dst, 3, x3);
    svst1_vnum(ptrue, dst, 4, x4);
    svst1_vnum(ptrue, dst, 5, x5);
    svst1_vnum(ptrue, dst, 6, x6);
    svst1_vnum(ptrue, dst, 7, x7);
  } else {
    throw std::runtime_error{"Unsupported number of shuffle stages"};
  }
}

} // namespace sventt

#endif /* SVENTT_TRANSPOSITION_SVE_COMMON_HPP_INCLUDED */

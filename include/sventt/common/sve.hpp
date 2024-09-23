// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_COMMON_SVE_HPP_INCLUDED
#define SVENTT_COMMON_SVE_HPP_INCLUDED

#include <cstdint>

#include <arm_sve.h>

namespace sventt {

#if __ARM_FEATURE_SVE_BITS == 0
#error SVE vector length needs to be fixed at compile time
#endif

static constexpr std::uint64_t cntb{__ARM_FEATURE_SVE_BITS / 8},
    cntd{__ARM_FEATURE_SVE_BITS / 64};

static inline void broadcast_and_advance(svuint64_t &x0,
                                         const std::uint64_t *&aux) {
  /* hope this compiles to the LD1RD instruction... */
  x0 = svdup_u64(*aux++);
}

static inline void broadcast_and_advance(svuint64_t &x0, svuint64_t &x1,
                                         const std::uint64_t *&aux) {
  const svbool_t ptrue{svptrue_b8()};

  svuint64_t x;

  if constexpr (cntd == 2) {
    x = svld1(ptrue, aux);
  } else {
    x = svld1rq(ptrue, aux);
  }
  aux += 2;

  x0 = svdup_lane(x, 0);
  x1 = svdup_lane(x, 1);
}

static inline void broadcast_and_advance(svuint64_t &x0, svuint64_t &x1,
                                         svuint64_t &x2,
                                         const std::uint64_t *&aux) {
  const svbool_t ptrue{svptrue_b8()};

  if constexpr (cntd == 2) {
    broadcast_and_advance(x0, x1, aux);
    broadcast_and_advance(x2, aux);
  } else {
    svuint64_t x;

    if constexpr (cntd == 3) {
      x = svld1(ptrue, aux);
    } else {
      x = svld1(svptrue_pat_b64(SV_VL3), aux);
    }
    aux += 3;

    x0 = svdup_lane(x, 0);
    x1 = svdup_lane(x, 1);
  }
}

static inline void broadcast_and_advance(svuint64_t &x0, svuint64_t &x1,
                                         svuint64_t &x2, svuint64_t &x3,
                                         const std::uint64_t *&aux) {
  const svbool_t ptrue{svptrue_b8()};

  static_assert(cntd != 3);
  if constexpr (cntd == 2) {
    broadcast_and_advance(x0, x1, aux);
    broadcast_and_advance(x2, x3, aux);
  } else {
    svuint64_t x;

    if constexpr (cntd == 4) {
      x = svld1(ptrue, aux);
    } else {
      x = svld1(svptrue_pat_b64(SV_VL4), aux);
    }
    aux += 4;

    x0 = svdup_lane(x, 0);
    x1 = svdup_lane(x, 1);
    x2 = svdup_lane(x, 2);
    x3 = svdup_lane(x, 3);
  }
}

static inline void broadcast_and_advance(svuint64_t &x0, svuint64_t &x1,
                                         svuint64_t &x2, svuint64_t &x3,
                                         svuint64_t &x4, svuint64_t &x5,
                                         const std::uint64_t *&aux) {
  const svbool_t ptrue{svptrue_b8()};

  static_assert(cntd != 5);
  if constexpr (cntd <= 4) {
    broadcast_and_advance(x0, x1, x2, x3, aux);
    broadcast_and_advance(x4, x5, aux);
  } else {
    svuint64_t x;

    if constexpr (cntd == 6) {
      x = svld1(ptrue, aux);
    } else {
      x = svld1(svptrue_pat_b64(SV_VL6), aux);
    }
    aux += 6;

    x0 = svdup_lane(x, 0);
    x1 = svdup_lane(x, 1);
    x2 = svdup_lane(x, 2);
    x3 = svdup_lane(x, 3);
    x4 = svdup_lane(x, 4);
    x5 = svdup_lane(x, 5);
  }
}

static inline void broadcast_and_advance(svuint64_t &x0, svuint64_t &x1,
                                         svuint64_t &x2, svuint64_t &x3,
                                         svuint64_t &x4, svuint64_t &x5,
                                         svuint64_t &x6, svuint64_t &x7,
                                         const std::uint64_t *&aux) {
  const svbool_t ptrue{svptrue_b8()};

  static_assert(cntd != 5 && cntd != 6 && cntd != 7);
  if constexpr (cntd <= 4) {
    broadcast_and_advance(x0, x1, x2, x3, aux);
    broadcast_and_advance(x4, x5, x6, x7, aux);
  } else {
    svuint64_t x;

    if constexpr (cntd == 8) {
      x = svld1(ptrue, aux);
    } else {
      x = svld1(svptrue_pat_b64(SV_VL8), aux);
    }
    aux += 8;

    x0 = svdup_lane(x, 0);
    x1 = svdup_lane(x, 1);
    x2 = svdup_lane(x, 2);
    x3 = svdup_lane(x, 3);
    x4 = svdup_lane(x, 4);
    x5 = svdup_lane(x, 5);
    x6 = svdup_lane(x, 6);
    x7 = svdup_lane(x, 7);
  }
}

} // namespace sventt

#endif /* SVENTT_COMMON_SVE_HPP_INCLUDED */

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_HPP_INCLUDED
#define SVENTT_HPP_INCLUDED

#include "sventt/modulus.hpp"
#include "sventt/utility.hpp"
#include "sventt/vector.hpp"
#include "sventt/wrapper.hpp"

#include "sventt/modmul/scalar/fixed-point-64.hpp"
#include "sventt/modmul/scalar/p-adic-64.hpp"

#include "sventt/layer/scalar/generic.hpp"
#include "sventt/layer/scalar/radix-eight.hpp"
#include "sventt/layer/scalar/radix-four.hpp"
#include "sventt/layer/scalar/radix-two.hpp"

#include "sventt/kernel/iterative.hpp"
#include "sventt/kernel/recursive.hpp"

#if defined(__ARM_FEATURE_SVE)

#include "sventt/common/sve.hpp"

#include "sventt/copy/sve/generic.hpp"

#include "transposition/sve/common.hpp"
#include "transposition/sve/gather-immediate-index-column-first.hpp"
#include "transposition/sve/gather-immediate-index-row-first.hpp"
#include "transposition/sve/gather-vector-index-column-first.hpp"
#include "transposition/sve/gather-vector-index-row-first.hpp"
#include "transposition/sve/in-register-explicit-blocking-row-first.hpp"
#include "transposition/sve/in-register-full-blocking-row-first.hpp"
#include "transposition/sve/in-register-row-first.hpp"
#include "transposition/sve/in-register.hpp"

#include "sventt/modmul/sve/fixed-point-64.hpp"
#include "sventt/modmul/sve/p-adic-64.hpp"

#include "sventt/layer/sve/blocked-generic.hpp"
#include "sventt/layer/sve/generic.hpp"
#include "sventt/layer/sve/radix-eight.hpp"
#include "sventt/layer/sve/radix-four.hpp"
#include "sventt/layer/sve/radix-two.hpp"

#endif

#endif /* SVENTT_HPP_INCLUDED */

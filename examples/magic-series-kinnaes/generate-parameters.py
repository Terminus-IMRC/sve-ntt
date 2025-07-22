# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

import math
import sys

import sympy


def main():

    e, m = [int(_) for _ in sys.argv[1:]]
    r = m * m * (m - 1) // 2

    for n in range(r + 1 + (r % 2), m * m * (m - 1), 2):
        if min(sympy.factorint(n, limit=m)) > m:
            break
    else:
        raise RuntimeError('failed to find n')

    bound = math.comb(m * m, m - 1)
    prod = 1
    moduli = []
    for k in range(2**e // n, 0, -1):
        N = k * n + 1
        if not sympy.isprime(N):
            continue

        moduli.append([N, sympy.primitive_root(N)])

        prod *= N
        if prod >= bound:
            break
    else:
        raise RuntimeError('insufficient number of primes found')

    print(',\n'.join(
        f'std::tuple<' + ', '.join([
            f'std::integral_constant<std::uint64_t, {m}>',
            f'sventt::Modulus<UINT64_C({N:#018x}), UINT64_C({omega})>',
            f'std::integral_constant<std::uint64_t, {n}>',
        ])
        + '>'
        for N, omega in moduli))


if __name__ == '__main__':

    main()

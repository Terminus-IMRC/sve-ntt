// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_WRAPPER_HPP_INCLUDED
#define SVENTT_WRAPPER_HPP_INCLUDED

#include <cstdint>

#include "sventt/vector.hpp"

namespace sventt {

template <class kernel_type_> class NTT {

  AuxiliaryVector aux;
  decltype(aux)::size_type aux_inverse_offset;

  static std::uint64_t get_forward_aux_size(void) {
    FakeByteVector aux;
    kernel_type::prepare_forward(aux);
    return aux.size();
  }

  static std::uint64_t get_inverse_aux_size(void) {
    FakeByteVector aux;
    kernel_type::prepare_inverse(aux);
    return aux.size();
  }

public:
  using kernel_type = kernel_type_;
  using modulus_type = kernel_type::modulus_type;

  NTT(const bool enable_forward = true, const bool enable_inverse = true,
      const bool allocate_huge_pages = true)
      : aux{(enable_forward ? get_forward_aux_size() : 0) +
                (enable_inverse ? get_inverse_aux_size() : 0),
            allocate_huge_pages} {
    if (enable_forward) {
      kernel_type::prepare_forward(aux);
    }
    aux_inverse_offset = aux.size();
    if (enable_inverse) {
      kernel_type::prepare_inverse(aux);
    }
  }

  static constexpr std::uint64_t get_m(void) { return kernel_type::get_m(); }

  void compute_forward(std::uint64_t *const dst,
                       const std::uint64_t *const src) const {
    const std::byte *aux_ptr{aux.data()};
    kernel_type::compute_forward(dst, src, aux_ptr);
    if (std::cmp_not_equal(aux_ptr - aux.data(), aux_inverse_offset)) {
      throw std::logic_error{"Auxiliary data were not properly consumed"};
    }
  }

  void compute_forward(std::uint64_t *const dst) const {
    const std::byte *aux_ptr{aux.data()};
    kernel_type::compute_forward(dst, aux_ptr);
    if (std::cmp_not_equal(aux_ptr - aux.data(), aux_inverse_offset)) {
      throw std::logic_error{"Auxiliary data were not properly consumed"};
    }
  }

  void compute_inverse(std::uint64_t *const dst,
                       const std::uint64_t *const src) const {
    const std::byte *aux_ptr{aux.data() + aux_inverse_offset};
    kernel_type::compute_inverse(dst, src, aux_ptr);
    if (std::cmp_not_equal(aux_ptr - aux.data(), aux.size())) {
      throw std::logic_error{"Auxiliary data were not properly consumed"};
    }
  }

  void compute_inverse(std::uint64_t *const dst) const {
    const std::byte *aux_ptr{aux.data() + aux_inverse_offset};
    kernel_type::compute_inverse(dst, aux_ptr);
    if (std::cmp_not_equal(aux_ptr - aux.data(), aux.size())) {
      throw std::logic_error{"Auxiliary data were not properly consumed"};
    }
  }

  void descramble_forward(std::uint64_t *const dst,
                          const std::uint64_t *const src) const {
    kernel_type::descramble_forward(dst, src);
  }

  void descramble_inverse(std::uint64_t *const dst,
                          const std::uint64_t *const src) const {
    kernel_type::descramble_inverse(dst, src);
  }
};

} // namespace sventt

#endif /* SVENTT_WRAPPER_HPP_INCLUDED */

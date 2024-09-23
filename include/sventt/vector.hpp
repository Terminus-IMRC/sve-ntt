// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright © 2025 Yukimasa Sugizaki

#ifndef SVENTT_VECTOR_HPP_INCLUDED
#define SVENTT_VECTOR_HPP_INCLUDED

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <vector>

#include <sys/mman.h>

#include <boost/align/aligned_allocator.hpp>

namespace sventt {

namespace pointer_utility {

template <class T> std::size_t get_padding(const std::size_t pos) {
  return (pos % alignof(T) == 0) ? 0 : (alignof(T) - pos % alignof(T));
}

template <class T> std::size_t get_padding(const std::byte *pointer) {
  return get_padding<T>(reinterpret_cast<std::uintptr_t>(pointer));
}

template <class T> void skip_padding(const std::byte *&pointer) {
  pointer += get_padding<T>(pointer);
}

template <class T> const T &get_and_advance(const std::byte *&pointer) {
  skip_padding<T>(pointer);
  pointer += sizeof(T);
  return *reinterpret_cast<const T *>(pointer - sizeof(T));
}

template <class T> const T &get(const std::byte *pointer) {
  return get_and_advance<T>(pointer);
}

} // namespace pointer_utility

static inline void touch_pages_cyclically(
    void *const ptr, const std::uint64_t size, const std::uint64_t page_size,
    const std::uint64_t num_domains, const std::uint64_t domain_num_threads,
    const std::uint64_t domain_num, const std::uint64_t domain_thread_num) {
  for (std::uint64_t i =
           page_size * (domain_thread_num * num_domains + domain_num);
       i < size; i += page_size * domain_num_threads * num_domains) {
    static_cast<std::byte *>(ptr)[i] = {};
  }
}

template <class value_type_> class PageMemory {

public:
  using value_type = value_type_;
  using size_type = std::uint64_t;

private:
  class Deleter {

    size_type size;

  public:
    Deleter(void) = default;

    Deleter(const size_type size) : size{size} {}

    void operator()(void *const ptr) const { munmap(ptr, size); }
  };

  size_type length;
  std::unique_ptr<value_type[], Deleter> ptr;

public:
  PageMemory(void) = default;

  PageMemory(const size_type length, const bool allocate_huge_pages) {
    reset(length, allocate_huge_pages);
  }

  size_type size(void) const { return length; }

  value_type *data(void) { return ptr.get(); }

  const value_type *data(void) const { return ptr.get(); }

  void reset(void) {
    length = 0;
    ptr.reset();
  }

  void reset(const size_type len, const bool allocate_huge_pages = false) {
    reset(len, allocate_huge_pages ? size_type{21} : size_type{});
  }

  /*
   * Warning: log2_page_size = 0 means normal pages, not huge pages with default
   *          size each.
   */
  void reset(const size_type len, const size_type log2_page_size) {
    length = len;
    ptr.reset();
    if (len == 0) {
      return;
    }

    if (std::bit_width(log2_page_size) >= 7) {
      throw std::invalid_argument{"Too large huge page size"};
    }

    const size_type align{size_type{1} << log2_page_size};
    const size_type size{(sizeof(value_type) * len + (align - 1)) &
                         ~(align - 1)};
    value_type *const p{static_cast<value_type *>(
        mmap(nullptr, size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                 (log2_page_size == 0
                      ? 0
                      : MAP_HUGETLB | (log2_page_size << MAP_HUGE_SHIFT)),
             -1, 0))};
    if (p == MAP_FAILED) {
      throw std::bad_alloc{};
    }

    ptr = {p, Deleter{size}};
  }

  value_type &operator[](const size_type index) { return ptr[index]; }

  const value_type &operator[](const size_type index) const {
    return ptr[index];
  }

  value_type &at(const size_type index) {
    if (index >= size()) {
      throw std::out_of_range{"Index out of range"};
    }
    return ptr[index];
  }

  const value_type &at(const size_type index) const {
    if (index >= size()) {
      throw std::out_of_range{"Index out of range"};
    }
    return ptr[index];
  }

  value_type *begin(void) { return ptr.get(); }

  value_type *end(void) { return ptr.get() + size(); }

  const value_type *begin(void) const { return ptr.get(); }

  const value_type *end(void) const { return ptr.get() + size(); }

  value_type *cbegin(void) const { return ptr.get(); }

  value_type *cend(void) const { return ptr.get() + size(); }
};

class FakeByteVector {

public:
  using value_type = std::byte;
  using size_type = std::uint64_t;

private:
  size_type length{};

public:
  template <class T> void push_back([[maybe_unused]] const T &value) {
    length += pointer_utility::get_padding<T>(length) + sizeof(T);
  }

  size_type size(void) const { return length; }

  template <class T> T &reinterpret_at(const size_type index) {
    if (index + sizeof(T) - 1 >= size()) {
      throw std::out_of_range{"Index out of range"};
    }

    static T value;
    return value;
  }
};

class AuxiliaryVector {

public:
  using value_type = std::byte;
  using size_type = PageMemory<value_type>::size_type;

private:
  size_type length;
  PageMemory<value_type> memory;

public:
  AuxiliaryVector(void) = default;

  AuxiliaryVector(const size_type length,
                  const bool allocate_huge_pages = false)
      : length{}, memory{length, allocate_huge_pages} {}

  size_type size(void) const { return length; }

  size_type capacity(void) const { return memory.size(); }

  value_type *data(void) { return memory.data(); }

  const value_type *data(void) const { return memory.data(); }

  value_type &operator[](const size_type index) { return memory[index]; }

  const value_type &operator[](const size_type index) const {
    return memory[index];
  }

  value_type &at(const size_t index) {
    if (index >= size()) {
      throw std::out_of_range{"Index out of range"};
    }

    return memory[index];
  }

  template <class T> T &reinterpret_at(const size_t index) {
    if (index + sizeof(T) - 1 >= size()) {
      throw std::out_of_range{"Index out of range"};
    }

    return *reinterpret_cast<T *>(&memory[index]);
  }

  template <class T> void push_back(const T &value) {
    const std::size_t padding{pointer_utility::get_padding<T>(length)};
    if (length + padding + sizeof(T) > capacity()) {
      throw std::bad_alloc{};
    }

    length += padding;
    std::ranges::copy(std::bit_cast<std::array<std::byte, sizeof(T)>>(value),
                      &memory[length]);
    length += sizeof(T);
  }
};

template <class value_type_, std::uint64_t alignment>
class UninitializedVector {

public:
  using value_type = value_type_;

private:
  std::uint64_t length{};
  value_type *ptr{};

  static value_type *allocate(const std::uint64_t length) {
    value_type *const ptr{static_cast<value_type *>(
        std::aligned_alloc(alignment, sizeof(value_type) * length))};
    if (ptr == nullptr) {
      throw std::bad_alloc{};
    }
    return ptr;
  }

public:
  UninitializedVector(void) = default;

  UninitializedVector(const std::uint64_t length)
      : length{length}, ptr{allocate(length)} {}

  UninitializedVector &operator=(const UninitializedVector &that) {
    length = that.length;
    std::free(ptr);
    ptr = allocate(length);
    std::copy_n(that.ptr, length, ptr);
    return *this;
  }

  ~UninitializedVector(void) {
    std::free(ptr);
    ptr = nullptr;
  }

  value_type *data(void) { return ptr; }

  const value_type *data(void) const { return ptr; }
};

} // namespace sventt

#endif /* SVENTT_VECTOR_HPP_INCLUDED */

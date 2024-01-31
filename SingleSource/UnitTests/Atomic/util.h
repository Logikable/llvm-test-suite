//===--- util.h -- Utility functions shared by tests -------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _UTIL_H_
#define _UTIL_H_

#include <cstdint>
#include <iostream>

#define TEST16 1

static constexpr uint64_t V = 0x4243444546474849;
static constexpr int kThreads = 10;
static constexpr int kIterations = 1'000'000;
static constexpr int kExpected = kThreads * kIterations;
// 1e-7 is approximately the reciprocal of 2^23. There are 23-bits in the
// mantissa of a single-precision float.
// Without an epsilon, the floating point tests fail because the mantissa of a
// 32-bit float doesn't have enough precision to be incremented 10^7 times.
static constexpr double kEpsilon = 1e-7 * kThreads * kIterations;

constexpr int atomic_fetch_models[] = {
  // TODO: Figure out a way to test all of these in parallel.
  // __ATOMIC_RELAXED, __ATOMIC_CONSUME, __ATOMIC_ACQUIRE,
  // __ATOMIC_ACQ_REL, __ATOMIC_RELEASE,
  __ATOMIC_SEQ_CST,
};
constexpr int atomic_exchange_models[] = {
  // TODO: Figure out a way to test all of these in parallel.
  // __ATOMIC_RELAXED, __ATOMIC_ACQUIRE, __ATOMIC_RELEASE,
  // __ATOMIC_ACQ_REL,
  __ATOMIC_SEQ_CST,
};
constexpr int atomic_compare_exchange_models[] = {
  // TODO: Figure out a way to test all of these in parallel.
  // __ATOMIC_RELAXED, __ATOMIC_CONSUME, __ATOMIC_ACQUIRE,
  // __ATOMIC_ACQ_REL, __ATOMIC_RELEASE,
  __ATOMIC_SEQ_CST,
};

template <typename T>
constexpr uint8_t right_shift() {
  switch (sizeof(T)) {
    case 16:
      return 0;
    case 8:
      return 24;
    case 4:
      return 32 + 24;
    default:
      return 63;
  }
}

template <typename T>
void print_int(T aint, T iint) {
  std::cout << "atomic: " << aint << " "
            << "nonatomic: " << iint << "\n";
}

template <>
void print_int<__uint128_t>(__uint128_t aint, __uint128_t iint) {
  std::cout << "atomic (top 8 bytes): "
            << static_cast<uint64_t>(aint >> 64) << " "
            << "nonatomic (top 8 bytes): "
            << static_cast<uint64_t>(iint >> 64) << "\n";
}

template <>
void print_int<__int128_t>(__int128_t aint, __int128_t iint) {
  print_int(static_cast<__uint128_t>(aint), static_cast<__uint128_t>(iint));
}

#endif  // _UTIL_H_

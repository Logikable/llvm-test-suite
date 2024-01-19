//===--- numeric.h -- Tests shared between integers and floats ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains loopers shared by integer and floating point types.
//
// The types tested are: uint32, uint64, int32, int64, float, double.
// The ops tested are: xchg, cmpxchg.
//
//===----------------------------------------------------------------------===//

#ifndef _NUMERIC_H_
#define _NUMERIC_H_

#include "util.h"

// The xchg tests work as follows:
//
// Each thread increments a local copy of the shared variable, and exchanges it
// with the shared value. Most of the time, the value moved from shared -> local
// is one less than the value moved from local -> shared. Other times, the
// difference is much bigger (or smaller). When this occurs, the thread
// accumulates the difference in a local error variable. Upon completion, the
// thread subtracts the error from the shared value, all at once.
//
// Like many tests, this test increments by more than 1 -- specifically, a
// number that scales with the width of the type is picked.
//
// This test also splits the atomic and nonatomic versions into separate loopers
// for readability.
template <typename T>
void looper_numeric_xchg_atomic(T *aint, int model) {
  static constexpr T val = V >> right_shift<T>();
  __int128_t error = 0;
  T next = *aint + val;
  T result;
  for (int n = 0; n < kIterations; ++n) {
    __atomic_exchange(aint, &next, &result, model);
    error +=
        static_cast<__int128_t>(next) - static_cast<__int128_t>(result + val);
    next = result + val;
  }
  __atomic_fetch_sub(aint, static_cast<T>(error), model);
}

template <typename T>
void looper_numeric_xchg_nonatomic(T &iint, int model) {
  static constexpr T val = V >> right_shift<T>();
  __int128_t error = 0;
  T next = iint + val;
  T result;
  for (int n = 0; n < kIterations; ++n) {
    result = iint;
    iint = next;
    error +=
        static_cast<__int128_t>(next) - static_cast<__int128_t>(result + val);
    next = result + val;
  }
  __atomic_fetch_sub(&iint, static_cast<T>(error), model);
}

// The cmpxchg tests act similar to fetch_add tests.
template <typename T>
void looper_numeric_cmpxchg(T *aint, T &iint, int success_model,
                            int fail_model) {
  static constexpr T val = V >> right_shift<T>();
  for (int n = 0; n < kIterations; ++n) {
    T desired, expected = 0;
    do {
      desired = expected + val;
    } while (!__atomic_compare_exchange(aint, &expected, &desired, true,
                                        success_model, fail_model));
    iint += val;
  }
}

#endif  // _NUMERIC_H_

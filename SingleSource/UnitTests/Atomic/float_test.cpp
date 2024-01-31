//===--- float_test.cc -- Testing aligned floating point numbers -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The following text is present in each test file:
//
// These tests aim to capture real-world multithreaded use cases of atomic
// builtins. Each test focuses on a single atomic operation. Those using
// multiple operations can be compared with other tests using the same
// operations to isolate bugs to a single atomic operation.
//
// Each test consists of a "looper" body and a test script. The test script
// instantiates 10 threads, each running the looper. The loopers contend the
// same memory address, performing atomic operations on it. Each looper executes
// 10^6 times for a total of 10^7 operations. The resultant value in the
// contended pointer is compared against a closed-form solution. It's expected
// that the two values equate.
//
// For example, a looper that increments the shared pointer is expected to end
// up with a value of 10^7. If its final value is not that, the test fails.
//
// Each test also tests the corresponding nonatomic operation with a separate
// shared variable. Ideally, this value differs from the atomic "correct" value,
// and the test can compare the two. In reality, some simpler operations (e.g.
// those conducted through the ALU) can still end up with the correct answer,
// even when performed nonatomically. These tests do not check the nonatomic
// result, although it is still outputted to aid in debugging.
//
// Each test is performed on all relevant types.
//
//===----------------------------------------------------------------------===//
//
// This file tests atomic operations on floating point types with aligned
// memory addresses.
//
// The types tested are: float, double.
// The ops tested are: xchg, cmpxchg.
//
//===----------------------------------------------------------------------===//

#include <sys/stat.h>

#include <cstdio>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "numeric.h"
#include "util.h"

template <typename T>
class ScalarFloatTest : public ::testing::Test {};
using ScalarFloatTestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ScalarFloatTest, ScalarFloatTestTypes);

// See numeric.h for an explanation of numeric xchg tests.
TYPED_TEST(ScalarFloatTest, Xchg) {
  static constexpr TypeParam val = V >> right_shift<TypeParam>();
  static constexpr TypeParam expected = val * kExpected;
  std::vector<std::thread> pool;

  for (int model : atomic_exchange_models) {
    TypeParam afloat = 0, ffloat = 0;
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_numeric_xchg_atomic<TypeParam>, &afloat, model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_numeric_xchg_nonatomic<TypeParam>,
                        std::ref(ffloat), model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    std::cout << "SCALAR (FETCH ADD): "
              << "atomic: " << afloat << " "
              << "nonatomic: " << ffloat << "\n";
    EXPECT_GE(afloat, ffloat);
    EXPECT_GE(afloat, expected * (1 - kEpsilon));
    EXPECT_LE(afloat, expected * (1 + kEpsilon));
  }
}

// See numeric.h for an explanation of numeric cmpxchg tests.
TYPED_TEST(ScalarFloatTest, Cmpxchg) {
  static constexpr TypeParam val = V >> right_shift<TypeParam>();
  static constexpr TypeParam expected = val * kExpected;
  std::vector<std::thread> pool;

  for (int success_model : atomic_compare_exchange_models) {
    for (int fail_model : atomic_compare_exchange_models) {
      TypeParam afloat = 0, ffloat = 0;
      for (int n = 0; n < kThreads; ++n)
        pool.emplace_back(looper_numeric_cmpxchg<TypeParam>, &afloat,
                          std::ref(ffloat), success_model, fail_model);
      for (int n = 0; n < kThreads; ++n)
        pool[n].join();
      pool.clear();
      std::cout << "SCALAR (FETCH ADD): "
                << "atomic: " << afloat << " "
                << "nonatomic: " << ffloat << "\n";
      EXPECT_GE(afloat, ffloat);
      EXPECT_GE(afloat, expected * (1 - kEpsilon));
      EXPECT_LE(afloat, expected * (1 + kEpsilon));
    }
  }
}

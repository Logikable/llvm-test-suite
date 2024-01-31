//===--- misshapen.cc -- Testing non-power-of-2 byte objects ------ C++ -*-===//
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
// This file tests atomic operations on oddly sized objects with aligned
// memory addresses.
//
// The types tested are: 3 byte up to 15 byte objects (skipping 4 & 8).
// The ops tested are: xchg, cmpxchg.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "util.h"

template<int N>
struct misshapen {
  unsigned char v[N];
};

template <typename T>
class MisshapenTest : public ::testing::Test {};
using MisshapenTestTypes = ::testing::Types<
    misshapen<3>, misshapen<5>, misshapen<6>, misshapen<7>, misshapen<9>,
    misshapen<10>, misshapen<11>, misshapen<12>, misshapen<13>, misshapen<14>,
    misshapen<15>>;
TYPED_TEST_SUITE(MisshapenTest, MisshapenTestTypes);

// See numeric.h for an explanation of numeric xchg tests.
template <int N>
void looper_misshapen_xchg_atomic(misshapen<N> *amis, int model) {
  unsigned char error[N] = {};
  misshapen<N> next, result;
  __atomic_load(amis, &next, model);
  for (int k = 0; k < N; ++k)
    next.v[k]++;
  for (int n = 0; n < kIterations; ++n) {
    __atomic_exchange(amis, &next, &result, model);
    for (int k = 0; k < N; ++k) {
      error[k] += next.v[k] - (result.v[k] + 1);
      next.v[k] = result.v[k] + 1;
    }
  }
  // We can't use atomic_sub here; combining atomic operations on array members
  // and the array as a whole is undefined.
  misshapen<N> desired, expected;
  __atomic_load(amis, &expected, model);
  do {
    desired = expected;
    for (int k = 0; k < N; ++k)
      desired.v[k] -= error[k];
  } while (!__atomic_compare_exchange(amis, &expected, &desired, true,
                                      model, model));
}

template <int N>
void looper_misshapen_xchg_nonatomic(misshapen<N> &mmis, int model) {
  unsigned char error[N] = {};
  misshapen<N> next, result;
  __atomic_load(&mmis, &next, model);
  for (int k = 0; k < N; ++k)
    next.v[k]++;
  for (int n = 0; n < kIterations; ++n) {
    result = mmis;
    mmis = next;
    for (int k = 0; k < N; ++k) {
      error[k] += next.v[k] - (result.v[k] + 1);
      next.v[k] = result.v[k] + 1;
    }
  }
  // We can't use atomic_sub here; combining atomic operations on array members
  // and the array as a whole is undefined.
  misshapen<N> desired, expected;
  __atomic_load(&mmis, &expected, model);
  do {
    desired = expected;
    for (int k = 0; k < N; ++k)
      desired.v[k] -= error[k];
  } while (!__atomic_compare_exchange(&mmis, &expected, &desired, true,
                                      model, model));
}

TYPED_TEST(MisshapenTest, Xchg) {
  std::vector<std::thread> pool;
  const int N = sizeof(TypeParam);

  for (int model : atomic_exchange_models) {
    TypeParam amis = {}, mmis = {};
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_misshapen_xchg_atomic<N>, &amis, model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_misshapen_xchg_nonatomic<N>, std::ref(mmis),
                        model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    std::cout << "XCHG: ";
    std::cout << "atomic: ";
    for (int n = 0; n < N; ++n)
      std::cout << unsigned(amis.v[n]) << " ";
    std::cout << "\n      ";
    std::cout << "nonatomic: ";
    for (int n = 0; n < N; ++n)
      std::cout << unsigned(mmis.v[n]) << " ";
    std::cout << "\n";
    for (int n = 0; n < N; ++n)
      EXPECT_EQ(amis.v[n], kExpected % (1 << (8 * sizeof(amis.v[0]))));
  }
}

// See numeric.h for an explanation of numeric cmpxchg tests.
template <int N>
void looper_misshapen_cmpxchg(misshapen<N> *amis, misshapen<N> &mmis,
                              int success_model, int fail_model) {
  for (int n = 0; n < kIterations; ++n) {
    misshapen<N> desired, expected = {};
    do {
      desired = expected;
      for (int k = 0; k < N; ++k)
        desired.v[k]++;
    } while (!__atomic_compare_exchange(amis, &expected, &desired, true,
                                        success_model, fail_model));
    for (int k = 0; k < N; ++k)
      mmis.v[k]++;
  }
}

TYPED_TEST(MisshapenTest, Cmpxchg) {
  std::vector<std::thread> pool;
  const int N = sizeof(TypeParam);

  for (int success_model : atomic_compare_exchange_models) {
    for (int fail_model : atomic_compare_exchange_models) {
      TypeParam amis = {}, mmis = {};
      for (int n = 0; n < kThreads; ++n)
        pool.emplace_back(looper_misshapen_cmpxchg<N>, &amis, std::ref(mmis),
                          success_model, fail_model);
      for (int n = 0; n < kThreads; ++n)
        pool[n].join();
      pool.clear();
      std::cout << "CMPXCHG: ";
      std::cout << "atomic: ";
      for (int n = 0; n < N; ++n)
        std::cout << unsigned(amis.v[n]) << " ";
      std::cout << "\n      ";
      std::cout << "nonatomic: ";
      for (int n = 0; n < N; ++n)
        std::cout << unsigned(mmis.v[n]) << " ";
      std::cout << "\n";
      for (int n = 0; n < N; ++n)
        EXPECT_EQ(amis.v[n], kExpected % (1 << (8 * sizeof(amis.v[0]))));
    }
  }
}

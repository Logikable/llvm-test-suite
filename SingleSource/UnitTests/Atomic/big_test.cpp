//===--- big_test.cc -- Testing big (17+ byte) objects ------------ C++ -*-===//
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
// This file tests atomic operations on big objects with aligned memory
// addresses.
//
// The types tested are: bigs.
// The ops tested are: xchg, cmpxchg.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

#include "util.h"

static constexpr int kBigSize = 10;
struct big_t {
  int v[kBigSize];
};

// The big struct cmpxchg test is identical to the numeric cmpxchg test, except
// each element of the underlying array is incremented.
void looper_big_cmpxchg(big_t *abig, big_t &bbig, int success_model,
                        int fail_model) {
  for (int n = 0; n < kIterations; ++n) {
    big_t desired, expected = {};
    do {
      desired = expected;
      for (int k = 0; k < kBigSize; ++k)
        desired.v[k]++;
    } while (!__atomic_compare_exchange(abig, &expected, &desired, true,
                                        success_model, fail_model));
    for (int k = 0; k < kBigSize; ++k)
      bbig.v[k]++;
  }
}

void test_big_cmpxchg() {
  std::vector<std::thread> pool;

  for (int success_model : atomic_compare_exchange_models) {
    for (int fail_model : atomic_compare_exchange_models) {
      big_t abig = {};
      big_t bbig = {};
      for (int n = 0; n < kThreads; ++n)
        pool.emplace_back(looper_big_cmpxchg, &abig, std::ref(bbig),
                          success_model, fail_model);
      for (int n = 0; n < kThreads; ++n)
        pool[n].join();
      pool.clear();
      std::cout << "CMPXCHG: ";
      std::cout << "atomic: ";
      for (int n = 0; n < kBigSize; ++n)
        std::cout << abig.v[n] << " ";
      std::cout << "\n      ";
      std::cout << "nonatomic: ";
      for (int n = 0; n < kBigSize; ++n)
        std::cout << bbig.v[n] << " ";
      std::cout << "\n";
      for (int n = 0; n < kBigSize; ++n)
        if (lt(abig.v[n], bbig.v[n]) || abig.v[n] != kExpected)
          fail();
    }
  }
}

void test_big() {
  printf("Testing big\n");
  test_big_cmpxchg();
}

int main() {
  printf("%d threads; %d iterations each; total of %d\n", kThreads, kIterations,
         kExpected);

  test_big();
  printf("PASSED\n");
}

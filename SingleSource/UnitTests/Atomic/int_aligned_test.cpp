//===--- int_aligned_test.cc -- Testing aligned integers ---------- C++ -*-===//
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
// This file tests atomic operations on signed and unsigned integer types with
// aligned memory addresses.
//
// The types tested are: uint32, uint64, int32, int64, uint128, int128.
// The ops tested are: add, sub, and, or, xor, xchg, xchg_n, cmpxchg, cmpxchg_n.
// The ALU operations are not tested on 128-bit integers.
//
//===----------------------------------------------------------------------===//

#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

#include "numeric.h"
#include "util.h"

template <typename T>
void looper_int_fetch_add(T *aint, T &iint, int model) {
  static constexpr T val = V >> right_shift<T>();
  for (int n = 0; n < kIterations; ++n) {
    __atomic_fetch_add(aint, val, model);
    iint += val;
  }
}

template <typename T>
void test_int_fetch_add(T &aint, T &iint) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int model : atomic_fetch_models) {
    aint = 0;
    iint = 0;
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_int_fetch_add<T>, &aint, std::ref(iint), model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    std::cout << "FETCH ADD: "
              << "atomic: " << aint << " "
              << "nonatomic: " << iint << "\n";
    if (lt(aint, iint) || aint != val * kExpected)
      fail();
  }
}

template <typename T>
void looper_int_fetch_sub(T *aint, T &iint, int model) {
  static constexpr T val = V >> right_shift<T>();
  for (int n = 0; n < kIterations; ++n) {
    __atomic_fetch_sub(aint, val, model);
    iint -= val;
  }
}

template <typename T>
void test_int_fetch_sub(T &aint, T &iint) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int model : atomic_fetch_models) {
    aint = val * kExpected;
    iint = val * kExpected;
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_int_fetch_sub<T>, &aint, std::ref(iint), model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    std::cout << "FETCH SUB: "
              << "atomic: " << aint << " "
              << "nonatomic: " << iint << "\n";
    if (lt(iint, aint) || aint != 0)
      fail();
  }
}

// The AND + OR tests work as follows:
//
// Each of the 10 threads "owns" one bit of the shared member. Each thread
// attempts to flip the bit it owns, from 1 -> 0 when testing AND, and 0 -> 1
// when testing OR. If successful, the thread increments a counter, and a
// cmpxchg loop flips the bit back. Once complete, that counter should equal
// 10^7.
template <typename T>
void __attribute__((optnone)) looper_int_fetch_and(
    const int id, T *aint, T &iint, T *acnt, T *icnt, int model) {
  T desired, expected = 0;
  for (int n = 0; n < kIterations; ++n) {
    const T mask = 1 << id;
    __atomic_fetch_and(aint, ~mask, model);
    if (~*aint & mask) {
      __atomic_fetch_add(acnt, 1, model);
      do {
        desired = expected | mask;
      } while (!__atomic_compare_exchange(aint, &expected, &desired, true,
                                          model, model));
    }
    iint &= ~mask;
    if (~iint & mask) {
      __atomic_fetch_add(icnt, 1, model);
      iint |= mask;
    }
  }
}

template <typename T>
void test_int_fetch_and(T &aint, T &iint) {
  std::vector<std::thread> pool;
  for (int model : atomic_fetch_models) {
    T acnt = 0, icnt = 0;
    aint = ~0, iint = ~0;
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_int_fetch_and<T>, n, &aint, std::ref(iint),
                        &acnt, &icnt, model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    std::cout << "FETCH AND: "
              << "atomic: " << acnt << " "
              << "nonatomic: " << icnt << "\n";
    if (acnt != kExpected)
      fail();
  }
}

template <typename T>
void __attribute__((optnone)) looper_int_fetch_or(
    const int id, T *aint, T &iint, T *acnt, T *icnt, int model) {
  T desired, expected = 0;
  for (int n = 0; n < kIterations; ++n) {
    const T mask = 1 << id;
    __atomic_fetch_or(aint, mask, model);
    if (*aint & mask) {
      __atomic_fetch_add(acnt, 1, model);
      do {
        desired = expected & ~mask;
      } while (!__atomic_compare_exchange(aint, &expected, &desired, true,
                                          model, model));
    }
    iint |= mask;
    if (iint & mask) {
      __atomic_fetch_add(icnt, 1, model);
      iint &= ~mask;
    }
  }
}

template <typename T>
void test_int_fetch_or(T &aint, T &iint) {
  std::vector<std::thread> pool;
  for (int model : atomic_fetch_models) {
    T acnt = 0, icnt = 0;
    aint = 0, iint = 0;
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_int_fetch_or<T>, n, &aint, std::ref(iint),
                        &acnt, &icnt, model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    std::cout << "FETCH OR: "
              << "atomic: " << acnt << " "
              << "nonatomic: " << icnt << "\n";
    if (acnt != kExpected)
      fail();
  }
}

template <typename T>
void looper_int_fetch_xor(T *aint, T &iint, int model) {
  for (int n = 0; n < kIterations; ++n) {
    __atomic_fetch_xor(aint, n, model);
    iint ^= n;
  }
}

template <typename T>
void test_int_fetch_xor(T &aint, T &iint) {
  std::vector<std::thread> pool;
  for (int model : atomic_fetch_models) {
    aint = 0;
    iint = 0;
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_int_fetch_xor<T>, &aint, std::ref(iint), model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    std::cout << "FETCH XOR: "
              << "atomic: " << aint << " "
              << "nonatomic: " << iint << "\n";
    if (aint != 0)
      fail();
  }
}

template <typename T>
void test_int_xchg(T &aint, T &iint) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int model : atomic_exchange_models) {
    aint = 0;
    iint = 0;
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_numeric_xchg_atomic<T>, &aint, model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    for (int n = 0; n < kThreads; ++n)
      pool.emplace_back(looper_numeric_xchg_nonatomic<T>, std::ref(iint),
                        model);
    for (int n = 0; n < kThreads; ++n)
      pool[n].join();
    pool.clear();
    std::cout << "XCHG: ";
    print_int(aint, iint);
    if (lt(aint, iint) || aint != val * kExpected)
      fail();
  }
}

// See numeric.h for an explanation of numeric xchg tests.
template <typename T>
void looper_int_xchg_n(T *aint, int model) {
  static constexpr T val = V >> right_shift<T>();
  __int128_t error = 0;
  T next = *aint + val;
  for (int n = 0; n < kIterations; ++n) {
    T result = __atomic_exchange_n(aint, next, model);
    error +=
        static_cast<__int128_t>(next) - static_cast<__int128_t>(result + val);
    next = result + val;
  }
  __atomic_fetch_sub(aint, static_cast<T>(error), model);
}

template <typename T>
void test_int_xchg_n(T &aint, T &iint) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int model : atomic_exchange_models) {
    aint = 0;
    iint = 0;
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_int_xchg_n<T>, &aint, model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_numeric_xchg_nonatomic<T>, std::ref(iint),
                        model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    std::cout << "XCHG_N: ";
    print_int(aint, iint);
    if (lt(aint, iint) || aint != val * kExpected) {
      fail();
    }
  }
}

template <typename T>
void test_int_cmpxchg(T &aint, T &iint) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int success_model : atomic_compare_exchange_models) {
    for (int fail_model : atomic_compare_exchange_models) {
      aint = 0;
      iint = 0;
      for (int n = 0; n < kThreads; ++n) {
        pool.emplace_back(looper_numeric_cmpxchg<T>, &aint, std::ref(iint),
                          success_model, fail_model);
      }
      for (int n = 0; n < kThreads; ++n) {
        pool[n].join();
      }
      pool.clear();
      std::cout << "CMPXCHG: ";
      print_int(aint, iint);
      if (lt(aint, iint) || aint != static_cast<T>(val) * kExpected) {
        fail();
      }
    }
  }
}

// See numeric.h for an explanation of numeric cmpxchg tests.
template <typename T>
void looper_int_cmpxchg_n(T *aint, T &iint, int success_model, int fail_model) {
  static constexpr T val = V >> right_shift<T>();
  for (int n = 0; n < kIterations; ++n) {
    T desired, expected = 0;
    do {
      desired = expected + val;
    } while (!__atomic_compare_exchange_n(aint, &expected, desired, true,
                                          success_model, fail_model));
    iint += val;
  }
}

template <typename T>
void test_int_cmpxchg_n(T &aint, T &iint) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int success_model : atomic_compare_exchange_models) {
    for (int fail_model : atomic_compare_exchange_models) {
      aint = 0;
      iint = 0;
      for (int n = 0; n < kThreads; ++n) {
        pool.emplace_back(looper_int_cmpxchg_n<T>, &aint, std::ref(iint),
                          success_model, fail_model);
      }
      for (int n = 0; n < kThreads; ++n) {
        pool[n].join();
      }
      pool.clear();
      std::cout << "CMPXCHG_N: ";
      print_int(aint, iint);
      if (lt(aint, iint) || aint != static_cast<T>(val) * kExpected) {
        fail();
      }
    }
  }
}

void test_aligned_int() {
#define INT_SUITE(type)                    \
  {                                        \
    printf("Testing aligned " #type "\n"); \
    type aint = 0;                         \
    type iint = 0;                         \
    test_int_fetch_add<type>(aint, iint);  \
    test_int_fetch_sub<type>(aint, iint);  \
    test_int_fetch_and<type>(aint, iint);  \
    test_int_fetch_or<type>(aint, iint);   \
    test_int_fetch_xor<type>(aint, iint);  \
    test_int_xchg<type>(aint, iint);       \
    test_int_xchg_n<type>(aint, iint);     \
    test_int_cmpxchg<type>(aint, iint);    \
    test_int_cmpxchg_n<type>(aint, iint);  \
  }
  INT_SUITE(uint32_t);
  INT_SUITE(uint64_t);
  INT_SUITE(int32_t);
  INT_SUITE(int64_t);
#undef INT_SUITE

#if TEST16
#define INT_SUITE(type)                    \
  {                                        \
    printf("Testing aligned " #type "\n"); \
    type aint = 0;                         \
    type iint = 0;                         \
    test_int_xchg<type>(aint, iint);       \
    test_int_xchg_n<type>(aint, iint);     \
    test_int_cmpxchg<type>(aint, iint);    \
    test_int_cmpxchg_n<type>(aint, iint);  \
  }
  INT_SUITE(__uint128_t);
  INT_SUITE(__int128_t);
#undef INT_SUITE
#endif
}

int main() {
  printf("%d threads; %d iterations each; total of %d\n", kThreads, kIterations,
         kExpected);

  test_aligned_int();
  printf("PASSED\n");
}

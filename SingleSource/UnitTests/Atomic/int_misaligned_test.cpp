//===--- int_misaligned_test.cc -- Testing unaligned integers ----- C++ -*-===//
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
// unaligned memory addresses.
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

#include "util.h"

template <typename T>
struct __attribute__((packed)) misaligned {
  char byte;
  T data;
  char pad[(31 - sizeof(T)) % 16];  // Pad struct to 16 or 32 bytes.
};

template <typename T>
void looper_int_misaligned_fetch_add(misaligned<T> &astruct,
                                     misaligned<T> &istruct, int model) {
  static constexpr T val = V >> right_shift<T>();
  for (int n = 0; n < kIterations; ++n) {
    __atomic_fetch_add(&astruct.data, val, model);
    istruct.data += val;
  }
}

template <typename T>
void test_int_misaligned_fetch_add(misaligned<T> &astruct,
                                   misaligned<T> &istruct) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int model : atomic_fetch_models) {
    astruct.data = 0;
    istruct.data = 0;
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_int_misaligned_fetch_add<T>, std::ref(astruct),
                        std::ref(istruct), model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    std::cout << "FETCH ADD: "
              << "atomic: " << astruct.data << " "
              << "nonatomic: " << istruct.data << "\n";
    if (lt(astruct.data, istruct.data) || astruct.data != val * kExpected) {
      fail();
    }
  }
}

template <typename T>
void looper_int_misaligned_fetch_sub(misaligned<T> &astruct,
                                     misaligned<T> &istruct, int model) {
  static constexpr T val = V >> right_shift<T>();
  for (int n = 0; n < kIterations; ++n) {
    __atomic_fetch_sub(&astruct.data, val, model);
    istruct.data -= val;
  }
}

template <typename T>
void test_int_misaligned_fetch_sub(misaligned<T> &astruct,
                                   misaligned<T> &istruct) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int model : atomic_fetch_models) {
    astruct.data = val * kExpected;
    istruct.data = val * kExpected;
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_int_misaligned_fetch_sub<T>, std::ref(astruct),
                        std::ref(istruct), model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    std::cout << "FETCH SUB: "
              << "atomic: " << astruct.data << " "
              << "nonatomic: " << istruct.data << "\n";
    if (lt(istruct.data, astruct.data) || astruct.data != 0) {
      fail();
    }
  }
}

// See int_aligned_test.cc for an explanation of AND tests.
template <typename T>
void __attribute__((optnone)) looper_int_misaligned_fetch_and(
    const int id, misaligned<T> &astruct, misaligned<T> &istruct,
    T *acnt, T *icnt, int model) {
  T desired, expected = 0;
  for (int n = 0; n < kIterations; ++n) {
    const T mask = 1 << id;
    __atomic_fetch_and(&astruct.data, ~mask, model);
    if (~astruct.data & mask) {
      __atomic_fetch_add(acnt, 1, model);
      do {
        desired = expected | mask;
      } while (!__atomic_compare_exchange(&astruct.data, &expected, &desired, 
                                          true, model, model));
    }
    istruct.data &= ~mask;
    if (~istruct.data & mask) {
      __atomic_fetch_add(icnt, 1, model);
      istruct.data |= mask;
    }
  }
}

template <typename T>
void test_int_misaligned_fetch_and(misaligned<T> &astruct,
                                   misaligned<T> &istruct) {
  std::vector<std::thread> pool;
  for (int model : atomic_fetch_models) {
    T acnt = 0, icnt = 0;
    astruct.data = ~0, istruct.data = ~0;
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_int_misaligned_fetch_and<T>, n,
                        std::ref(astruct), std::ref(istruct),
                        &acnt, &icnt, model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    std::cout << "FETCH AND: "
              << "atomic: " << acnt << " "
              << "nonatomic: " << icnt << "\n";
    if (acnt != kExpected) {
      fail();
    }
  }
}

// See int_aligned_test.cc for an explanation of OR tests.
template <typename T>
void __attribute__((optnone)) looper_int_misaligned_fetch_or(
    const int id, misaligned<T> &astruct, misaligned<T> &istruct,
    T *acnt, T *icnt, int model) {
  T desired, expected = 0;
  for (int n = 0; n < kIterations; ++n) {
    const T mask = 1 << id;
    __atomic_fetch_or(&astruct.data, mask, model);
    if (astruct.data & mask) {
      __atomic_fetch_add(acnt, 1, model);
      do {
        desired = expected & ~mask;
      } while (!__atomic_compare_exchange(&astruct.data, &expected, &desired, 
                                          true, model, model));
    }
    istruct.data |= mask;
    if (istruct.data & mask) {
      __atomic_fetch_add(icnt, 1, model);
      istruct.data &= ~mask;
    }
  }
}

template <typename T>
void test_int_misaligned_fetch_or(misaligned<T> &astruct,
                                  misaligned<T> &istruct) {
  std::vector<std::thread> pool;
  for (int model : atomic_fetch_models) {
    T acnt = 0, icnt = 0;
    astruct.data = 0, istruct.data = 0;
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_int_misaligned_fetch_or<T>, n,
                        std::ref(astruct), std::ref(istruct),
                        &acnt, &icnt, model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    std::cout << "FETCH OR: "
              << "atomic: " << acnt << " "
              << "nonatomic: " << icnt << "\n";
    if (acnt != kExpected) {
      fail();
    }
  }
}

template <typename T>
void looper_int_misaligned_fetch_xor(misaligned<T> &astruct,
                                     misaligned<T> &istruct, int model) {
  for (int n = 0; n < kIterations; ++n) {
    __atomic_fetch_xor(&astruct.data, n, model);
    istruct.data ^= n;
  }
}

template <typename T>
void test_int_misaligned_fetch_xor(misaligned<T> &astruct,
                                   misaligned<T> &istruct) {
  std::vector<std::thread> pool;
  for (int model : atomic_fetch_models) {
    astruct.data = 0;
    istruct.data = 0;
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_int_misaligned_fetch_xor<T>, std::ref(astruct),
                        std::ref(istruct), model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    std::cout << "FETCH XOR: "
              << "atomic: " << astruct.data << " "
              << "nonatomic: " << istruct.data << "\n";
    if (astruct.data != 0) {
      fail();
    }
  }
}

// See numeric.h for an explanation of numeric xchg tests.
template <typename T>
void looper_int_misaligned_xchg_atomic(misaligned<T> &astruct, int model) {
  static constexpr T val = V >> right_shift<T>();
  __int128_t error = 0;
  T next = astruct.data + val;
  T result;
  for (int n = 0; n < kIterations; ++n) {
    __atomic_exchange(&astruct.data, &next, &result, model);
    error +=
        static_cast<__int128_t>(next) - static_cast<__int128_t>(result + val);
    next = result + val;
  }
  __atomic_fetch_sub(&astruct.data, static_cast<T>(error), model);
}

template <typename T>
void looper_int_misaligned_xchg_nonatomic(misaligned<T> &istruct, int model) {
  static constexpr T val = V >> right_shift<T>();
  __int128_t error = 0;
  T next = istruct.data + val;
  T result;
  for (int n = 0; n < kIterations; ++n) {
    result = istruct.data;
    istruct.data = next;
    error +=
        static_cast<__int128_t>(next) - static_cast<__int128_t>(result + val);
    next = result + val;
  }
  __atomic_fetch_sub(&istruct.data, static_cast<T>(error), model);
}

template <typename T>
void test_int_misaligned_xchg(misaligned<T> &astruct, misaligned<T> &istruct) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int model : atomic_exchange_models) {
    astruct.data = 0;
    istruct.data = 0;
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_int_misaligned_xchg_atomic<T>, std::ref(astruct),
                        model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_int_misaligned_xchg_nonatomic<T>,
                        std::ref(istruct), model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    std::cout << "XCHG: ";
    print_int(astruct.data, istruct.data);
    if (lt(astruct.data, istruct.data) || astruct.data != val * kExpected) {
      fail();
    }
  }
}

// See numeric.h for an explanation of numeric xchg tests.
template <typename T>
void looper_int_misaligned_xchg_n(misaligned<T> &astruct, int model) {
  static constexpr T val = V >> right_shift<T>();
  __int128_t error = 0;
  T next = astruct.data + val;
  for (int n = 0; n < kIterations; ++n) {
    T result = __atomic_exchange_n(&astruct.data, next, model);
    error +=
        static_cast<__int128_t>(next) - static_cast<__int128_t>(result + val);
    next = result + val;
  }
  __atomic_fetch_sub(&astruct.data, static_cast<T>(error), model);
}

template <typename T>
void test_int_misaligned_xchg_n(misaligned<T> &astruct,
                                misaligned<T> &istruct) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int model : atomic_exchange_models) {
    astruct.data = 0;
    istruct.data = 0;
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_int_misaligned_xchg_n<T>, std::ref(astruct),
                        model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    for (int n = 0; n < kThreads; ++n) {
      pool.emplace_back(looper_int_misaligned_xchg_nonatomic<T>,
                        std::ref(istruct), model);
    }
    for (int n = 0; n < kThreads; ++n) {
      pool[n].join();
    }
    pool.clear();
    std::cout << "XCHG_N: ";
    print_int(astruct.data, istruct.data);
    if (lt(astruct.data, istruct.data) || astruct.data != val * kExpected) {
      fail();
    }
  }
}

// See numeric.h for an explanation of numeric cmpxchg tests.
template <typename T>
void looper_int_misaligned_cmpxchg(misaligned<T> &astruct,
                                   misaligned<T> &istruct, int success_model,
                                   int fail_model) {
  static constexpr T val = V >> right_shift<T>();
  for (int n = 0; n < kIterations; ++n) {
    T desired, expected = 0;
    do {
      desired = expected + val;
    } while (!__atomic_compare_exchange(&astruct.data, &expected, &desired,
                                        true, success_model, fail_model));
    istruct.data += val;
  }
}

template <typename T>
void test_int_misaligned_cmpxchg(misaligned<T> &astruct,
                                 misaligned<T> &istruct) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int success_model : atomic_compare_exchange_models) {
    for (int fail_model : atomic_compare_exchange_models) {
      astruct.data = 0;
      istruct.data = 0;
      for (int n = 0; n < kThreads; ++n) {
        pool.emplace_back(looper_int_misaligned_cmpxchg<T>, std::ref(astruct),
                          std::ref(istruct), success_model, fail_model);
      }
      for (int n = 0; n < kThreads; ++n) {
        pool[n].join();
      }
      pool.clear();
      std::cout << "CMPXCHG: ";
      print_int(astruct.data, istruct.data);
      if (lt(astruct.data, istruct.data) ||
          astruct.data != static_cast<T>(val) * kExpected) {
        fail();
      }
    }
  }
}

// See numeric.h for an explanation of numeric cmpxchg tests.
template <typename T>
void looper_int_misaligned_cmpxchg_n(misaligned<T> &astruct,
                                     misaligned<T> &istruct, int success_model,
                                     int fail_model) {
  static constexpr T val = V >> right_shift<T>();
  for (int n = 0; n < kIterations; ++n) {
    T desired, expected = 0;
    do {
      desired = expected + val;
    } while (!__atomic_compare_exchange_n(&astruct.data, &expected, desired,
                                          true, success_model, fail_model));
    istruct.data += val;
  }
}

template <typename T>
void test_int_misaligned_cmpxchg_n(misaligned<T> &astruct,
                                   misaligned<T> &istruct) {
  static constexpr T val = V >> right_shift<T>();
  std::vector<std::thread> pool;
  for (int success_model : atomic_compare_exchange_models) {
    for (int fail_model : atomic_compare_exchange_models) {
      astruct.data = 0;
      istruct.data = 0;
      for (int n = 0; n < kThreads; ++n) {
        pool.emplace_back(looper_int_misaligned_cmpxchg_n<T>, std::ref(astruct),
                          std::ref(istruct), success_model, fail_model);
      }
      for (int n = 0; n < kThreads; ++n) {
        pool[n].join();
      }
      pool.clear();
      std::cout << "CMPXCHG_N: ";
      print_int(astruct.data, istruct.data);
      if (lt(astruct.data, istruct.data) ||
          astruct.data != static_cast<T>(val) * kExpected) {
        fail();
      }
    }
  }
}

void test_misaligned_int() {
#define INT_SUITE(type)                                    \
  {                                                        \
    printf("Testing misaligned " #type "\n");              \
    misaligned<type> astruct;                              \
    misaligned<type> istruct;                              \
    test_int_misaligned_fetch_add<type>(astruct, istruct); \
    test_int_misaligned_fetch_sub<type>(astruct, istruct); \
    test_int_misaligned_fetch_and<type>(astruct, istruct); \
    test_int_misaligned_fetch_or<type>(astruct, istruct);  \
    test_int_misaligned_fetch_xor<type>(astruct, istruct); \
    test_int_misaligned_xchg<type>(astruct, istruct);      \
    test_int_misaligned_xchg_n<type>(astruct, istruct);    \
    test_int_misaligned_cmpxchg<type>(astruct, istruct);   \
    test_int_misaligned_cmpxchg_n<type>(astruct, istruct); \
  }
  INT_SUITE(uint32_t);
  INT_SUITE(uint64_t);
  INT_SUITE(int32_t);
  INT_SUITE(int64_t);
#undef INT_SUITE

#if TEST16
#define INT_SUITE(type)                                    \
  {                                                        \
    printf("Testing misaligned " #type "\n");              \
    misaligned<type> astruct;                              \
    misaligned<type> istruct;                              \
    test_int_misaligned_xchg<type>(astruct, istruct);      \
    test_int_misaligned_xchg_n<type>(astruct, istruct);    \
    test_int_misaligned_cmpxchg<type>(astruct, istruct);   \
    test_int_misaligned_cmpxchg_n<type>(astruct, istruct); \
  }
  INT_SUITE(__uint128_t);
  INT_SUITE(__int128_t);
#undef INT_SUITE
#endif
}

int main() {
  printf("%d threads; %d iterations each; total of %d\n", kThreads, kIterations,
         kExpected);

  test_misaligned_int();
  printf("PASSED\n");
}

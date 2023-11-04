//===- ExecutionEngine.h - MLIR Execution engine and utils -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a JIT-backed execution engine for MLIR modules.
//
//===----------------------------------------------------------------------===//

#ifndef HANDS_ON_MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_
#define HANDS_ON_MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"

#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

namespace mlir {
namespace hands_on_mlir {

class ExecutionEngine {
public:
  using InvokeFn = void (*)(void **);

  ExecutionEngine(const std::string &s) {
    handle = dlopen(s.c_str(), RTLD_LAZY);
    if (!handle) {
      std::cerr << dlerror() << std::endl;
    }
  }

  /// Trait that defines how a given type is passed to the JIT code. This
  /// defaults to passing the address but can be specialized.
  template <typename T> struct Argument {
    static void pack(SmallVectorImpl<void *> &args, T &val) {
      args.push_back(&val);
    }
  };

  /// Tag to wrap an output parameter when invoking a jitted function.
  template <typename T> struct Result {
    Result(T &result) : value(result) {}
    T &value;
  };

  /// Helper function to wrap an output operand when using
  /// ExecutionEngine::invoke.
  template <typename T> static Result<T> result(T &t) { return Result<T>(t); }

  // Specialization for output parameter: their address is forwarded directly to
  // the native code.
  template <typename T> struct Argument<Result<T>> {
    static void pack(SmallVectorImpl<void *> &args, Result<T> &result) {
      args.push_back(&result.value);
    }
  };

  llvm::Expected<void *> lookup(StringRef name) const;

  /// Invokes the function with the given name passing it the list of arguments
  /// by value. Function result can be obtain through output parameter using the
  /// `Result` wrapper defined above. For example:
  ///
  ///     func @foo(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface }
  ///
  /// can be invoked:
  ///
  ///     int32_t result = 0;
  ///     llvm::Error error = jit->invoke("foo", 42,
  ///                                     result(result));
  template <typename... Args>
  llvm::Error invoke(StringRef funcName, Args... args) {
    const std::string adapterName =
        std::string("_hom_ciface_") + funcName.str();
    llvm::SmallVector<void *> argsArray;
    // Pack every arguments in an array of pointers. Delegate the packing to a
    // trait so that it can be overridden per argument type.
    (Argument<Args>::pack(argsArray, args), ...);
    std::cout << argsArray.size() << std::endl;
    return invokePacked(adapterName, argsArray);
  }

private:
  void *handle;

  llvm::Error invokePacked(StringRef adapterName,
                           MutableArrayRef<void *> argsArray);

  llvm::Expected<InvokeFn> lookupPacked(StringRef name) const;

  llvm::Expected<void *> lookupHandle(StringRef name) const;
};

} // namespace hands_on_mlir
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_

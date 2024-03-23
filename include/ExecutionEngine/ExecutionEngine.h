#ifndef HANDS_ON_MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_
#define HANDS_ON_MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_

#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

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

  ~ExecutionEngine() {
    for (auto it : argMap) {
      for (size_t i = 0; i < it.second.argNum; i++) {
        // Only free the descriptor here, the actual memory space should be
        // managed by mlir c function.
        free(it.second.data.get()[i].descriptor);
      }
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

  template <typename T> struct PackedArguments {
    PackedArguments(size_t argNumber) : argNum(argNumber) {
      data.reset(new T[argNum]);
    }
    PackedArguments() = delete;
    size_t argNum;
    std::shared_ptr<T[]> data;
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

  template <typename T> struct Argument<PackedArguments<T>> {
    static void pack(SmallVectorImpl<void *> &args, PackedArguments<T> &val) {
      for (size_t i = 0; i < val.argNum; i++) {
        Argument<T>::pack(args, val.data.get()[i]);
      }
    }
  };

  template <typename T> struct Argument<Result<PackedArguments<T>>> {
    static void pack(SmallVectorImpl<void *> &args,
                     Result<PackedArguments<T>> &result) {
      Argument<PackedArguments<T>>::pack(args, result.value);
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
  ///
  /// The first call of this function will call the init function to create the
  /// tensor managed by hands-on-mlir compiler. Currently we don't have explicit
  /// init function for it. The second call would use the cached result.
  template <typename... Args>
  llvm::Error invoke(StringRef funcName, Args... args) {
    auto it = argMap.find(funcName.str());
    if (it == argMap.end()) {
      auto packedArgs = invokeInit(funcName);
      if (!packedArgs) {
        return packedArgs.takeError();
      }
      argMap.insert({funcName.str(), packedArgs.get()});
    }

    llvm::SmallVector<void *> argsArray;
    // Pack every arguments in an array of pointers. Delegate the packing to a
    // trait so that it can be overridden per argument type.
    Argument<PackedArguments<C_UnrankedMemRefType>>::pack(
        argsArray, argMap.at(funcName.str()));
    (Argument<Args>::pack(argsArray, args), ...);
    return invokePacked(funcName, argsArray);
  }

private:
  void *handle;

  // Map to store the extra tensor(e.g. weight tensor) managed by hands on mlir
  // compiler.
  std::unordered_map<std::string, PackedArguments<C_UnrankedMemRefType>> argMap;

  llvm::Error invokePacked(StringRef funcName,
                           MutableArrayRef<void *> argsArray);

  llvm::Expected<PackedArguments<C_UnrankedMemRefType>>
  invokeInit(StringRef adapteName);

  template <typename... Args>
  llvm::Error invokeInternal(StringRef funcName, Args... args) {
    llvm::SmallVector<void *> argsArray;
    // Pack every arguments in an array of pointers. Delegate the packing to a
    // trait so that it can be overridden per argument type.
    (Argument<Args>::pack(argsArray, args), ...);
    return invokePacked(funcName, argsArray);
  }

  llvm::Expected<InvokeFn> lookupPacked(StringRef name) const;

  llvm::Expected<void *> lookupHandle(StringRef name) const;
};

template <> struct ExecutionEngine::Argument<C_UnrankedMemRefType> {
  static void pack(SmallVectorImpl<void *> &args, C_UnrankedMemRefType &val) {
    args.emplace_back(&val.rank);
    args.emplace_back(&val.descriptor);
  }
};

} // namespace hands_on_mlir
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_EXECUTIONENGINE_H_

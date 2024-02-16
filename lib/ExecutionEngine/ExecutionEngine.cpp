//===- ExecutionEngine.cpp - MLIR Execution engine and utils --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the execution engine for MLIR modules based on LLVM Orc
// JIT engine.
//
//===----------------------------------------------------------------------===//
#include "ExecutionEngine/ExecutionEngine.h"
#include "Conversions/Function/FunctionUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <dlfcn.h>

#define DEBUG_TYPE "execution-engine"

using namespace mlir;
using namespace hands_on_mlir;
using llvm::Error;
using llvm::Expected;
using llvm::StringError;

/// Wrap a string into an llvm::StringError.
static Error makeStringError(const Twine &message) {
  return llvm::make_error<StringError>(message.str(),
                                       llvm::inconvertibleErrorCode());
}

Expected<void *> ExecutionEngine::lookupHandle(StringRef name) const {
  if (handle) {
    auto symbol = dlsym(handle, name.str().c_str());
    auto err = dlerror();
    if (err) {
      return makeStringError(err);
    } else {
      return symbol;
    }
  }
  return makeStringError("Handle is invalid.");
}

Expected<ExecutionEngine::InvokeFn>
ExecutionEngine::lookupPacked(StringRef name) const {
  auto result = lookup(name);
  if (!result)
    return result.takeError();
  return reinterpret_cast<InvokeFn>(result.get());
}

Expected<void *> ExecutionEngine::lookup(StringRef name) const {
  const std::string lookupName = "_hom_ciface_" + name.str();
  auto expectedSymbol = lookupHandle(lookupName);

  if (!expectedSymbol) {
    std::string errorMessage;
    llvm::raw_string_ostream os(errorMessage);
    llvm::handleAllErrors(expectedSymbol.takeError(),
                          [&os](llvm::ErrorInfoBase &ei) { ei.log(os); });
    return makeStringError(os.str());
  }

  if (void *fptr = expectedSymbol.get())
    return fptr;
  return makeStringError("looked up function is null");
}

Error ExecutionEngine::invokePacked(StringRef name,
                                    MutableArrayRef<void *> args) {
  auto expectedFPtr = lookupPacked(name);
  if (!expectedFPtr)
    return expectedFPtr.takeError();
  auto fptr = *expectedFPtr;

  (*fptr)(args.data());

  return Error::success();
}

Expected<ExecutionEngine::PackedArguments<C_UnrankedMemRefType>>
ExecutionEngine::invokeInit(StringRef name) {
  SmallVector<char> concatName;

  int32_t argNum;

  auto argNumError =
      invokeInternal((name + kArgNum).toStringRef(concatName), result(argNum));
  if (!argNumError) {
    llvm::handleAllErrors(std::move(argNumError));
  }

  concatName.clear();

  PackedArguments<C_UnrankedMemRefType> packedArgs(argNum);

  auto packedMemRefRes = invokeInternal((name + kInit).toStringRef(concatName),
                                        result(packedArgs));

  if (packedMemRefRes) {
    return packedMemRefRes;
  } else {
    return packedArgs;
  }
}

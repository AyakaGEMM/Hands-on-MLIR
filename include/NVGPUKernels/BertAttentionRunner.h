#pragma once

#include "ExecutionEngine/HandsOnRunnerUtils.h"
#include "NVGPUKernels/OperationRunner.h"
#include "NVGPUKernels/Utils.h"
#include "transformer_engine/fused_attn.h"
#include "transformer_engine/transformer_engine.h"
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace mlir {
namespace hands_on_mlir {
namespace homnvgpu_kernel {

template <typename ElementType>
class BertAttentionRunner : public OperationRunner {
  using TensorWrapper = transformer_engine::TensorWrapper;

  void getWorkSpace(size_t bs, size_t seq_len,
                    const std::vector<size_t> &qkv_shape,
                    const std::vector<size_t> &bias_shape,
                    const std::vector<size_t> &output_shape, size_t head_num,
                    float scale, TensorWrapper &workspace,
                    NVTETensorPack &aux_output_tensors) {
    TensorWrapper qkv(nullptr, qkv_shape,
                      NVTEWrapperDTypeMap<ElementType>::kType);
    TensorWrapper bias(nullptr, bias_shape,
                       NVTEWrapperDTypeMap<ElementType>::kType);
    TensorWrapper cu_seqlen(nullptr, std::vector<size_t>{bs + 1},
                            NVTEWrapperDTypeMap<int32_t>::kType);
    TensorWrapper s(nullptr, std::vector<size_t>{1},
                    NVTEWrapperDTypeMap<ElementType>::kType);
    TensorWrapper output(nullptr, output_shape,
                         NVTEWrapperDTypeMap<ElementType>::kType);
    TensorWrapper rng_state(nullptr, std::vector<size_t>{2},
                            NVTEWrapperDTypeMap<int64_t>::kType);

    nvte_tensor_pack_create(&aux_output_tensors);

    assert(workspace.dptr() == nullptr);

    nvte_fused_attn_fwd_qkvpacked(
        qkv.data(), bias.data(), s.data(), output.data(), &aux_output_tensors,
        cu_seqlen.data(), rng_state.data(), seq_len, false, scale, 0,
        NVTE_BS3HD, NVTE_Bias_Type::NVTE_NO_BIAS,
        NVTE_Mask_Type::NVTE_PADDING_MASK, workspace.data(), nullptr);
  }

  std::tuple<TensorWrapper, TensorWrapper, TensorWrapper, TensorWrapper,
             TensorWrapper, size_t, TensorWrapper, TensorWrapper>
  construct_tensors(int64_t rankA, void *dstA, int64_t rankSeqlen,
                    void *dstSeqlen, int64_t rankOut, void *dstOut, float scale,
                    size_t head_num, NVTETensorPack &aux_output_tensors) {
    auto A = convertToDynamicMemRefType<ElementType>(rankA, dstA);
    auto SeqLen = convertToDynamicMemRefType<int32_t>(rankSeqlen, dstSeqlen);
    auto Out = convertToDynamicMemRefType<ElementType>(rankOut, dstOut);

    assert(A.rank == 3);
    assert(A.sizes[2] % (head_num * 3) == 0);

    assert(Out.rank == 3);
    assert(Out.sizes[0] == A.sizes[0]);
    assert(Out.sizes[1] == A.sizes[1]);
    assert(Out.sizes[2] * 3 == A.sizes[2]);

    size_t bs = A.sizes[0], seq_len = A.sizes[1],
           head_size = A.sizes[2] / head_num / 3;

    assert(SeqLen.rank == 1);
    assert(SeqLen.sizes[0] == bs + 1);

    std::vector<size_t> qkv_shape = {bs * seq_len, 3, head_num, head_size};

    std::vector<size_t> output_shape = {bs * seq_len, head_num, head_size};

    std::vector<size_t> bias_shape = {1, head_num, seq_len, seq_len};
    TensorWrapper workspace;

    getWorkSpace(bs, seq_len, qkv_shape, bias_shape, output_shape, head_num,
                 scale, workspace, aux_output_tensors);

    TensorWrapper qkv(A.data, qkv_shape,
                      NVTEWrapperDTypeMap<ElementType>::kType);
    TensorWrapper bias(nullptr, bias_shape,
                       NVTEWrapperDTypeMap<ElementType>::kType); // Not used.
    TensorWrapper cu_seqlen(SeqLen.data, std::vector<size_t>{bs + 1},
                            NVTEWrapperDTypeMap<int32_t>::kType);
    TensorWrapper s(nullptr, std::vector<size_t>{1},
                    NVTEWrapperDTypeMap<ElementType>::kType); // Not used.
    TensorWrapper output(Out.data, output_shape,
                         NVTEWrapperDTypeMap<ElementType>::kType);
    TensorWrapper rng_state(
        nullptr, std::vector<size_t>{2},
        NVTEWrapperDTypeMap<int64_t>::kType); // Not used for inference. This
                                              // state is for dropout.

    size_t workspace_size =
        workspace.shape().data[0] * getNVTEWrapperDTypeSize(workspace.dtype());

    auto workspace_buffer = getDummyPointer(workspace_size);
    workspace = TensorWrapper(workspace_buffer.get(), workspace.shape(),
                              workspace.dtype());

    return {std::move(qkv),       std::move(bias),     std::move(cu_seqlen),
            std::move(s),         std::move(output),   seq_len,
            std::move(rng_state), std::move(workspace)};
  }

public:
  Status run(int64_t rankA, void *dstA, int64_t rankSeqlen, void *dstSeqlen,
             int64_t rankOut, void *dstOut, float scale, int64_t headNum) {

    NVTETensorPack aux_output_tensors;
    auto [qkv, bias, cu_seqlen, s, output, seq_len, rng_state, workspace] =
        construct_tensors(rankA, dstA, rankSeqlen, dstSeqlen, rankOut, dstOut,
                          scale, headNum, aux_output_tensors);

    nvte_fused_attn_fwd_qkvpacked(
        qkv.data(), bias.data(), s.data(), output.data(), &aux_output_tensors,
        cu_seqlen.data(), rng_state.data(), seq_len, false, scale, 0,
        NVTE_BS3HD, NVTE_Bias_Type::NVTE_NO_BIAS,
        NVTE_Mask_Type::NVTE_PADDING_MASK, workspace.data(), nullptr);

    nvte_tensor_pack_destroy(&aux_output_tensors);

    auto error = cudaGetLastError();

    if (error != cudaSuccess) {
      return Status::kErrorInternal;
    }

    return Status::kSuccess;
  }
};

} // namespace homnvgpu_kernel
} // namespace hands_on_mlir
} // namespace mlir

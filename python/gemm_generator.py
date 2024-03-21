import os
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(__file__), "../thirdparty/cutlass/python/cutlass_library"
    )
)

from generator import CreateGemmOperator, CudaToolkitVersionSatisfies, define_parser
from hom_manifest import EpilogueFunctor, HOM_Manifest
from library import (
    DataType,
    LayoutType,
    MathInstruction,
    MathOperation,
    OpcodeClass,
    TileDescription,
)


def GenerateSM70_TensorOp_884(manifest: HOM_Manifest, cuda_version):
    if not CudaToolkitVersionSatisfies(cuda_version, 10, 1):
        return

    layouts = [
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    ]

    math_instructions = [
        MathInstruction(
            [8, 8, 4],
            DataType.f16,
            DataType.f16,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        # No fp16 acc
        # MathInstruction(
        #     [8, 8, 4],
        #     DataType.f16,
        #     DataType.f16,
        #     DataType.f16,
        #     OpcodeClass.TensorOp,
        #     MathOperation.multiply_add,
        # ),
    ]

    min_cc = 70
    max_cc = 75

    alignment_constraints = [8, 4, 2, 1]
    epilogue_functors = [EpilogueFunctor.LinearCombination, EpilogueFunctor.GELU]

    for math_inst in math_instructions:
        for epilogue_functor in epilogue_functors:
            tile_descriptions = [
                TileDescription(
                    [256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription(
                    [128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc
                ),
                TileDescription(
                    [128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription([256, 64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([128, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
            ]

            data_type = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_accumulator,
                math_inst.element_accumulator,
            ]

            CreateGemmOperator(
                manifest,
                layouts,
                tile_descriptions,
                data_type,
                alignment_constraints,
                epilogue_functor=epilogue_functor,
            )


def GenerateSM75_TensorOp_1688(manifest, cuda_version):

    if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
        return

    layouts = [
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    ]

    math_instructions = [
        MathInstruction(
            [16, 8, 8],
            DataType.f16,
            DataType.f16,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        # No fp16 acc
        # MathInstruction(
        #     [16, 8, 8],
        #     DataType.f16,
        #     DataType.f16,
        #     DataType.f16,
        #     OpcodeClass.TensorOp,
        #     MathOperation.multiply_add,
        # ),
    ]

    min_cc = 75
    max_cc = 1024

    alignment_constraints = [8, 4, 2, 1]
    epilogue_functors = [EpilogueFunctor.LinearCombination, EpilogueFunctor.GELU]

    for math_inst in math_instructions:
        for epilogue_functor in epilogue_functors:
            tile_descriptions = [
                TileDescription(
                    [256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription(
                    [128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc
                ),
                TileDescription(
                    [128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription([64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
                TileDescription([256, 64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([128, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 128, 64], 2, [1, 2, 2], math_inst, min_cc, max_cc),
            ]

            data_type = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_accumulator,
                math_inst.element_accumulator,
            ]

            CreateGemmOperator(
                manifest,
                layouts,
                tile_descriptions,
                data_type,
                alignment_constraints,
                epilogue_functor=epilogue_functor,
            )

            # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
            if math_inst.element_a != math_inst.element_accumulator:

                data_type_mixed = [
                    math_inst.element_a,
                    math_inst.element_b,
                    math_inst.element_a,
                    math_inst.element_accumulator,
                ]

                CreateGemmOperator(
                    manifest,
                    layouts,
                    tile_descriptions,
                    data_type_mixed,
                    alignment_constraints,
                    epilogue_functor=epilogue_functor,
                )


def GenerateSM80_TensorOp_16816(manifest, cuda_version):

    if not CudaToolkitVersionSatisfies(cuda_version, 11, 0):
        return

    layouts = [
        (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
        (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    ]

    math_instructions = [
        MathInstruction(
            [16, 8, 16],
            DataType.f16,
            DataType.f16,
            DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add,
        ),
        # No fp16 acc
        # MathInstruction(
        #     [16, 8, 16],
        #     DataType.f16,
        #     DataType.f16,
        #     DataType.f16,
        #     OpcodeClass.TensorOp,
        #     MathOperation.multiply_add,
        # ),
        # MathInstruction(
        #     [16, 8, 16],
        #     DataType.bf16,
        #     DataType.bf16,
        #     DataType.f32,
        #     OpcodeClass.TensorOp,
        #     MathOperation.multiply_add,
        # ),
    ]

    min_cc = 80
    max_cc = 1024

    alignment_constraints = [8, 4, 2]
    epilogue_functors = [EpilogueFunctor.LinearCombination, EpilogueFunctor.GELU]

    for math_inst in math_instructions:
        for epilogue_functor in epilogue_functors:
            tile_descriptions = [
                TileDescription(
                    [256, 128, 32], 3, [4, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription(
                    [128, 256, 32], 3, [2, 4, 1], math_inst, min_cc, max_cc
                ),
                TileDescription([256, 64, 32], 3, [4, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([256, 64, 32], 4, [4, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 256, 32], 4, [1, 4, 1], math_inst, min_cc, max_cc),
                TileDescription(
                    [128, 128, 32], 3, [2, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription(
                    [128, 128, 32], 4, [2, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription(
                    [128, 128, 32], 5, [2, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription([128, 64, 32], 6, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 128, 32], 6, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 64, 32], 10, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription(
                    [256, 128, 64], 3, [4, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription(
                    [128, 256, 64], 3, [2, 4, 1], math_inst, min_cc, max_cc
                ),
                TileDescription([256, 64, 64], 4, [4, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 256, 64], 4, [1, 4, 1], math_inst, min_cc, max_cc),
                TileDescription(
                    [128, 128, 64], 4, [2, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription([256, 64, 64], 3, [4, 1, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 256, 64], 3, [1, 4, 1], math_inst, min_cc, max_cc),
                TileDescription(
                    [128, 128, 64], 3, [2, 2, 1], math_inst, min_cc, max_cc
                ),
                TileDescription([128, 64, 64], 3, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 128, 64], 3, [2, 2, 1], math_inst, min_cc, max_cc),
                TileDescription([64, 64, 64], 5, [2, 2, 1], math_inst, min_cc, max_cc),
            ]

            data_type = [
                math_inst.element_a,
                math_inst.element_b,
                math_inst.element_accumulator,
                math_inst.element_accumulator,
            ]

            CreateGemmOperator(
                manifest,
                layouts,
                tile_descriptions,
                data_type,
                alignment_constraints,
                epilogue_functor=epilogue_functor,
            )

            # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
            if math_inst.element_a != math_inst.element_accumulator:

                data_type_mixed = [
                    math_inst.element_a,
                    math_inst.element_b,
                    math_inst.element_a,
                    math_inst.element_accumulator,
                ]

                CreateGemmOperator(
                    manifest,
                    layouts,
                    tile_descriptions,
                    data_type_mixed,
                    alignment_constraints,
                    epilogue_functor=epilogue_functor,
                )


if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()

    manifest = HOM_Manifest(args)

    GenerateSM70_TensorOp_884(manifest, args.cuda_version)
    GenerateSM75_TensorOp_1688(manifest, args.cuda_version)
    GenerateSM80_TensorOp_16816(manifest, args.cuda_version)

    manifest.emit()

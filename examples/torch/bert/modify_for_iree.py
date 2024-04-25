for name in ["bert-base-uncased", "bert-large-uncased"]:
    for bs in [1, 8, 16, 32]:
        for len in [64, 128]:

            with open(f"{name}_{bs}_{len}.mlir", "r") as fl_in:
                with open(f"iree_{name}_{bs}_{len}.mlir", "w") as fl_out:
                    found = False
                    for line in fl_in.readlines():
                        if not found and "func.func @forward" in line:
                            idx = line.find(") ->")
                            line = (
                                line[:idx]
                                + ", %arg3: !hal.buffer {iree.abi.output = 0 : index}"
                                + line[idx:]
                            )
                            found = True
                        print(line, file=fl_out, end="")

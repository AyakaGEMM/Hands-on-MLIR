res = {"base": {}, "large": {}}

with open("iree_new.log", "r") as fl:
    state = 0
    mode = ""
    bs = 0
    seq = 0
    for line in fl.readlines():
        line = line.strip()
        if "Compilation successful:" in line:
            mode = "base" if "bert-base" in line else "large"
            idx = line.find("d_")
            line = line[idx + 2 :]
            idx = line.find("_")
            bs = int(line[:idx])
            seq = int(line[idx + 1 : -5])
        elif "BM_forward/process_time/real_time_mean" in line:
            idx = line.find("ms")
            res[mode][(bs, seq)] = float(line[idx - 5 : idx])

for mode in ["base", "large"]:
    for bs in [1, 8, 16, 32]:
        for seq in [64, 128]:
            try:
                print(f"{mode} ({bs}, {seq}): {res[mode][(bs, seq)]}")
            except:
                print("No Info")

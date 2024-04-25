res = {"base": {}, "large": {}}

with open("hom_sync.log", "r") as fl:
    bs = 0
    seq = 0
    mode = ""
    autotune = False
    p = 0
    for line in fl.readlines():
        line = line.strip()
        if "Tag" in line:
            mode = "base" if "base" in line else "large"
            line = line.split()
            bs = int(line[2])
            seq = int(line[3])
            autotune = int(line[1])
            if (bs, seq, autotune) not in res[mode]:
                res[mode][(bs, seq, autotune)] = "err"
            p = 2
            continue
        elif p == 1 and "E2E" in line:
            idx = line.find(":")
            if res[mode][(bs, seq, autotune)] == "err":
                res[mode][(bs, seq, autotune)] = float(line[idx + 1 : -2])

        p -= 1
        pre = line

for mode in ["base", "large"]:
    for bs in [1, 8, 16, 32]:
        for seq in [64, 128]:
            for auto in [0, 1]:
                try:
                    print(f"{mode} ({bs}, {seq}, {auto}): {res[mode][(bs, seq, auto)]}")
                except:
                    print("No info")

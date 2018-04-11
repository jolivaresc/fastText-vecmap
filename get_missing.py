import utils


w2v = []
with open("datasets/en-it/it.200k.300d.w2v", "r", encoding="utf-8", errors="surrogateescape") as f:
    header_w2v = f.readline().split()
    for i in range(int(header_w2v[0])):
        w, _ = f.readline().split(" ", 1)
        w2v.append(w)

fst = []
with open("datasets/en-it/it.200k.300d.tmp.fst", "r", encoding="utf-8", errors="surrogateescape") as f:
    #header_fst=f.readline().split()
    for i in range(163031):
        w, _ = f.readline().split(" ", 1)
        fst.append(w)

for i in w2v:
    if i not in fst:
        try:
            print(i)
        except Exception as identifier:
            pass

import pandas as pd
from pandas import DataFrame
from utils.filegetter import afns
from os.path import commonprefix

files = afns()
common = commonprefix(files)
clen = len(common)
# names = {}
frames:dict[str,DataFrame] = {}
cols = set()
for file in files:
    if not file.endswith(".csv"):
        raise Exception("must all be csv files!")
    name = file[clen:].lstrip(' _').split(".csv")[0]
    data = pd.read_csv(file)
    if name in frames:
        raise Exception(f"two identical filenames given: {common+name}") #I think this is impossible but sanity check
    frames[name] = data
    cols.update(data.columns)

for col in cols:
    l = {}
    for name,frame in frames.items():
        if col in frame.columns:
            l[name] = frame[col]
    res = DataFrame(l).dropna()
    out = common + col
    res.to_csv(out+".csv",index=False)



# dats = pd.read_csv(files[0])
# print(dats)
# dats.to_csv("out.csv")
# import code
# code.interact(local=locals())
import scipy.stats as stats
import pandas as pd
from pathlib import Path


auto = Path(r"output\analysis\2023.4.2 OptoTiam Exp 53")
d1 = "speeds_raw.csv"
d2 = "speeds_smoothed.csv"

datas = [pd.read_csv(auto/d) for d in [d1,d2]]


for col in ["No Light","Shallow","Steep"]:
    res = stats.ttest_ind(a=datas[0][col].dropna(),b=datas[1][col].dropna(),equal_var=True)
    print(col,res)

# datas = [pd.read_csv(r"output\analysis\2023.4.2 OptoTiam Exp 53 $manual\FMIs_Shallow.csv"),pd.read_csv(r"output\analysis\2023.4.2 OptoTiam Exp 53 $manual\FMIs_Steep.csv")]
# names = ["shallow","steep"]

# datas = [d.dropna() for d in datas]

# for d,name in zip(datas,names):
#     print(name)
#     print(d)
#     import code
#     code.interact(local=locals())
#     res = stats.ttest_ind(a=d['raw'], b=d['smoothed'], equal_var=True)
#     print(res)
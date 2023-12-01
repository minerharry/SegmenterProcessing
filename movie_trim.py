import os
from pathlib import Path
import shutil

from tqdm import tqdm
from utils.filegetter import askdirectory

name = "p_s{0}_t{1}.TIF"

class Et(BaseException): pass;
class Es(BaseException): pass;

src = askdirectory();
assert src;
src = Path(src);

dst = askdirectory();
assert dst;
dst = Path(dst);

##removes trim[0], trim[1] off the start and end of a movie
##DOES NOT COPY .nd FILES
trim = [6,0]

try:
    s = 0;
    st = tqdm()
    while True:
        st.update();
        s += 1;
        t = 0;
        try:
            with tqdm(leave=False) as tt:
                while True:
                    tt.update();
                    t += 1;
                    if t <= trim[0]: continue;
                    inf = src/name.format(s,t);
                    outf = dst/name.format(s,t-trim[0])
                    try:
                        shutil.copy(inf,outf);
                    except FileNotFoundError:
                        if t <= trim[0]+1:
                            raise Es();
                        else:
                            raise Et();
        except Et:
            for _ in range(trim[1]):
                t -= 1;
                file = dst/name.format(s,t);
                os.remove(file);
            pass;
except Es:
    pass;

print("copying complete; trimmed", s-1, "stage positions")
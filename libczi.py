from pylibCZIrw import czi
from IPython import start_ipython
im = r"C:\Users\Harrison Truscott\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\Mitch Morphodynamics Panels\5.31 dyed tl3 cel post MeCell spreading.czi"

with czi.open_czi(im) as c:
    start_ipython(user_ns=locals())

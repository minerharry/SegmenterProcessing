from pathlib import Path

import numpy as np
from tifffile import TiffFile
from tqdm import tqdm


folder = Path(r"C:\Users\miner\OneDrive - University of North Carolina at Chapel Hill\Bear Lab\Honors Thesis\Figures\Figure 2 Assets\trackmasks")

tiffs = list(folder.glob("*"));

combined = None
for t in tqdm(tiffs):
    file = TiffFile(t)
    idx = np.max(np.nonzero(file.asarray())[0])
    tqdm.write(str(idx))
    frame = file.pages[idx].asarray()
    if combined is None:
        combined = np.zeros(frame.shape,dtype=np.uint8)
    combined = combined | (frame > 0)

from imageio.v3 import imwrite

imwrite(folder.parent/"trackmasks_combined.TIF",combined*255);

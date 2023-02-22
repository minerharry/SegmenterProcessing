from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from skimage.io import imread,imsave
from skimage.exposure import rescale_intensity

root = tk.Tk()
root.withdraw()

file_paths = [Path(s) for s in filedialog.askopenfilenames()];

outdir = Path(filedialog.askdirectory());

for path in file_paths:
    imsave((outdir/path.name).with_suffix('.png'),rescale_intensity(imread(path)));


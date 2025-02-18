import os
import subprocess

from tqdm import tqdm
from utils.filegetter import afn
from utils.trackmasks import get_trackmasks
from tifffile import TiffFile
from skimage.io import imsave

tmfolder,trackmasks = get_trackmasks(afn(title="Tracks Masks Zip",key="trackmasks",filetypes=[("Trackmasks Zip file","*.zip")]))
dest = tmfolder/"unstacked"

selection = [(1, 7), (1, 39), (1, 40), (1, 106), (1, 78), (1, 110), (1, 52), (2, 52), (3, 39), (3, 74), (3, 50), (3, 52), (4, 9), (4, 77), (4, 83), (4, 88), (4, 41), (4, 42), (4, 55), (4, 58), (4, 59), (4, 62), (5, 0), (5, 2), (5, 131), (5, 67), (5, 6), (5, 14), (5, 271), (5, 273), (5, 21), (5, 149), (5, 214), (5, 153), (5, 155), (5, 44), (5, 306), (5, 51), (5, 180), (5, 244), (6, 195), (6, 197), (6, 71), (6, 75), (6, 86), (6, 91), (6, 98), (6, 100), (6, 37), (6, 167), (6, 171), (6, 45), (6, 185), (6, 122), (6, 190), (7, 0), (7, 3), (7, 9), (7, 19), (7, 30), (7, 31), (7, 117), (8, 3), (8, 117), (8, 231), (8, 25), (8, 234), (9, 3), (9, 67), (9, 262), (9, 9), (9, 16), (9, 273), (9, 149), (9, 278), (9, 23), (9, 90), (9, 91), (9, 33), (9, 353), (9, 300), (9, 47), (10, 5), (10, 197), (10, 70), (10, 11), (10, 141), (10, 77), (10, 79), (10, 209), (10, 19), (10, 214), (10, 23), (10, 91), (10, 29), (10, 31), (10, 98), (10, 99), (10, 102), (10, 40), (10, 105), (10, 113), (10, 117), (10, 254)]

for movie in tqdm(trackmasks):
    for tid in tqdm(trackmasks[movie],leave=False):
        if selection and (movie,tid) not in selection:
            continue
        tiff = TiffFile(trackmasks[movie][tid])
        startframe = tiff.shaped_metadata[0]["startframe"]
        folder = dest/f"movie{movie}_track{tid}"
        os.makedirs(folder,exist_ok=True)
        for idx,im in enumerate(tqdm(tiff.series[0],leave=False)):
            frame = startframe + idx
            file = folder/f"p_s{movie}_t{frame}.TIF"
            imsave(file,im.asarray(),check_contrast=False)

subprocess.Popen(f'explorer "{dest}"')
from datetime import timedelta
import numpy as np
from skimage.exposure import rescale_intensity
from pathlib import Path

from tqdm import tqdm
from image_annotations import draw_scalebar, draw_timestamp, applicable
import mediapy
# from libraries.movie_reading import Movie
from libraries.parse_moviefolder import get_movie

if __name__ == "__main__":
    loc = r"C:\Users\miner\GCP_TRANSFER\images\2023.4.2 OptoTiam Exp 53"
    mov = get_movie(loc)

    movie_process = 6

    td = timedelta(minutes=5)
    # pixel = 1.625
    # bar = 1
    # barunit = "mm"

    out = Path("temp/annotated")/Path(loc).name
    out.mkdir(parents=True,exist_ok=True)
    with mov.get_writer(out) as writer:
        with writer.sequence(movie_process) as seq:
            frames = tqdm(mov[movie_process])
            scaled = applicable(lambda shape: (lambda im,i: rescale_intensity((rescale_intensity(im)/255).astype(np.uint8),in_range=(10,65))))(frames) #I think I should kill myself
            timed = draw_timestamp(scaled,td,smallformat=None,textsize=2)
            barred = draw_scalebar(timed,1.625,scaleLength=500,scaleWidth=10)
            for b in barred:
                seq.write(b)

    mov = get_movie(out)
    mediapy.write_video("mov6.mp4",mov[movie_process],qp=2,fps=30)
                

from utils.filegetter import afns
from skimage.io import imread, imshow, imsave
from skimage.exposure import rescale_intensity
from skimage.measure import label
from skimage.morphology import remove_small_objects
import numpy as np
import matplotlib.pyplot as plt
from libraries.centers import generate_annotated_image,get_centers
from pathlib import Path
import os

# files = afns()
source = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\images\2023.4.2 OptoTiam Exp 53"
base = r"C:\Users\Harrison Truscott\Documents\GitHub\cell-tracking\gcp_transfer\cellmasks\2023.4.2 OptoTiam Exp 53\Cell"
files = ["p_s1_t7","p_s8_t247.TIF"]
out = Path("output/centers")

for name in files:
    file = base/name
    im = imread(file)
    im = label(im)
    im = remove_small_objects(im)
    im = im.astype("uint8")
    print(im.dtype)
    ids = np.unique(im)
    ids = ids[ids != 0]
    apm_image = generate_annotated_image(im,"approximate-medoid",ids,annotation_value=int(max(ids)+5))
    ell_image = generate_annotated_image(im,"ellipse",ids,annotation_value=int(max(ids)+5))
    try: os.makedirs(out/name) 
    except: pass

    plt.imshow(im)
    plt.savefig(out/name/"mask.png");
    plt.imshow(apm_image)
    plt.savefig(out/name/"approximate-medoid.png")
    plt.show()
    plt.imshow(ell_image)
    plt.savefig(out/name/"ellipse.png")
    plt.show()

    raw = imread(source/name)
    raw = rescale_intensity(raw);
    imsave(out/name/"raw.png",raw)

    # imsave(out/name/"raw.png",im)
    # imsave(out/name/"approximate-medoid.png",apm_image)
    # imsave(out/name/"ellipse.png",ell_image)


    plt.show()


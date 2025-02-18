from stitch_tiles import stitch_nd

gradientnum = 13
path = fr"c:\Users\bearlab\Documents\Data_temp\Harrison\Gradient Analysis\{gradientnum}\On slider\p.nd"

stitch_nd(path,'../tiling.tif',mag='4x')
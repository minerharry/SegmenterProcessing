### GOAL: given a .stg file of stage positions at a particular magnification, generate a "backtiling" in a different resolution -
### a new set of stage positions that should tile the original stage position. 
### one issue with this is that if the source tiles are near enough/overlapping (e.g. they themselves are tiled) the backtiles will 
### overlap and potentially be redundant. **THIS PROGRAM DOESN'T HAVE THE SMARTS TO HANDLE THAT**.
### This means if you are backtiling at multiple different magnifications, start each backtile from the same initial stage or else
### the extra tiling will be highly inefficient as the overlaps will compound!

import contextlib
import itertools
import os
from pathlib import Path
from typing import Iterable

from matplotlib import pyplot as plt
from metamorph_tiling import Tiler, Tiling, apply_zpoints, mag_adjust_bounds, get_tiling, monkey_tiler, plot_tiling, plot_z, read_tiling, write_tiling, xOffset

class hold_hostage(contextlib.ContextDecorator):

    def __init__(self,file):
        self.file = Path(file)
        self.hostage_loc = self.file.with_name("__hostage_" + self.file.name)
        if self.hostage_loc.exists():
            raise Exception(f"Hostage location {self.hostage_loc} exists")

    def __enter__(self):
        os.rename(self.file,self.hostage_loc)

    def __exit__(self,*args):
        os.rename(self.hostage_loc,self.file)


def make_backtiling(in_tiling:Tiling,
                    dest_mag:str,
                    orig_mag:str,
                    orig_z:int|Iterable[float],
                    get_tiling:Tiler = get_tiling)->tuple[Tiling,list[float]]:
    if not isinstance(orig_z,Iterable):
        orig_z = itertools.cycle([orig_z])
    res_tiles:Tiling = []
    res_z:list[float] = []
    for (name,(x,y)),z in zip(in_tiling,orig_z):
        print((name,(x,y)))
        x_bounds,y_bounds = mag_adjust_bounds(((x,x),(y,y)),dest_mag,orig_mag) #start with a zero-area point
        tiles = get_tiling(x_bounds[0],x_bounds[1],y_bounds[0],y_bounds[1],magnification=dest_mag)
        res_tiles += [(f"backtile{{{name}}}_{tile_name}",(tx,ty)) for tile_name,(tx,ty) in tiles]
        res_z += [z]*len(tiles)
    return res_tiles,res_z

if __name__ == "__main__":
    in_mag = "4x"
    in_file = fr"C:\Olympus\app\mmproc\DATA\auto_tiling_{in_mag}.STG"
    in_file = Path(in_file)

    in_tiling,in_z = read_tiling(in_file)




    ## pre-backtile z adjustments go here
    z_offset = (4247.59 - 4580)
    z_offset = 0
    in_z = [z + z_offset for z in in_z]
    

    ## do the backtiling
    out_mag = "20x"
    
    do_monkey = False
    tiler:Tiler = get_tiling
    if do_monkey:
        width = int(xOffset[out_mag]/(2.8))
        tiler = monkey_tiler('x',width,width);

    back_tiling,back_z = make_backtiling(in_tiling,out_mag,in_mag,in_z,get_tiling=tiler)

    ## post-backtile z adjustments go here
    back_z = 4487.58
    zpoints = [
        #pre-exp
        # ("backtile{p1}_s1_1",4489.68),
        # ("backtile{p1}_s4_1",4486.61),
        # ("backtile{p1}_s7_1",4483.54),
        # ("backtile{p2}_s1_2",4501.18),
        # ("backtile{p2}_s3_2",4498.74),
        # ("backtile{p2}_s7_2",4493.59),
        # ("backtile{p3}_s1_6",4506.09),
        # ("backtile{p3}_s4_6",4501.93),
        # ("backtile{p3}_s7_6",4497.21),
        # ("backtile{p4}_s1_7",4501.72),
        # ("backtile{p4}_s4_7",4496.17),
        # ("backtile{p4}_s7_7",4493.16),

        #post
        # ("backtile{p4}_s1_1",4525),
        # ("backtile{p1}_s1_3",4525),
        # ("backtile{p3}_s3_1",4499),
        # ("backtile{p2}_s2_1",4517.25),
        # ("backtile{p2}_s2_4",4503),
        # ("backtile{p5}_s2_2",4532),
        # ("backtile{p5}_s3_3",4521),
        # ("backtile{p6}_s1_3",4546),
        # ("backtile{p6}_s3_1",4541),
        # ("backtile{p7}_s3_1",4518),
        # ("backtile{p7}_s1_3",4534)
    ]

    if (len(zpoints) >= 3): ##this fully overwrites the original z values
        back_z = list(apply_zpoints(back_tiling,zpoints,method='quadratic'))

        z_shift = 0
        # z_shift = 25.07
        back_z = [z + z_shift for z in back_z]

    dest_file = rf"C:\Olympus\app\mmproc\DATA\auto_tiling_{out_mag}_backtiled_from_{{{in_file.stem}}}.STG"

    do_plot = True
    if do_plot:
        fig,ax = plt.subplots()

        plot_tiling(in_mag,in_tiling,in_z,ax,do_text=False,color="blue")

        plot_tiling(out_mag,back_tiling,back_z,ax,do_text=False,color="orange")

        ax.autoscale_view()        

        do_zplot = True
        if len(zpoints) >= 3 and do_zplot:
            plt.figure()
            ax2 = plt.subplot(projection="3d")
            plot_z(back_tiling,back_z,zpoints,ax2)


        if os.path.exists(dest_file): 
            lock = hold_hostage(dest_file)
        else:
            lock = contextlib.nullcontext();
        with lock:
            try:    
                plt.show()
            except KeyboardInterrupt as e:
                raise Exception("Operation canceled. Stage positions not saved.") from e




    write_tiling(dest_file,back_tiling,back_z);
    print(f"stage position saved to {dest_file}")



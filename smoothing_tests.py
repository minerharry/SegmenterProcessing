from calendar import c
import math

from matplotlib import cm

from tqdm import tqdm
from utils.filegetter import askopenfilehandle,askopenfilename

from PRW_model_functions import PRW
import matplotlib.pyplot as plt
import numpy as np
from utils.parse_tracks import FijiManualTrack, QCTracksDict
from libraries.smoothing import me, mavg, moving_average, exp_avg



def fmi(x,y):
    lengths = np.sqrt(np.sum(np.diff(np.stack([x,y],axis=1), axis=0)**2, axis=1)) # Length between corners
    total_length = np.sum(lengths)
    print(total_length)

    fx = (x[-1]-x[0])/total_length;
    fy = (y[-1]-y[0])/total_length;
    return (fx,fy)




if __name__ == "__main__":
    # px,py,_ = PRW(0.4,1,1,200);
    # px,py = np.array(px),np.array(py)

    fn = askopenfilename()


    manual = False
    if manual:
        tracks = FijiManualTrack(fn)
    else:
        movie = 7
        tracks = QCTracksDict(fn)[movie]
 
    for t in [7]:
        print(fn,t)
        track = tracks[t];

        p = np.array([k[1] for k in sorted(track.items())]).astype(float);
        px = p[:,0];
        py = p[:,1]

        px = px[::2]
        py = py[::2]
        noise = np.random.normal(0,0.7,[2,*px.shape]);
        print(fmi(px,py));
        # px += noise[0];
        # py += noise[1];
        plt.figure(1);
        plt.plot(px,py,color='black',alpha=1);
        plt.figure(2)
        plt.plot(px,py,color='black');
        

        start = px[0],py[0];
        end = px[-1],py[-1];
        print(f"start: {start}, end: {end}")

        x = np.array(px);
        y = np.array(py);

        sx = (moving_average**1)(x,5);
        sy = (moving_average**1)(y,5);
        # plt.plot(sx,sy,linestyle='--');

        prange = range(1,7);
        wrange = range(3,4);
        fxs = np.zeros([max(prange)+1,max(wrange)+1])
        fys = np.zeros([max(prange)+1,max(wrange)+1])

        colors = cm.jet(np.linspace(0,1,len(wrange)));
        alphas = np.linspace(0.9,0.2,len(prange))

        for i,power in enumerate(prange):
            for j,width in enumerate(wrange):
                tx = (moving_average**power)(x,width) #shorthand for recursively executing the average for power times
                ty = (moving_average**power)(y,width)
                plt.figure(2)
                plt.plot(tx,ty,linestyle='-',label=f"{power},{width}",color=colors[j],alpha=alphas[i])
                fxs[power,width],fys[power,width] = fmi(tx,ty);
                print(fmi(tx,ty))

        plt.figure(1)
        w,p = 3,3
        tx = (moving_average**p)(x,w);
        ty = (moving_average**p)(y,w);
        plt.plot(tx,ty,color=colors[wrange.index(w)],alpha=0.7)
    plt.figure(2)
    plt.legend()


    plt.figure(label="fmi x");
    plt.imshow(fxs)
    plt.ylabel("power")
    plt.xlabel("width")
    plt.figure(label="fmi y");
    plt.ylabel("power")
    plt.xlabel("width")
    plt.imshow(fys)

    # print(fxs,fys)


    plt.show();

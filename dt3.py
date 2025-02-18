import numpy as np
from tqdm import tqdm
from utils.filegetter import askopenfilename
import pandas as pd
import matplotlib.pyplot as plt
from smoothing_tests import moving_average

infile = askopenfilename();
if not infile.endswith(".csv"):
    raise Exception("needs flattened analysis, not shaped analysis!")
    
frame:pd.DataFrame = pd.read_csv(infile);

print("read")

invs = ('gradient.x','x steepness'),('gradient.y','y steepness'),('gradient intensity', 'intensity')
outvs = ('persistence',)*2,



for (inv,inshort) in tqdm(invs):
    for outv, outshort in tqdm(outvs,leave=False):
        plt.figure();
        frame = frame.sort_values(inv);
        plt.xlabel(inshort);
        plt.ylabel(outshort);
        valid = frame[(frame[inv].notna()) & (frame[outv].notna())];
        x,y = valid[inv],valid[outv]
        plt.scatter(x,y);

        plt.title(f"{inshort} vs {outshort}");
        outmean = [np.average(valid.loc[x == g][outv]) for g in tqdm(x,leave=False)]

        outmean = (moving_average**4)(outmean,18);
        # breakpoint()
        plt.plot(x,outmean,color='red');
        
        plt.figure()
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
        # heatmap = gaussian_filter(heatmap, sigma=10)
        
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent,origin='lower',aspect='auto');
        plt.scatter(x,y,marker='.');


# persistence, gx, gy, gi = frame['persistence'], frame['gradient.x'], frame['gradient.y'], frame['gradient intensity']
# plt.scatter(gi,persistence);
# pgi_mean = [np.average(frame.loc[frame['gradient intensity'] == g]['persistence']) for g in gi];

# plt.plot(gi,pgi_mean,color='red');
# plt.xlabel('intensity')
# plt.ylabel('persistence')
# plt.title("intensity vs persistence");
# plt.figure();
# plt.scatter(gx,persistence);
# pgx_mean = [np.average(frame.loc[frame['gradient.x'] == g]['persistence']) for g in gx];
# plt.plot(gx,pgx_mean,color='red');
# plt.xlabel('x steepness')
# plt.ylabel('persistence')
# plt.title("gradient x steepness vs persistence");
# plt.figure();
# plt.scatter(gy,persistence);
# pgy_mean = [np.average(frame.loc[frame['gradient.y'] == g]['persistence']) for g in gy];
# plt.plot(gy,pgy_mean,color='red');
# plt.xlabel('y steepness')
# plt.ylabel('persistence')
# plt.title("gradient y steepness vs persistence");


plt.show()
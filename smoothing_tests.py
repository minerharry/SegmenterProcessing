import math
from PRW_model_functions import run_PRW_sim
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    paths = run_PRW_sim(10,1,100,0.4);
    for p in paths[:1]:
        plt.plot(p['x'],p['y']);
    
    path = paths[0];
    px,py = path['x'],path['y'];
    start = px.iloc[0],py.iloc[0];
    end = px.iloc[-1],py.iloc[-1];
    print(f"start: {start}, end: {end}")

    s_smooth = np.array([0,0]);
    e_smooth = np.array([0,0]);
    i = 0;
    p = np.array([0,0]);
    for i,p in enumerate(zip(px,py)):
        if i != 0:
            s_smooth = np.add(s_smooth,p);
        e_smooth = np.add(e_smooth,p);
    e_smooth = np.subtract(e_smooth,p);
    s_smooth /= i-1
    e_smooth /= i-1;
    print(f"endless avg: {e_smooth}, startless avg: {s_smooth}")
    
    plt.plot(*start,marker='.')
    plt.plot(*end,marker='o');
    plt.plot(*e_smooth,marker='x');
    plt.plot(*s_smooth,marker='*');
    
    a1 = math.atan2(end[1]-start[1],end[0]-start[0]);
    a2 = math.atan2(-e_smooth[1]+s_smooth[1],-e_smooth[0]+s_smooth[0])
    print(a1,a2,(a1-a2)%(math.pi*2))

    plt.show();

    
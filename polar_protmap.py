from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence
from matplotlib import cm
from matplotlib.projections import PolarAxes
import numpy as np
from scipy.io import loadmat
from divergentcolor import MidpointNormalize
from utils.filegetter import afn

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

def roundto(x, base:float=5):
    return base * round(x/base)

def realign_polar_xticks(ax:PolarAxes):
    for theta, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        theta = theta * ax.get_theta_direction() + ax.get_theta_offset()
        theta = np.pi/2 - theta
        y, x = np.cos(theta), np.sin(theta)
        if x >= 0.1:
            label.set_horizontalalignment('left')
        if x <= -0.1:
            label.set_horizontalalignment('right')
        if y >= 0.5:
            label.set_verticalalignment('bottom')
        if y <= -0.5:
            label.set_verticalalignment('top')

CELL1_COLOR_RANGE = (-15.7903071218729,30.461424524784086)
CELL1_COLOR_RANGE = (-30,30)

def make_colorbar(cmap="seismic"):
    colornorm = MidpointNormalize(*CELL1_COLOR_RANGE)
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.colorbar(cm.ScalarMappable(norm=colornorm, cmap=cmap),ax=ax,location="left")

def plot_protrusion(analysis,draw_ROI = True,plot_type:Literal["scatter","surface"] = "surface",times:Sequence[datetime]|None=None):
    analysis = Path(analysis)

    mapname = "PVTAMF"
    smoothedname = "PVTAMF"
    data = loadmat(analysis)
    stats = data["CompStats"]

    static_reference = True; #whether to use protareaTAM2 (static reference) or protareaTAM (dynamic reference). Will plot ROI angular locations accordingly

    # from IPython import embed; embed()
    # print(data)
    protmap = stats[mapname][0,0] #cause it's an array inside an object for some reason
    sprotmap = stats[smoothedname][0,0]
    protareamap_centroid = stats["ProtAreaTAM2" if static_reference else "ProtAreaTAM"][0,0]
    # areamap_centroid = data["PADStats"][0,0]["AngleMap"][0]
    # areamap_doubleROI = data["DoubleROIStats"][0,0]["PAD_Stats"][0,0]["AngleMap"] #I want to kill matlab and then myself

    frames = stats["PSF"][0,0][0,0],stats["LSF"][0,0][0,0] #I hate matlab so much

    # times = stats["TotalTS"][0,0]
    # from IPython import embed; embed()



    protmap = protmap[:,:360] #this technically has 361*, I think so 0* is centered. We don't need that here so make it 360*
    sprotmap = sprotmap[:,:360]
    protareamap_centroid = protareamap_centroid[:,:360]
    # areamap_centroid = areamap_centroid[:,:360]
    # areamap_doubleROI = areamap_doubleROI[:,:360]
    print(protareamap_centroid.shape)

    fig = plt.figure(figsize=(16,12))
    plt.suptitle(analysis.name)
    ncols,nrows = 2,1

    if draw_ROI:
        bright_color = "#6e00f4";
        dim_color = "#ff72ed";

    

    ### ADJUST PROTRUSION MAP ANGLES SO THEY POINT IN THE SAME DIRECTION AS THE DELTA IMAGE
    firstframe = frames[0]-1
    lastframe = frames[1]-1
    protmap = np.pad(protmap,((firstframe-0,0),(0,0)),constant_values=0,)
    sprotmap = np.pad(sprotmap,((firstframe-0,0),(0,0)),constant_values=0)
    # protareamap_centroid = np.pad(protareamap_centroid,((firstframe-0,0),(0,0)),constant_values=0)
    # from IPython import embed; embed()
    dstats = stats["DoubleROIStats"][0,0]
    leading = dstats["LeadingRoi"][0,0]-1 #-1 because matlab indices
    leading_ROI_angle = dstats["ROIAngle"][0,0][firstframe,leading][0,0]
    # from IPython import embed; embed()
    ROI_angle_offset = leading_ROI_angle*np.pi/180
    

    up = ROI_angle_offset #np.pi/2
    up = roundto(up,np.pi/2)
    ROI_angle_offset = 0
    angles = np.linspace(- np.pi, + np.pi,360,endpoint=False)
    maxR = protmap.shape[0]
    r = np.linspace(0,maxR,protmap.shape[0])
    rdelta = np.diff(r)[0] #size of each radial bin
    assert np.allclose(rdelta,np.diff(r))
    assert np.allclose(angles[180],0)

    print(np.min(protareamap_centroid),np.max(protareamap_centroid))
    # exit()

    maps = ((protmap,"Prot Map"),(sprotmap,"Smoothed Prot Map"),(protareamap_centroid,"Delta Area Map"))
    for i,(prot,name) in enumerate(maps[2:]):#,(areamap_doubleROI,"Delta Area Map - DoubleROI")):
        print(prot.shape,name)
        ax:PolarAxes = fig.add_subplot(nrows,ncols,i+1,projection="polar")
        ax.set_title(name,fontsize=24)

        ax.set_theta_offset(up)
        ax.set_rlabel_position(-up*180/np.pi);



        colornorm = MidpointNormalize(*CELL1_COLOR_RANGE)

        if plot_type == "scatter":
            for rad,row in zip(r,prot):
                from IPython import embed
                assert len(row) == len(angles),(len(row),len(angles),(row,angles),embed())
                ax.scatter(angles,[rad+rdelta/2]*len(angles),c=row,s=5,cmap="RdBu_r")
        else:
            #to do: binning
            # t,r = np.meshgrid(angles,r)
            # ax.contourf(t,r,prot,cmap='RdBu_r')
            print("woaw")

            # loopr = np.concatenate([r,[r[-1]+rdelta]]) #these radii define the boundaries between the pixels
            # loopa = np.concatenate([angles,[angles[0]]]) #glue edges together
            # ax.pcolormesh(loopa,loopr,prot,cmap='RdBu_r')

            mesh = ax.pcolormesh(angles,r+rdelta/2,prot,cmap='seismic',shading="nearest",norm=colornorm)
            mesh.set_mouseover(True)
            # from IPython import embed; embed()
        
        # with plt.ion():
        #     plt.show()
        #     from IPython import embed; embed()

        if draw_ROI:

            ranges = dstats["ROI_timed_angle_ranges"][0,0] #ROI, frame, [min,max]
            # from IPython import embed; embed()
            min_angles = ranges[:,0,0];
            max_angles = ranges[:,0,1]
            
            start:np.floating = np.max([r[firstframe],0])
            end = maxR
            if static_reference:
                # ranges = np.stack([dstats["ROI_min_angle"][:,0],dstats["ROI_max_angle"][:,0]],axis=1)[0] #list of [[min,max],...]
                # from IPython import embed; embed()
                for i,(min,max) in enumerate(zip(min_angles,max_angles)):
                    min = min*np.pi/180 + ROI_angle_offset
                    max = max*np.pi/180 + ROI_angle_offset
                    color = bright_color if i==leading else dim_color

                    xpoints = []; ypoints = []

                    # from IPython import embed; embed()
                    # ax.plot([min,min],[start,end],linestyle='-',color=color);
                    # ax.plot([max,max],[start,end],linestyle='-',color=color);
                    
                    if min < max:
                        arcangles = np.linspace(min,max,100)
                    else:
                        arcangles = np.concatenate([np.linspace(min,np.pi + ROI_angle_offset,50),np.linspace(-np.pi+ROI_angle_offset,max,50)])
                    # ax.plot(arcangles,[end,]*100,linestyle='--',color=color);
                    # ax.plot(arcangles,[start,]*100,linestyle='--',color=color);
                    cat = lambda *x: np.concatenate([*x])

                    xpoints = cat(xpoints,[min,min]);       ypoints = cat(ypoints,[start,end])
                    xpoints = cat(xpoints, arcangles);      ypoints = cat(ypoints,[end,]*100)
                    xpoints = cat(xpoints,[max,max]);       ypoints = cat(ypoints,[end,start])
                    xpoints = cat(xpoints,arcangles[::-1]); ypoints = cat(ypoints,[start,]*100)
                    poly = Polygon(np.array(list(zip(xpoints,ypoints))),fill=False,linestyle='--',edgecolor=color,closed=True)
                    ax.add_patch(poly)
                    
            else:
                
                for i,rangelist in enumerate(ranges):
                    rangelist = rangelist*np.pi/180 + ROI_angle_offset
                    color = bright_color if i==leading else dim_color
                    leftT = rangelist[firstframe:,0]
                    rightT = rangelist[firstframe:,0]
                    # leftT = np.pad(leftT,1,'edge');
                    # rightT = np.pad(rightT,1,'edge');
                    radii = [start] + r[firstframe:] + [end];

                    # ax.plot(leftT,radii,linestyle='--',color=color);
                    # ax.plot(rightT,radii,linestyle='--',color=color);
                    
                    upmin,upmax = rangelist[-1,0],rangelist[-1,1]
                    if upmin < upmax:
                        uparcangles = np.linspace(upmin,upmax,100)
                    else:
                        uparcangles = np.concatenate([np.linspace(upmin,np.pi + ROI_angle_offset,50),np.linspace(-np.pi+ROI_angle_offset,upmax,50)])
                    
                    downmin,downmax = rangelist[firstframe,0],rangelist[firstframe,1]
                    if downmin < downmax:
                        downarcangles = np.linspace(downmin,downmax,100)
                    else:
                        downarcangles = np.concatenate([np.linspace(downmin,np.pi + ROI_angle_offset,50),np.linspace(-np.pi+ROI_angle_offset,downmax,50)])
                    # ax.plot(uparcangles,[end,]*100,linestyle='--',color=color);
                    # ax.plot(downarcangles,[start,]*100,linestyle='--',color=color);
                    cat = lambda *x: np.concatenate([*x])
    
                    xpoints = []; ypoints = []
                    xpoints = cat(xpoints,leftT);               ypoints = cat(ypoints,radii)
                    xpoints = cat(xpoints,uparcangles);         ypoints = cat(ypoints,[end,]*100)
                    xpoints = cat(xpoints,rightT)[::-1];        ypoints = cat(ypoints,radii[::-1])
                    xpoints = cat(xpoints,downarcangles)[::-1]; ypoints = cat(ypoints,[start,]*100)

                    poly = Polygon(np.array(zip(xpoints,ypoints)),fill=False,linestyle='--',edgecolor=color,closed=True)
                    ax.add_patch(poly)

        realign_polar_xticks(ax)
        # for rtick in ax.get_yticklabels():
        #     rtick.set_text(rtick.get_text() + ' seventeen') #add "minute" tick
        #     rtick.set_fontsize('large')
        ticks = [13,25,37] #frame#s of 10, 20, 30 minutes
        if times:
            ticklabels = [f"{(times[tick-1] - times[0]).seconds // 60}' " for tick in ticks]
        else:
            ticklabels = list(map(str,ticks))
        ax.set_rgrids(ticks,ticklabels,ha="right",fontsize=18)
        for ttick in ax.get_xticklabels():
            ttick.set_fontsize(20)


    # from IPython import embed; embed()

    mask = data["Final_Mask"]
    delta = mask[:,:,frames[1]-1].astype(int) - mask[:,:,frames[0]-1]

    ax = fig.add_subplot(nrows,ncols,ncols*nrows)
    ax.set_title("pad_image")
    ax.imshow(delta,cmap="RdBu_r")

    if draw_ROI:

        dstats = stats["DoubleROIStats"][0,0]

        leading = dstats["LeadingRoi"][0,0]-1 #-1 because matlab indices

        rois = dstats["Rois"][0,0]["roiData"][:,0]

        for i,R in enumerate(rois):
            color = bright_color if i==leading else dim_color
            L,T,W,H = [x[0,0] for x in R[0,0]] #gonna murder matlab. Also TL are xy flipped because matplotlib
            rect = Rectangle([T,L],W,H,linestyle="--",edgecolor=color,fill=None)
            ax.add_artist(rect)

        # firstframe = frames[0]-1
        # start:np.floating = r[firstframe]
        # if static_reference:
        #     ranges = np.stack([dstats["ROI_min_angle"][:,0],dstats["ROI_max_angle"][:,0]],axis=1)[0] #list of [[min,max],...]
        #     for i,((min,),(max,)) in enumerate(ranges):
        #         color = bright_color if i==leading else dim_color
        #         from IPython import embed; embed()
        #         ax.plot([min,min],[start,maxR],linestyle='--',color=color);
        #         ax.plot([max,max],[start,maxR],linestyle='--',color=color);
        #         ax.plot([min,max],[maxR,maxR],linestyle='--',color=color);
        #         ax.plot([min,max],[start,start],linestyle='--',color=color);
        # else:
        #     ranges = dstats["ROI_timed_angle_ranges"] #ROI, frame, [min,max]
        #     for i,rangelist in enumerate(ranges):
        #         color = bright_color if i==leading else dim_color
        #         ax.plot(rangelist[firstframe:,0],r[firstframe:],linestyle='--',color=color);
        #         ax.plot(rangelist[firstframe:,1],r[firstframe:],linestyle='--',color=color);
        #         ax.plot([rangelist[-1,0],rangelist[-1,1]],[maxR,maxR],linestyle='--',color=color);
        #         ax.plot([rangelist[-1,0],rangelist[-1,1]],[start,start],linestyle='--',color=color);


    # fig.subti

    return fig,ax

if __name__ == "__main__":
    # analy = afn()
    # plot_protrusion(analy,draw_ROI=True)#,plot_type="scatter")

    # from utils.filegetter import adir
    # def iter_all(d):
    #     p = Path(d)
    #     for k in p.glob("*.mat"):
    #         print(k.name)
    #         plot_protrusion(k)
    #         plt.show()

    # from IPython import embed; embed()

    cell1_times = [
        "09-Oct-2024 07:59:08.625 PM",
        "09-Oct-2024 08:00:11.989 PM",
        "09-Oct-2024 08:01:02.710 PM",
        "09-Oct-2024 08:01:53.463 PM",
        "09-Oct-2024 08:02:44.153 PM",
        "09-Oct-2024 08:03:34.858 PM",
        "09-Oct-2024 08:04:25.739 PM",
        "09-Oct-2024 08:05:16.428 PM",
        "09-Oct-2024 08:06:07.197 PM",
        "09-Oct-2024 08:06:58.205 PM",
        "09-Oct-2024 08:07:48.910 PM",
        "09-Oct-2024 08:08:39.616 PM",
        "09-Oct-2024 08:09:30.560 PM",
        "09-Oct-2024 08:10:21.329 PM",
        "09-Oct-2024 08:11:12.051 PM",
        "09-Oct-2024 08:12:02.756 PM",
        "09-Oct-2024 08:12:53.636 PM",
        "09-Oct-2024 08:13:44.294 PM",
        "09-Oct-2024 08:14:35.047 PM",
        "09-Oct-2024 08:15:26.007 PM",
        "09-Oct-2024 08:16:16.697 PM",
        "09-Oct-2024 08:17:07.689 PM",
        "09-Oct-2024 08:17:58.616 PM",
        "09-Oct-2024 08:18:49.338 PM",
        "09-Oct-2024 08:19:40.250 PM",
        "09-Oct-2024 08:20:30.972 PM",
        "09-Oct-2024 08:21:21.725 PM",
        "09-Oct-2024 08:22:12.510 PM",
        "09-Oct-2024 08:23:03.422 PM",
        "09-Oct-2024 08:23:54.176 PM",
        "09-Oct-2024 08:24:44.864 PM",
        "09-Oct-2024 08:25:35.729 PM",
        "09-Oct-2024 08:26:26.514 PM",
        "09-Oct-2024 08:27:17.283 PM",
        "09-Oct-2024 08:28:08.004 PM",
        "09-Oct-2024 08:28:58.694 PM",
        "09-Oct-2024 08:29:49.543 PM",
    ]
    cell2_times = [
        "09-Oct-2024 08:40:52.615 PM",
        "09-Oct-2024 08:41:55.868 PM",
        "09-Oct-2024 08:42:46.589 PM",
        "09-Oct-2024 08:43:37.374 PM",
        "09-Oct-2024 08:44:28.079 PM",
        "09-Oct-2024 08:45:18.832 PM",
        "09-Oct-2024 08:46:09.522 PM",
        "09-Oct-2024 08:47:00.197 PM",
        "09-Oct-2024 08:47:50.854 PM",
        "09-Oct-2024 08:48:41.608 PM",
        "09-Oct-2024 08:49:32.503 PM",
        "09-Oct-2024 08:50:23.209 PM",
        "09-Oct-2024 08:51:13.961 PM",
        "09-Oct-2024 08:52:04.698 PM",
        "09-Oct-2024 08:52:55.453 PM",
        "09-Oct-2024 08:53:46.141 PM",
        "09-Oct-2024 08:54:36.910 PM",
        "09-Oct-2024 08:55:27.680 PM",
        "09-Oct-2024 08:56:18.401 PM",
        "09-Oct-2024 08:57:09.091 PM",
        "09-Oct-2024 08:57:59.796 PM",
        "09-Oct-2024 08:58:50.533 PM",
        "09-Oct-2024 08:59:41.270 PM",
        "09-Oct-2024 09:00:32.055 PM",
        "09-Oct-2024 09:01:22.762 PM",
        "09-Oct-2024 09:02:13.499 PM",
        "09-Oct-2024 09:03:04.236 PM",
        "09-Oct-2024 09:03:55.020 PM",
        "09-Oct-2024 09:04:45.805 PM",
        "09-Oct-2024 09:05:36.528 PM",
        "09-Oct-2024 09:06:27.186 PM",
        "09-Oct-2024 09:07:17.842 PM",
        "09-Oct-2024 09:08:08.802 PM",
        "09-Oct-2024 09:08:59.555 PM",
        "09-Oct-2024 09:09:50.293 PM",
        "09-Oct-2024 09:10:41.015 PM",
        "09-Oct-2024 09:11:31.704 PM",
    ]

    cell1_times = list(map(lambda s: datetime.strptime(s,"%d-%b-%Y %I:%M:%S.%f %p"),cell1_times))
    cell2_times = list(map(lambda s: datetime.strptime(s,"%d-%b-%Y %I:%M:%S.%f %p"),cell2_times))

    plot_protrusion(r"E:/Lab Data/doubleROI/Analysis/first_reanalyzed/10_9_cell1_grad1.mat",draw_ROI=True,times=cell1_times)
    plot_protrusion(r"E:/Lab Data/doubleROI/Analysis/second_reanalyzed/10_9_cell1_grad2.mat",draw_ROI=True,times=cell2_times)
    make_colorbar()

    plt.show()

    # iter_all(adir())




# # from IPython import embed; embed()
    
# plt.show()
    
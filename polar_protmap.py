from pathlib import Path
from typing import Literal
from matplotlib.projections import PolarAxes
import numpy as np
from scipy.io import loadmat
from divergentcolor import MidpointNormalize
from utils.filegetter import afn

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def realign_polar_xticks(ax):
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

def plot_protrusion(analysis,draw_ROI = True,plot_type:Literal["scatter","surface"] = "surface"):
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



    protmap = protmap[:,:360] #this technically has 361*, I think so 0* is centered. We don't need that here so make it 360*
    sprotmap = sprotmap[:,:360]
    protareamap_centroid = protareamap_centroid[:,:360]
    # areamap_centroid = areamap_centroid[:,:360]
    # areamap_doubleROI = areamap_doubleROI[:,:360]
    print(protareamap_centroid.shape)

    fig = plt.figure(figsize=(16,12))
    plt.suptitle(analysis.name)
    ncols,nrows = 2,2

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
    ROI_angle_offset = 0
    angles = np.linspace(- np.pi, + np.pi,360,endpoint=False)
    maxR = protmap.shape[0]
    r = np.linspace(0,maxR,protmap.shape[0])
    rdelta = np.diff(r)[0] #size of each radial bin
    assert np.allclose(rdelta,np.diff(r))
    assert np.allclose(angles[180],0)



    for i,(prot,name) in enumerate(((protmap,"Prot Map"),(sprotmap,"Smoothed Prot Map"),(protareamap_centroid,"Delta Area Map"))):#,(areamap_doubleROI,"Delta Area Map - DoubleROI")):
        print(prot.shape,name)
        ax:PolarAxes = fig.add_subplot(nrows,ncols,i+1,projection="polar")
        ax.set_title(name)

        ax.set_theta_offset(up)
        ax.set_rlabel_position(-up*180/np.pi);

        colornorm = MidpointNormalize()

        # ax.set_yticklabels([])
        # ax.set_xticklabels([])
        # ax.set_ylim((0,maxR))

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
                    # from IPython import embed; embed()
                    ax.plot([min,min],[start,end],linestyle='--',color=color);
                    ax.plot([max,max],[start,end],linestyle='--',color=color);
                    
                    if min < max:
                        arcangles = np.linspace(min,max,100)
                    else:
                        arcangles = np.concatenate([np.linspace(min,np.pi + ROI_angle_offset,50),np.linspace(-np.pi+ROI_angle_offset,max,50)])
                    ax.plot(arcangles,[end,]*100,linestyle='--',color=color);
                    ax.plot(arcangles,[start,]*100,linestyle='--',color=color);
            else:
                
                for i,rangelist in enumerate(ranges):
                    rangelist = rangelist*np.pi/180 + ROI_angle_offset
                    color = bright_color if i==leading else dim_color
                    leftT = rangelist[firstframe:,0]
                    rightT = rangelist[firstframe:,0]
                    # leftT = np.pad(leftT,1,'edge');
                    # rightT = np.pad(rightT,1,'edge');
                    radii = [start] + r[firstframe:] + [end];

                    ax.plot(leftT,radii,linestyle='--',color=color);
                    ax.plot(rightT,radii,linestyle='--',color=color);
                    
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
                    ax.plot(uparcangles,[end,]*100,linestyle='--',color=color);
                    ax.plot(downarcangles,[start,]*100,linestyle='--',color=color);

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
            rect = Rectangle([T,L],W,H,linestyle="--",color=color)
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
    analy = afn()
    plot_protrusion(analy,draw_ROI=True)#,plot_type="scatter")

    from utils.filegetter import adir
    def iter_all(d):
        p = Path(d)
        for k in p.glob("*.mat"):
            print(k.name)
            plot_protrusion(k)
            plt.show()

    from IPython import embed; embed()

    # iter_all(adir())




# # from IPython import embed; embed()
    
# plt.show()
    
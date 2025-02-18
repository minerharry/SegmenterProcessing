from functools import cache
import itertools
from pathlib import Path
import re
from typing import Any, DefaultDict, Final, Iterable, Literal, final
from indexed import IndexedOrderedDict as OrderedDict
from matplotlib.widgets import CheckButtons
from skimage.measure import label,regionprops_table
from matplotlib.collections import LineCollection,PolyCollection
from matplotlib.image import AxesImage
from matplotlib.text import Text
from libraries.filter_cells_fns import CellFilters
from playslider import Player
import matplotlib.pyplot as plt
import joblib
import numpy as np
from ordered_set import OrderedSet
from tqdm import tqdm as tqdm
from libraries.movie_reading import FramedMovieSequence
from libraries.parse_moviefolder import get_movie
from libraries.filenames import filename_regex_tiff_png
from utils.associations import get_full_associations
from utils.extractmasks import extract_labeled_cellmasks
from utils.filegetter import afn,adir,skip_cached_popups
from utils.parse_tracks import QCTracks, QCTracksDict
from utils.trackmasks import get_trackmasks,read_trackmask
from utils.outlines import get_labeled_mask_outlines, get_mask_outlines
from utils.inquire import inquire
import contextlib as ctx

use_cached_locations = False

with skip_cached_popups() if use_cached_locations else ctx.nullcontext():
    mfolder = adir(key="images",title="Images Folder")
    mask_type:Final[Literal["trackmasks","cellmasks","labeledmasks","None"]] = "None" #inquire("Mask type",["cellmasks","trackmasks","labeledmasks"])
    match mask_type:
        case "trackmasks":
            mask_path = afn(key="trackmasks",title="Track Masks (auto)",filetypes=[("Trackmask .zip files","*.zip")])
        case "cellmasks":
            mask_path = adir(key="cellmasks",title="Segmented Cell Masks Folder")
        case "labeledmasks":
            mask_path = afn(key="labeledmasks",title="Labeled Masks (auto)",filetypes=[("labeledmask .zip files","*.zip")])
        case "None":            
            mask_path = None
            # raise NotImplementedError("labeled masks unzipping and usage not yet implemented in scripting. If you really want it, you can unzip the file manually and pass it to cellmasks")

    nucmask_type:Final[Literal["nucmasks","labeledmasks","None"]] = "None"
    match nucmask_type:
        case "nucmasks":
            nucmask_path = adir(key="nucmasks",title="Segmented Nucleus Masks Folder")
        case "labeledmasks":
            nucmask_path = afn(key="labelednucs",title="Labeled Nucleus Masks (auto)",filetypes=[("labelednuc .zip files","*.zip")])
        case "None":
            nucmask_path = None
            # raise NotImplementedError("labeled masks unzipping and usage not yet implemented in scripting. If you really want it, you can unzip the file manually and pass it to cellmasks")
    dnames:tuple[str,str] = [afn(key="track1_unscaled",title="Automatic QC Tracks (Unscaled)"),afn(key="track2_unscaled",title="Manual QC Tracks (Unscaled)")]

qs = [QCTracks(d) for d in dnames]
d1 = QCTracksDict(qs[0])
d2 = QCTracksDict(qs[1])
images = get_movie(mfolder)

print(str(Path(mfolder).parent))
mat = re.match(r".*?(\d+)$",str(Path(mfolder).parent))
if mat is not None:
    experiment = int(mat.group(1))
    smoothing = "smoothed" if "smoothed" in dnames[0] else "raw" if "raw" in dnames[0] else "noqc"
    assoc_path = f"associations/assoc_results_{experiment}_{smoothing}.txt"
    associations,inclusions,remainders = get_full_associations(qs[0],qs[1],names=("Automatic","Manual"),savepath=assoc_path,originpaths=tuple(dnames))
else:
    print("unable to ascertain experiment")
    associations,inclusions,remainders = get_full_associations(qs[0],qs[1],names=("Automatic","Manual"))



print(list(images.keys()))
movie_num = 1
movie:FramedMovieSequence = images[movie_num]

# graph = dcc.Graph(id="graph")
# app = DashProxy(transforms=[BlockingCallbackTransform(timeout=2)])
# slider = Toolbar(graph,slider=True,slider_range=range(1,len(movie)+1))

framed_points:dict[int,tuple[tuple[list[float],list[float],list[str]],tuple[list[float],list[float],list[str]]]] = {f:(([],[],[]),([],[],[])) for f in movie.frames} #x,y,label


memory = joblib.Memory(location="caches/dots")

for (tid,track) in tqdm(d1[movie_num].items(),desc="framing track1 points"):
    for frame,pos in track.items():
        framed_points[frame][0][0].append(pos[0])
        framed_points[frame][0][1].append(pos[1])
        framed_points[frame][0][2].append(str(tid))

for (tid,track) in tqdm(d2[movie_num].items(),desc="framing track2 points"):
    for frame,pos in track.items():
        framed_points[frame][1][0].append(pos[0])
        framed_points[frame][1][1].append(pos[1])
        framed_points[frame][1][2].append(str(tid))

framed_associations:dict[int,list[tuple[tuple[float,float],tuple[float,float]]]] = {f:[] for f in movie.frames} #list of pairs of associated points in each frame

for tid1,tid2 in tqdm(associations[movie_num],desc="storing associations..."):
    t1,t2 = d1[movie_num][tid1],d2[movie_num][tid2]
    frames = set(t1.keys()).intersection(t2.keys())
    for frame in frames:
        framed_associations[frame].append((t1[frame],t2[frame]))

framed_visibilities:dict[int,tuple[dict[int,bool],dict[int,bool]]] = {f:(DefaultDict(bool),DefaultDict(bool)) for f in movie.frames}
for tid,track in tqdm(d1[movie_num].items(),desc="visibility",leave=False):
    for frame in track:
        framed_visibilities[frame][0][tid] = True
for tid,track in tqdm(d2[movie_num].items(),desc="visibility",leave=False):
    for frame in track:
        framed_visibilities[frame][1][tid] = True


full_breadcrumbs:tuple[list[tuple[list[int],np.ndarray]],list[tuple[list[int],np.ndarray]]] = ([],[]) #startframe,points
for i,d in [(0,d1),(1,d2)]:
    for tid,track in tqdm(d[movie_num].items(),desc="breadcrumbs",leave=False):
        keys = sorted(track.keys())
        blank = np.ndarray((len(keys),2))
        for frame,pos in track.items():
            blank[keys.index(frame)] = pos
        full_breadcrumbs[i].append((keys,blank))

framed_breadcrumbs:dict[int,tuple[list[np.ndarray],list[np.ndarray]]] = {f:([],[]) for f in movie.frames}
for frame in framed_breadcrumbs:
    for i in [0,1]:
        for frames,crumbs in full_breadcrumbs[i]:
            if frame in frames:
                framed_breadcrumbs[frame][i].append(crumbs[0:frames.index(frame)+1]) 
            elif frame < min(frames):
                framed_breadcrumbs[frame][i].append(np.ndarray((0,2)))
            else:
                framed_breadcrumbs[frame][i].append(crumbs)
                

# print(framed_associations)
@memory.cache
def get_trackmask_outlines(movie_num:int,trackmask_path:str):
    _,trackmasks = get_trackmasks(trackmask_path)
    framed_outlines:dict[int,dict[int,np.ndarray]] = {f:{} for f in movie.frames}
    for tid,trackmask in tqdm(trackmasks[movie_num].items(),desc="outlining..."):
        frames,series = read_trackmask(trackmask)
        for frame,im in zip(frames,series):
            # print(movie_num,tid,frame)
            os = get_mask_outlines(im.asarray())
            if len(os) > 0:
                outline = os[0] #should only be one!!
                framed_outlines[frame][tid] = outline
    return framed_outlines

@memory.cache
def get_cellmask_outlines(movie_num:int,mask_folder:str):
    masks = get_movie(mask_folder,custom_regex=filename_regex_tiff_png)[movie_num]
    framed_outlines:dict[int,dict[int,np.ndarray]] = {f:{} for f in movie.frames}
    for frame in tqdm(movie.frames,"outlining..."):
        outlines = get_mask_outlines(masks[frame])
        framed_outlines[frame] = dict(enumerate(outlines))
    return framed_outlines

@memory.cache
def get_coded_labeledmask_outlines(movie_num:int,mask_zipfile:str):
    # just getting the code instead of the full label should greatly reduce the number 
    # of cv2 contours calls as all labels with the same code can be calculated in parallel.
    # otherwise you'd need a new mask for every label
    mask_folder = extract_labeled_cellmasks(mask_zipfile)
    masks = get_movie(mask_folder,custom_regex=filename_regex_tiff_png)[movie_num]
    framed_outlines:dict[int,dict[int,np.ndarray]] = {f:{} for f in movie.frames}
    outline_codes:dict[int,dict[int,int]] = {f:{} for f in movie.frames}
    for frame in tqdm(movie.frames,"outlining..."):
        m:np.ndarray = CellFilters.get_code(masks[frame])
        # print(m.dtype)
        # print(m.min())
        # print(np.unique(m))
        coded_outlines = get_labeled_mask_outlines(m,ignore=-1)
        for i,(code,outline) in enumerate(itertools.chain(*[zip(itertools.repeat(code),outlines) for code,outlines in coded_outlines])):
            # print(code)
            # if code == 2:
            #     from IPython import embed; embed()
            framed_outlines[frame][i] = outline
            outline_codes[frame][i] = code
    return framed_outlines,outline_codes

@memory.cache
def get_labeled_labeledmask_outlines(movie_num:int,mask_zipfile:str):
    # less efficient than just getting the code, as it has to calculate a new mask for each label individually
    mask_folder = extract_labeled_cellmasks(mask_zipfile)
    masks = get_movie(mask_folder,custom_regex=filename_regex_tiff_png)[movie_num]
    framed_outlines:dict[int,dict[int,np.ndarray]] = {f:{} for f in movie.frames}
    outline_labels:dict[int,dict[int,int]] = {f:{} for f in movie.frames}
    for frame in tqdm(movie.frames,"outlining..."):
        labeled_outlines = get_labeled_mask_outlines(masks[frame])
        for i,(label,outline) in enumerate(itertools.chain(*[zip(itertools.chain([code]),outlines) for code,outlines in labeled_outlines])):
            framed_outlines[frame][i] = outline
            outline_labels[frame][i] = label
    return framed_outlines,outline_labels


@memory.cache
def get_cellmask_areas(movie_num:int,mask_folder:str):
    masks = get_movie(mask_folder,custom_regex=filename_regex_tiff_png)[movie_num]
    areas:dict[int,list[tuple[tuple[float,float],float]]] = {f:[] for f in movie.frames} #center,area
    for frame in tqdm(movie.frames,"areaing..."):
        props = regionprops_table(label(masks[frame]),properties=["area_filled","centroid"])
        # from IPython import embed; embed()
        for (centroidx,centroidy,area) in zip(props["centroid-1"],props["centroid-0"],props["area_filled"]):
            areas[frame].append(((centroidx,centroidy),area))
    return areas

if mask_type == "trackmasks":
    framed_outlines = get_trackmask_outlines(movie_num,str(mask_path))
elif mask_type == "cellmasks":
    framed_outlines = get_cellmask_outlines(movie_num,str(mask_path))
elif mask_type == "labeledmasks":
    framed_outlines,framed_outline_codes = get_coded_labeledmask_outlines(movie_num,str(mask_path))
elif mask_type == "None":
    framed_outlines = None
    framed_outline_codes = None
else:
    raise Exception()

if nucmask_type == "nucmasks":
    framed_nuc_outlines = get_cellmask_outlines(movie_num,str(nucmask_path))
elif nucmask_type == "labeledmasks":
    raise NotImplementedError()
elif nucmask_type == "None":
    framed_nuc_outlines = None
else:
    raise Exception()

if mask_type == "cellmasks":
    framed_areas = get_cellmask_areas(movie_num,str(mask_path))
elif mask_type == "labeledmasks":
    raise NotImplementedError()
elif mask_type == "None":
    framed_areas = None
else:
    raise Exception()

fig,ax = plt.subplots(num=f"Movie {movie_num}")#,layout="constrained")
title = plt.title(f"Movie {movie_num}, Frame 1")


masks = PolyCollection([],color="grey",alpha=0.4)
nucmasks = PolyCollection([],color="red",alpha=0.3)
ax.add_collection(masks)
ax.add_collection(nucmasks)


blankdot = (float('nan'),float('nan'))
blankline = [blankdot]

maxauto = max(len(f[0][0]) for f in framed_points.values())
autodots = ax.scatter([None]*maxauto,[None]*maxauto,color="red",alpha=0.5,label="auto")
maxman = max(len(f[1][0]) for f in framed_points.values())
mandots = ax.scatter([None]*maxman,[None]*maxman,color="blue",alpha=0.5,label="manual")

maxconn = max(len(k) for k in framed_associations.values())
connections = LineCollection([blankline]*maxconn,color="purple")

autolabels = {k:ax.text(0,0,str(k),visible=False,color="darkred") for k in d1[movie_num].keys()}
manlabels = {k:ax.text(0,0,str(k),visible=False,color="darkblue",ha="right") for k in d2[movie_num].keys()}

if framed_areas:
    maxareas = max(len(k) for k in framed_areas.values())
    areas = [ax.text(0,0,"0",visible=False,color="black") for i in range(maxareas)]

autocrumbs = LineCollection([blankline]*len(d1[movie_num]),color="red",alpha=0.4)
mancrumbs = LineCollection([blankline]*len(d2[movie_num]),color="blue",alpha=0.4)

code_colors = {CellFilters.valid:"grey",CellFilters.two_nuclei:"blue",CellFilters.too_large:"yellow",CellFilters.too_small:"green",CellFilters.touching_edge:"purple"}

code_t = []
for code,color in code_colors.items():
    a = ax.plot([],color=color,label=code.name)[0]
    # a.set_visible(False)
    code_t.append(a)
code_legend = fig.legend(code_t,[code.name for code in code_colors], title="Code Colors",loc="outside left lower")
ax.add_artist(code_legend)
ax.add_collection(connections)
ax.add_collection(autocrumbs)
ax.add_collection(mancrumbs)
image = ax.imshow(movie[1])
fig.legend(["auto","manual"],loc="outside left upper",title="Track/Dot Colors")

labels_visible = True
areas_visible = True
do_code_colors = True


def set_dots_visible(visible:bool): autodots.set_visible(visible); mandots.set_visible(visible);
def set_conns_visible(visible:bool): connections.set_visible(visible);
def set_breadcrumbs_visible(visible:bool): autocrumbs.set_visible(visible); mancrumbs.set_visible(visible)
def set_labels_visible(visible:bool): global labels_visible; labels_visible = visible;
def set_areas_visible(visible:bool): global areas_visible; areas_visible = visible;
def set_code_colors_visible(visible:bool): global do_code_colors; code_legend.set_visible(visible); do_code_colors = visible;
def set_cellmasks_visible(visible:bool): masks.set_visible(visible)
def set_nucmasks_visible(visible:bool): nucmasks.set_visible(visible)

visibility:CheckButtons
groups = OrderedDict({
        "dots":set_dots_visible,
        "connections":set_conns_visible,
        "breadcrumbs":set_breadcrumbs_visible,
        "track labels":set_labels_visible})

if mask_type != "None":
    groups["cellmasks"] = set_cellmasks_visible

if nucmask_type != "None":
    groups["nucmasks"] = set_nucmasks_visible


if framed_areas is not None:
    groups["areas"] = set_areas_visible
else:
    set_areas_visible(False)

if mask_type == "labeledmasks":
    groups["code colors"] = set_code_colors_visible
else:
    set_code_colors_visible(False)


def set_visibility(label:str):
    idx = groups.keys().index(label)
    state = visibility.get_status()[idx]
    groups[label](state)
    update_graph(player.i)


button_ax = fig.add_axes((0.05,0.3,0.1,0.4))

visibility = CheckButtons(button_ax,groups.keys(),[True for t in groups])
visibility.on_clicked(set_visibility)

def update_graph(frame:int):
    print("updating graph")


    frame = int(frame)
    currframe = frame
    assert frame in movie.frames

    title.set_text(f"Movie {movie_num}, Frame {frame}")

    [label.set_visible(framed_visibilities[frame][0][k] and labels_visible) for k,label in autolabels.items()]
    [label.set_visible(framed_visibilities[frame][1][k] and labels_visible) for k,label in manlabels.items()]

    if framed_areas is not None:
        [areas[i].set_visible(i < len(framed_areas[frame]) and areas_visible) for i in range(len(areas))]
        [(atext.set_text(str(area)),atext.set_position(pos)) for atext,(pos,area) in zip(areas,framed_areas[frame])]

    [label.set_position(d1[movie_num][k][frame]) for k,label in autolabels.items() if frame in d1[movie_num][k]]
    [label.set_position(d2[movie_num][k][frame]) for k,label in manlabels.items() if frame in d2[movie_num][k]]

    if mask_type == "labeledmasks" and do_code_colors:
        masks.set_facecolors([code_colors[CellFilters(c)] for c in framed_outline_codes[frame].values()])
    elif mask_type != "None":
        masks.set_facecolors([code_colors[CellFilters.valid] for c in range(len(framed_outline_codes[frame]))])

    auto_offsets = list(zip(framed_points[frame][0][0],framed_points[frame][0][1]))
    man_offsets = list(zip(framed_points[frame][1][0],framed_points[frame][1][1]))

    autodots.set_offsets(np.array(auto_offsets))
    mandots.set_offsets(np.array(man_offsets))

    autocrumbs.set_segments(framed_breadcrumbs[frame][0])
    mancrumbs.set_segments(framed_breadcrumbs[frame][1])

    connections.set_segments(framed_associations[frame])
    
    if mask_type != "None":
        masks.set_verts(list(framed_outlines[frame].values()))
    
    if nucmask_type != "None":
        nucmasks.set_verts(list(framed_nuc_outlines[frame].values()))

    im = movie[frame]
    image.set_data(im)
    fig.canvas.draw()

player = Player(fig,update_graph,mini=1,maxi=len(movie.frames))

from IPython import embed; embed()
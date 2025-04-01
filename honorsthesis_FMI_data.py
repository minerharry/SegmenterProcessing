### script to collate analysis data from optoTiam1, optoITSN, and optoPLC movies

from pathlib import Path
from typing import DefaultDict

from make_fmi_plots import make_fmi_plots


tiam_auto_movies = [
    ("migration50",),
    ("migration53",[8,11]),
    ("migration54",),
    ("migration56",),
    ("migration61",),
    ("migration64",),
    ("migration65",),
    ("migration70",[11,12]),
]

tiam_auto_movies_3notch = [
    # ("migration50",),
    ("migration53",[8,11]),
    # ("migration54",),
    # ("migration56",),
    ("migration61",),
    ("migration64",),
    ("migration65",),
    # ("migration70",[11,12]),
]
tiam_auto_movies_4notch = [
    ("migration50",),
    # ("migration53",[8,11]),
    ("migration54",),
    ("migration56",),
    # ("migration61",),
    # ("migration64",),
    # ("migration65",),
    ("migration70",[11,12]),
]

tiam_auto_movies_manual_faithful = [
    ("migration53",),
    ("migration70",), 
    #TODO: WHY IS MIGRATION 50 SO WEIRD??? CHECK **_CONDITION LABELING_** AND MANUAL-AUTOMATIC TRACK AGREEMENT
]

tiam_manual_movies = [ #all of these come with a $manual tag
    ("migration53",),
    # ("migration50",),
    ("migration70",),
]


itsn_auto_movies = [ 
    ("migration44",),
    # ("migration43",[11,12]),
    ("migration42",[6,7,9,10,11]),
    # ("migration41",[1,6]),
    ("itsn1_better",),
    # ("itsn2",)
]

itsn_manual_movies = [
    ("itsn1",),
    ("itsn2",),
]






# plc_auto_movies = [
#     ("plcpeg3",),
#     ("plcpeg4",),
# ]



suffix_delim = " $"
def get_analysis_folder(exp_key,suffix:str|None="",analysis_folder = Path.home()/'OneDrive - University of North Carolina at Chapel Hill/Bear Lab/optotaxis calibration/data/Segmentation Analysis'):
    from fetch_images import keydict
    if exp_key not in keydict:
        raise ValueError("Unrecognized source: " + exp_key)

    exp = keydict[exp_key]
    if suffix:
        exp += suffix_delim + suffix
    
    folder = Path(analysis_folder)/exp
    return folder

def get_analysis_file(exp_key,smoothing:bool|None=True,collection:bool|None=None,**kwargs):
    basename = "track_analysis"
    if smoothing is not None:
        basename += ("_smoothed" if smoothing else "_raw")
    if collection is not None:
        basename += ("_using_cell_collection" if collection else "_no_cell_collection")
    
    name = basename + ".csv"
    file = get_analysis_folder(exp_key,**kwargs)/name
    if not file.exists():
        raise FileNotFoundError(file)
    return file

def get_plot_input(movies:list[tuple[str]|tuple[str,list[int]]],name:str,smoothing:bool|None=True,collection:bool|None=None,suffix:str|None=None):
    files = []
    excludes = []
    for i,S in enumerate(movies):
        key,*exclude = S
        exclude = exclude[0] if len(exclude) > 0 else []

        file = str(get_analysis_file(key,smoothing=smoothing,collection=collection,suffix=suffix))
        files.append(file)
        excludes += [(i,ex) for ex in exclude]

    return {"filenames":tuple(files),"exclude_stages":excludes,"names":name}

if __name__ == "__main__":
    scens:list[tuple[list,str,bool|None,bool|None,str|None]] = [
        (tiam_auto_movies,"Tiam auto smoothed",True,None,None),
        (tiam_auto_movies,"Tiam auto raw",False,None,None),
        (tiam_auto_movies_4notch,"Tiam auto 4notch smoothed",True,None,None),
        (tiam_auto_movies_4notch,"Tiam auto 4notch raw",False,None,None),
        (tiam_auto_movies_3notch,"Tiam auto 3notch smoothed",True,None,None),
        (tiam_auto_movies_3notch,"Tiam auto 3notch raw",False,None,None),
        # (tiam_manual_movies,"Tiam manual smoothed",True,None,"manual"),
        (tiam_auto_movies_manual_faithful,"Tiam auto manual_data_only smoothed",True,None,None),
        (tiam_auto_movies_manual_faithful,"Tiam auto manual_data_only raw",False,None,None),
        (tiam_manual_movies,"Tiam manual raw",False,None,"manual"),
        
        # (itsn_auto_movies,"ITSN auto collection",None,True,None),
        # (itsn_auto_movies,"ITSN auto nocollection",None,False,None),
        # (itsn_manual_movies,"ITSN manual collection",None,True,"manual"),
        # (itsn_manual_movies,"ITSN manual nocollection",None,False,"manual"),
    ]

    argss = [get_plot_input(*scen) for scen in scens]
    a = DefaultDict(list)
    for ar in argss:
        for k in ar:
            a[k].append(ar[k])
    
    figfolder = Path("honors_thesis_output/figures")
    figfolder.mkdir(parents=True,exist_ok=True)
    make_fmi_plots(**a,auto_groups=True,figfolder=figfolder);

    record = {
        "commands":scens,
        "plot_args":argss
    }
    from json import dump
    with open(figfolder/"record.json","w") as f:
        dump(record,f,indent=1)



        



from utils.associations import get_full_associations, try_read_association
from utils.parse_tracks import QCTracks
from utils.filegetter import afn
from utils.trackmasks import get_trackmasks

d1 = QCTracks(afn(key="track1",title="Automatic QC Tracks"))
d2 = QCTracks(afn(key="track2",title="Manual QC Tracks"))

tmfolder,trackmasks = get_trackmasks(afn(title="Tracks Masks Zip",key="trackmasks",filetypes=[("Trackmasks Zip file","*.zip")]))

assoc_path = "associations/assoc_results_53_raw.txt"

try:
    associations,inclusions,remainders = try_read_association(assoc_path)
except:
    associations,inclusions,remainders = get_full_associations(d1,d2,names=("Automatic","Manual"),savepath=assoc_path)


groups:list[tuple[str,QCTracks,list[tuple[int,int]],list[tuple[int,int]]|None]] = [("auto remainders",d1,remainders[0],remainders[0]),("manual remainders",d2,remainders[1],None),("paired tracks",inclusions[0],inclusions[0])]


def stack_remainders(name:str,tracks:QCTracks,selection:list[tuple[int,int]],trackmask_indices:list[tuple[int,int]]|None):
    writer = TiffWriter



for g in groups:
    stack_remainders(*g)


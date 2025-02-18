from libraries.analysis import analyze_experiment_tracks,save_tracks_analysis_csv,scale_tracks
from utils.filegetter import afn,asksaveasfilename
from utils.parse_tracks import QCTracks
trackfile = afn(key="track",title="Choose Tracks File",filetypes=[("Tracks pickle file","*.pkl")])

tracks = QCTracks(trackfile)

do_scale = True
if do_scale:
    tracks = scale_tracks(tracks,"approximate-medoid",1,1)
try:
    analysis = analyze_experiment_tracks(tracks,"approximate-medoid",do_progressbar=True)
except KeyError:
    raise Exception("error encountered analyzing tracks. Are you sure you are using scaled tracks? If not, set do_scale to true!")

out = asksaveasfilename(defaultextension="*.csv",filetypes=[("Tracks Analysis CSV Files","*.csv")]);
save_tracks_analysis_csv(out,analysis,"pixels","frame")
from matplotlib import pyplot as plt
from utils.filegetter import afn
from utils.parse_tracks import QCTracks

a = "C:/Users/Harrison Truscott/Documents/GitHub/cell-tracking/gcp_transfer/Segmentation Analysis/2023.4.2 OptoTiam Exp 53/scaled_qc_tracks_raw.pkl"
print(a)
tracks = QCTracks(a)

for m,track in {k:v for k,v in tracks.items() if k in [1]}.items():
    for t,frame in track.items():
        plt.plot(range(len(frame)),frame['area'],label=f"{m},{t}")
plt.legend()
plt.show()
import base64
from io import BytesIO
from typing import Iterable, Iterator
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.decomposition import PCA
from tifffile import TiffFile
from tqdm import tqdm
from utils.filegetter import afns
import IPython
from PIL import Image

pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        margin=dict(l=20, r=20, t=40, b=0)
    )
)
pio.templates.default = "simple_white+custom"


ts = [TiffFile(f) for f in afns()]
images = [list(i.asarray() for i in t.series[0][:20]) for t in ts]

# a = px.imshow(images[0][0])
# print(a)
# IPython.embed()


class AnimationButtons():
    def play_scatter(frame_duration = 500, transition_duration = 300):
        return dict(label="Play", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": False},
                            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "quadratic-in-out"}}])
    
    def play(frame_duration = 1000, transition_duration = 0):
        return dict(label="Play", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": True},
                            "mode":"immediate",
                            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "linear"}}])
    
    def pause():
        return dict(label="Pause", method="animate", args=
                    [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])

# pca = PCA(n_components=15).fit(X.reshape((X.shape[0], -1)))
# pcs = pca.components_.reshape((-1, X.shape[1], X.shape[2]))

# img, loadings = X[1], pca.transform(X[1].reshape(-1, 1)).T


# reconstructed, distortion, frames = np.zeros_like(X[0]), [], []
frames = []
for i in tqdm(range(len(images[0]))):
    frames.append(go.Frame(
        data = [go.Image(source=px.imshow(imstack[i],binary_string=True).data[0].source) for imstack in images],
        layout = go.Layout(title=rf"$\text{{ Image Reconstruction - Number of PCs: {i+1} }}$")))
        


# class PrintyIterator(Iterator):
#     def __init__(self,it:Iterable):
#         self.iter = iter(it);

#     def __next__(self):
#         return next(self.iter)
    
# class PrintyIterable(list):
#     def __init__(self,source:Iterable) -> None:
#         self.it = source

#     def __iter__(self):
#         return PrintyIterator(self.it)
    
# iterframes = PrintyIterable(frames)        


fig = make_subplots(rows=1, cols=len(images), 
                    subplot_titles=[f"im{i}" for i in range(len(images))],
                    specs=[[{},]*len(images)], row_heights=[500],)
fig.add_traces(data=frames[0]["data"], rows = [1,]*len(images), cols = list(range(1,len(images)+1)))
fig.update(frames=frames)


fig.update_layout(title=frames[0]["layout"]["title"],
                  margin = dict(t = 100),
                  width=800,
                  updatemenus=[dict(type="buttons", buttons=[AnimationButtons.play(), AnimationButtons.pause()])])

if __name__ == "__main__":
    fig.show()
    IPython.embed()
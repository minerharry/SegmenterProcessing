import numpy as np
from scipy.io import loadmat
from utils.filegetter import afn

analysis = afn()

mapname = "PVTAMF"
smoothedname = "PVTAMF"
data = loadmat(analysis)
stats = data["CompStats"]

# from IPython import embed; embed()
# print(data)
protmap = stats[mapname][0,0] #cause it's an array inside an object for some reason
sprotmap = stats[smoothedname][0,0]
protareamap_centroid = stats["ProtAreaTAM"][0,0]
# areamap_centroid = data["PADStats"][0,0]["AngleMap"][0]
# areamap_doubleROI = data["DoubleROIStats"][0,0]["PAD_Stats"][0,0]["AngleMap"] #I want to kill matlab and then myself




import matplotlib.pyplot as plt

protmap = protmap[:,:360] #this technically has 361*, I think so 0* is centered. We don't need that here so make it 360*
sprotmap = sprotmap[:,:360]
protareamap_centroid = protareamap_centroid[:,:360]
# areamap_centroid = areamap_centroid[:,:360]
# areamap_doubleROI = areamap_doubleROI[:,:360]
print(protmap.shape)

up = np.pi/2
angles = np.linspace(up - np.pi, up + np.pi,360,endpoint=False)
maxR = 10
r = np.linspace(0,maxR,protmap.shape[0])
assert angles[180] == up

fig = plt.figure()
ncols,nrows = 2,2

for i,(prot,name) in enumerate(((protmap,"Prot Map"),(sprotmap,"Smoothed Prot Map"),(protareamap_centroid,"Delta Area Map - Centroid"))):#,(areamap_doubleROI,"Delta Area Map - DoubleROI")):
    print(prot.shape,name)
    # .figure(name)
    ax = fig.add_subplot(nrows,ncols,i+1,projection="polar")
    ax.set_title(name)
    for rad,row in zip(r,prot):
        from IPython import embed
        assert len(row) == len(angles),(len(row),len(angles),(row,angles),embed())
        ax.scatter(angles,[rad]*len(angles),c=row,s=5,cmap="RdBu")

mask = data["Final_Mask"]
frames = stats["PSF"][0,0][0,0],stats["LSF"][0,0][0,0] #I hate matlab so much
delta = mask[:,:,frames[1]-1].astype(int) - mask[:,:,frames[0]-1]

ax = fig.add_subplot(nrows,ncols,i+1)
ax.set_title("pad_image")
ax.imshow(delta,cmap="RdBu")

from IPython import embed; embed()
    
plt.show()
    
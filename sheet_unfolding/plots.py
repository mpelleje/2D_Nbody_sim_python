import numpy as np
from . import math

def plot_eulerian_2d(ax, pos, mass, tri=None, idptr=None, maxsize=100., plottri=True, plotseg=True, plotpart=True, mask=None):
    mass = mass.reshape(-1)
    pos = pos.reshape(-1,2)
    
    mass0 = np.min(mass[mass > 0.])
    
    if plottri:
        ax.triplot(pos[:,0], pos[:,1], tri[math.trimask(pos, tri, maxsize=maxsize)], alpha=0.5, linewidth=0.5)
    if plotseg:
        mseg_xy = math.massive_segments(mass, tri, pos=pos, maxsize=maxsize, forplot=True)
        if len(mseg_xy[0]) > 0:
            ax.plot(*mseg_xy, color="black", alpha=0.5)
    
    if plotpart:
        ax.scatter(pos[mass>0,0], pos[mass>0,1], s=5, marker=".", color="black", alpha=0.5)
        
        if np.sum(mass > mass0) > 0:
            sel = mass > mass0
            if mask is not None:
                sel &= mask
            
            if idptr is None:
                ax.scatter(pos[sel,0], pos[sel,1], s=mass[sel]/1e11, color="blue")
            else:
                val = (1.+(idptr*1337)%20)
                ax.scatter(pos[sel,0], pos[sel,1], s=mass[sel]/1e11, c=val[sel], cmap="rainbow", alpha=1.)
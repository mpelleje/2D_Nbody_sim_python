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
                
def plot_id_lagrangian_2d(ax, mass, idptr, mark_boundaries=False, extent=(0,512,0,512), mask=None):
    ngrid = np.int64(np.sqrt(mass.size))
    
    while np.any(idptr != idptr[idptr]):
        idptr = idptr[idptr]
        
    mass = mass.reshape(ngrid,ngrid)
    idptr = idptr.reshape(ngrid,ngrid)
    
    m0 = np.min(mass[mass > 0.])
    sel = mass != m0
    if mask is not None:
        sel = sel & mask.reshape(ngrid, ngrid)
    
    val = (1.+(idptr*1337)%20)    
    boundary = ((np.roll(idptr, 1, axis=1) != idptr) | (np.roll(idptr, 1, axis=0) != idptr)) & (mass != m0)
    
    imargs = dict(extent=extent, origin="lower", interpolation="nearest")
    ax.imshow(np.zeros((ngrid,ngrid)).T, cmap="flag_r", **imargs) # for black background
    ax.imshow(val.T, cmap="tab20c", alpha=1.*sel.T, **imargs)
    
    if mark_boundaries:
        ax.imshow(np.zeros((ngrid,ngrid)).T, cmap="viridis", alpha=0.4*boundary.T, **imargs)

def nb_fof_plot(ax, pos, linking_length=0.3, extent=(0,512,0,512), minlength=10):
    import pyfof
    
    ngrid = np.int64(np.sqrt(np.product(pos.shape[:-1])))
    
    groups = pyfof.friends_of_friends(np.float64(pos.reshape(-1, 2)), linking_length)
    haloes_lag = np.zeros(ngrid*ngrid, dtype = np.int64).flatten()
    for j, g in enumerate(groups):
        if len(g) >= minlength:
            haloes_lag[g] = j+1
    haloes_lag = haloes_lag.reshape(ngrid,ngrid)
    
    val = (1.+(haloes_lag*1337)%20) 
    
    imargs = dict(extent=extent, origin="lower", interpolation="nearest")
    
    ax.imshow(np.zeros((ngrid,ngrid)).T, cmap="flag_r", **imargs) # for black background
    ax.imshow(val.T, cmap="tab20c", alpha=1.*(haloes_lag > 0.).T, **imargs)
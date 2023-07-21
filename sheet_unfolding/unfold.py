from .cpy_unfolding import cpy_unfold
import numpy as np

def tesselation2d(ngrid):
    """Returns triangles (Delaunay) tesselating a uniform grid of ngrid x ngrid particles"""
    i0 = np.arange(ngrid*ngrid).reshape(ngrid,ngrid)
    ileft = np.roll(i0, 1, axis=0)
    iright = np.roll(i0, -1, axis=0)
    iup = np.roll(i0, 1, axis=1)
    idown = np.roll(i0, -1, axis=1)

    tri1 = np.stack((i0, iright, idown), axis=-1).reshape(-1,3)
    tri2 = np.stack((i0, ileft, iup), axis=-1).reshape(-1,3)

    triangles = np.concatenate((tri1, tri2), axis=0)
    
    return triangles

def unfold2d(pos, L, mass=1., mode=None, simp=None, idptr=None):
    assert pos.shape[-1] == 2
    
    pos_in = np.float32(pos.reshape(-1,2))
    mass_in = np.ones(pos_in.shape[:-1], dtype=np.float32) * np.float32(mass)
    
    ngrid = np.int64(np.sqrt(pos_in.shape[0]))
    
    if mode is None:
        mode = 1
        
    if simp is None:
        simp_in = tesselation2d(ngrid)
    else:
        simp_in = np.int64(simp)
        
    if idptr is None:
        idptr_in = np.arange(0, pos_in.shape[0])
    else:
        idptr_in = np.int64(idptr)

    pos, mass, tri, idptr = cpy_unfold(mode, L, pos_in, mass_in, simp_in, idptr_in)
    
    return pos, mass, tri, idptr
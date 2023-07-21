import numpy as np

from .cpy_unfolding import cpy_unfold
from .math import tesselation2d

def unfold2d(pos, L, mass=1., mode=None, simp=None, idptr=None, output_flat=True):
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

    pos_out, mass_out, tri_out, idptr_out = cpy_unfold(mode, L, pos_in, mass_in, simp_in, idptr_in)
    
    tri_out = tri_out[np.min(tri_out, axis=-1) >= 0]
    
    if output_flat:
        return pos_out, mass_out, tri_out, idptr_out
    else:
        return pos_out.reshape(pos.shape), mass_out.reshape(pos.shape[:-1]), tri, idptr.reshape(pos.shape[:-1])
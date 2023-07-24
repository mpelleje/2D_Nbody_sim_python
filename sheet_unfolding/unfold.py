import numpy as np

from .cpy_unfolding import cpy_unfold
from . import math

def unfold2d(pos, L, mass=1., mode=None, simp=None, idptr=None, output_flat=True):
    assert pos.shape[-1] == 2
    
    pos_in = np.float32(pos.reshape(-1,2))
    mass_in = np.ones(len(pos_in), dtype=np.float32) * np.reshape(np.float32(mass), -1)
    
    ngrid = np.int64(np.sqrt(pos_in.shape[0]))
    
    if mode is None:
        mode = 1
        
    if simp is None:
        simp_in = math.tesselation2d(ngrid)
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
        return pos_out.reshape(pos.shape), mass_out.reshape(pos.shape[:-1]), tri_out, idptr_out.reshape(pos.shape[:-1])
    
def multistep_unfold2d(pos_array, L, mass=1., mode=None, simp=None, idptr=None, output_flat=True):
    """pos_array: an array of particle positions at different times, e.g. shape [nsnap, 128, 128, 2]
    other parmeters are like in unfold2d
    """
    shape = pos_array[0].shape[:-1]
    
    pos = pos_array[0].reshape(-1,2)
    mass0 =  np.ones(len(pos), dtype=np.float32) * np.reshape(np.float32(mass), -1)
    
    # Note that the only variables that accumulate changes in this loop are idptr and tri
    for pos in pos_array:
        if idptr is not None:
            pos, mass = math.stick_through_idptr(idptr, pos.reshape(-1,2), mass=mass0, L=L)
        
        pos, mass, simp, idptr = unfold2d(pos, L, mass=mass, mode=mode, simp=simp, idptr=idptr, output_flat=True)
        
        while np.any(idptr[idptr] != idptr):
            idptr = idptr[idptr]
        
    if not output_flat:
        return pos.reshape(shape + (2,)), mass.reshape(shape), simp, idptr.reshape(shape)
    else:
        return pos, mass, simp, idptr
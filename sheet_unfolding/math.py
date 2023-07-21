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

def wrap(dx, L):
    return ((dx + L/2.) % L) - L/2.   

def triangle_area(triangles, pos, L=None):
    v0 = pos[triangles[...,0]]
    v1 = pos[triangles[...,1]]
    v2 = pos[triangles[...,2]]

    #IMPORTANT: Wrapping!!! --> periodic boundaries!!
    u = wrap(v1 - v0, L)
    v = wrap(v2 - v0, L)

    return 0.5 * (u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])

def triangle_parity(triangles, pos, L=None):
    v0 = pos[triangles[...,0]]
    v1 = pos[triangles[...,1]]
    v2 = pos[triangles[...,2]]

    #IMPORTANT: Wrapping!!! --> periodic boundaries!!
    u = wrap(v1 - v0, L)
    v = wrap(v2 - v0, L)
    u_norm = np.linalg.norm(u, axis=-1)
    v_norm = np.linalg.norm(v, axis=-1)

    #compute "normal", return its sign
    normal = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    
    is_zero = np.abs(normal) < (u_norm * v_norm * 1e-14)
    
    result = np.sign(normal) * (~is_zero)

    return result

def trimask(pos, tri, maxsize=None):
    """Creates a mask over triangles, e.g. discarding large triangles"""
    sel = np.min(tri, axis=-1) >= 0 # Filter deactivated
    
    if maxsize is not None:
        r1 = np.linalg.norm(pos.reshape(-1,2)[tri[:,1]] - pos.reshape(-1,2)[tri[:,0]], axis=-1)
        r2 = np.linalg.norm(pos.reshape(-1,2)[tri[:,2]] - pos.reshape(-1,2)[tri[:,1]], axis=-1)
        r3 = np.linalg.norm(pos.reshape(-1,2)[tri[:,0]] - pos.reshape(-1,2)[tri[:,2]], axis=-1)

        sel &= (r1 < maxsize) & (r2 < maxsize) & (r3 < maxsize)
    return sel

def uniform_grid_nd(npix, L=1., endpoint=False):
    assert len(np.shape(npix)) > 0, "Please give npix in form (npixx,npixy,..)"
    
    L = np.ones_like(npix) * L
    
    ardim = [np.linspace(0, Li, npixi, endpoint=endpoint) for npixi, Li in zip(npix, L)]
    q = np.stack(np.meshgrid(*ardim, indexing="ij"), axis=-1)

    return q
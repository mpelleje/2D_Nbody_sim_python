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

def massive_segments(mass, tri, pos=None, maxsize=None, forplot=False):
    """Finds all the line segments connecting two particles with m > m0
    
    This can be used to find the backbone of the Cosmic Web
    """
    m0 = np.min(mass[mass > 0.])
    
    tritype = np.sum((mass.flat[tri] > m0) & (tri >= 0), axis=-1)
    
    lineseg = tri[tritype == 2]
    sort = np.argsort(mass.flat[lineseg], axis=-1)[...,::-1]
    lineseg = np.take_along_axis(lineseg, sort[...,0:2], axis=-1)
    lineseg = np.sort(lineseg, axis=-1)
    if len(lineseg) == 0:
        return [],[]
    lineseg = np.unique(lineseg, axis=0)
    
    if maxsize is not None:
        r = np.linalg.norm(pos.reshape(-1,2)[lineseg[:,1]] - pos.reshape(-1,2)[lineseg[:,0]], axis=-1)

        lineseg = lineseg[r < maxsize]
    
    if forplot:
        x, y = [],[]
        for l in lineseg:
            x.extend([pos[l[0],0], pos[l[1],0], None])
            y.extend([pos[l[0],1], pos[l[1],1], None])
        return x,y
    else:
        return lineseg
    
def stick_through_idptr(idptr, pos, vel=None, mass=None, L=None, contract_idptr=True):
    """Given a distribution of particles and a stickiness relation defined through idptr
    creates the positions, (velocities) and masses of the sticked particles"""
    if contract_idptr:
        while(np.any(idptr != idptr[idptr])):
            idptr = idptr[idptr]
    
    if mass is None:
        mass = np.ones(pos.shape[:-1])
    npart = mass.size
    mstick = np.bincount(idptr, weights=mass.flat, minlength=npart)
    velstick = np.zeros_like(vel)
    posstick = np.zeros_like(pos)
    
    dx = wrap(pos.reshape(npart,-1) - pos.reshape(npart,-1)[idptr], L)
    for i in range(0,pos.shape[-1]):
        if vel is not None:
            pstick = np.bincount(idptr, weights=mass.flatten()*vel[...,i].flatten(), minlength=npart)
            velstick[...,i].flat = pstick / np.clip(mstick, 1e-10, None)
        
        dx_m = np.bincount(idptr, weights=mass.flatten()*dx[...,i].flatten(), minlength=npart)
        posstick[...,i].flat = pos[...,i].flat + dx_m / np.clip(mstick, 1e-10, None)
    
    if vel is not None:
        return posstick, velstick, mstick.reshape(mass.shape)
    else:
        return posstick, mstick.reshape(mass.shape)
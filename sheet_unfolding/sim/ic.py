import numpy as np

from ..math import uniform_grid_nd

def create_modes(npix=256, ndim=3):
    """Creates modes starting from large scales down to small scales
    
    this ensures consistency of random modes accross different resolutions
    """
    
    fmodes_k    = np.zeros([npix]*ndim, dtype=np.complex128)
    
    kinorm = np.fft.fftfreq(npix)
    vknorm = np.array(np.meshgrid(*[kinorm]*ndim, indexing='ij')).astype("f4")

    maxlvl = np.int64(np.log2(npix))
    assert (2**maxlvl == npix), "Only powers of two allowed for npix"
    
    selected = np.zeros(vknorm.shape[1:], dtype=bool)
    
    for lvl in range(maxlvl, 0, -1):
        klow = -2**(-lvl)
        kup  =  2**(-lvl)
        select = np.ones(vknorm.shape[1:], dtype=bool)
        for dim in range(0, ndim):
            select = select & ((vknorm[dim] >= klow) & (vknorm[dim] < kup))
        select = select & (~ selected)
        selected[select] = True
        
        nselect = np.sum(select)
        
        A  = np.random.normal(size=nselect)
        B  = np.random.normal(size=nselect)
        
        fmodes_k[select] = (A * 1j + B)
        
    return fmodes_k

def transfer_cdm_eisenstein_hu(k, Omega_m= 0.30964, Omega_l = 0.26067, Omega_b = 0.04897, h=  0.6766):
    Omega_c = Omega_m - Omega_b
    k = k*h    
    
    h2 = h*h

    theta = 2.728/2.7
    keq = 7.46e-2 * Omega_m * h2 * theta**-2
    ksilk = 1.6 * (Omega_b*h2)**0.52 * (Omega_m * h2)**0.73 * (1. + (10.4 * Omega_m * h2)**-0.95)
    
    zeq = 2.5e4 * Omega_m * h2 * theta**-4
    b1 = 0.313 * (Omega_m * h2)**(-0.419) * (1. + 0.607 *(Omega_m*h2)**0.674)
    b2 = 0.238 * (Omega_m * h2) ** 0.223
    zd = 1291. * (Omega_m * h2)**0.251 / (1. + 0.659 * (Omega_m * h2)**0.828) * (1. + b1*(Omega_b * h2)**b2)

    a1 = (46.9 * Omega_m*h2)**0.670 * (1. + (32.1 * Omega_m * h**2)**(-0.532))
    a2 = (12.0 * Omega_m*h2)**0.424 * (1. + (45.0 * Omega_m * h**2)**(-0.582))
    bb1 = 0.944 / (1. + (458. * Omega_m * h2)**(-0.708))
    bb2 = (0.395 * Omega_m * h2)**-0.0266
 
    alphac = a1**-(Omega_b/Omega_m) * a2**-((Omega_b/Omega_m)**3.)
    betac  = 1./(1. + bb1 * ((Omega_c/Omega_m)**bb2 - 1. ))
    
    q = k / (13.41*keq)
    e = np.e
    
    def R(z):
        return 31.5 * Omega_b * h2 * theta**-4. * (z/1000.)**-1
    
    Rd  = R(zd)
    Req = R(zeq)
    
    betanode = 8.41 * (Omega_m * h2)**0.435
    s  = 2./(3*keq) * np.sqrt(6./Req) * np.log( (np.sqrt(1. + Rd) + np.sqrt(Rd + Req) ) / (1. + np.sqrt(Req)) )
    ks = k*s
    ss = s / (1. + (betanode / ks)**3.)**(1./3.)
    
    
    
    def T0(k, ac, bc):
        C = 14.2/ac + 386./(1. + 69.9 * q**1.08)
        return np.log(e + 1.8 * bc * q) / (np.log(e + 1.8*bc * q) + C*q**2)
   
    def Tc(k):
        f = 1. / (1. + (ks/5.4)**4.)
        return f*T0(k, 1., betac) + (1. - f)*T0(k, alphac, betac)
        
    def G(y):
        return y * (-6. * np.sqrt(1.+y) + (2. + 3.*y) * np.log( (np.sqrt(1. + y) + 1.)/(np.sqrt(1. + y) - 1.)) )
        
    alphab = 2.07 * keq * s * (1 + Rd)**-0.75 * G((1.+zeq)/(1.+zd))
    betab  = 0.5 + (Omega_b / Omega_m) + (3. - 2. * Omega_b/Omega_m) * np.sqrt((17.2 * Omega_m * h2)**2 + 1.)
        
    def j0(x):
        return np.sin(x) / x
        
    def Tb(k):
        return (T0(k, 1., 1.)/(1. + (ks/5.2)**2) + alphab / (1. + (betab/ks)**3. ) * np.exp(-(k/ksilk)**1.4)) * j0(k * ss)
        
    return Omega_b/Omega_m * Tb(k) + Omega_c/Omega_m * Tc(k)



def power_cdm_eisenstein_hu(k, kmin=0.0, kmax=1.0e10, 
                            As = 2.105e-09, ns= 0.9665,
                            Omega_m=0.30964, Omega_l = 0.26067, Omega_b = 0.04897, h= 0.6766): #eisentsein & Hu (1998) power spectrum

    return As * k**ns * (transfer_cdm_eisenstein_hu(k, Omega_m=Omega_m, Omega_l=Omega_l, Omega_b=Omega_b, h=h))**2


def power_eisenstein_smoothed(k, rs=None, **kwargs):
    pk = power_cdm_eisenstein_hu(k)
    if rs is not None:
        pk *= np.exp(-k**2 * rs**2 / 2.)**2
    return pk

def power_eisenstein_smoothed_2d(k, rs=None, dimless=False, **kwargs):
    pk = power_cdm_eisenstein_hu(k) * 2. * k
    if rs is not None:
        pk *= np.exp(-k**2 * rs**2 / 2.)**2
    if dimless:
        return pk * k**2 / (2.*np.pi)
    return pk

def get_kmesh(npix, L, real=False):
    ndim = len(npix)

    L = np.ones_like(npix) * L

    k1d = []
    for i in range(0, ndim):
        if (i == ndim-1) & real:  # last dim can have different shape in real fft
            k1d.append(np.float32(np.fft.fftfreq(npix[i])[0:(npix[i]//2)+1] * (2*np.pi * npix[i]) / L[i]))
        else:
            k1d.append(np.float32(np.fft.fftfreq(npix[i]) * (2*np.pi * npix[i]) / L[i]))

    knd  = np.array(np.meshgrid(*k1d, indexing='ij'))
    knd  = np.rollaxis(knd, 0, ndim+1)  # bring into shape [npix,npix,...,ndim]

    return knd

class IC2DCosmo():
    def __init__(self, ngrid=128, seed=42, L=100., power=power_eisenstein_smoothed_2d, rs=0.5, sig=2., G=43.0071057317063e-10, omega_m=1., norm = None,  **kwargs):
        """Creates the initial conditions for a 2D cosmological simulations
        
        ngrid : number of particles per dimension
        seed : random seed
        L : boxsize
        power : provide a function to define your own power-spectrum p(k)
        rs : smoothing scale (predefined power spectrum)
        sig : density field will get renormalized to have std(delta) = sig at a=1.
        norm : If this is provided, sig will be ignored
        omega_m : omega matter
        """
        
        np.random.seed(seed)
        self.L = L
        self.qi = uniform_grid_nd((ngrid,ngrid), L=L, endpoint=False)
        
        self.ki = get_kmesh((ngrid, ngrid), L)
        
        assert omega_m == 1., "Non EdS has not been properly implemented"

        self.modes = create_modes(npix=ngrid, ndim=2)
        
        self.rs = rs

        self.power = power
        self.kabs = np.sqrt(np.sum(self.ki**2, axis=-1))
        self.phik = self.modes * np.sqrt(self.power(np.clip(self.kabs, 1e-10, None), rs=rs, **kwargs) / np.clip(self.kabs, 1e-10, None)**4 )
        self.phik[self.kabs <= (2.*np.pi / L * 2.)] = 0. # Set mean always to 0
        
        if(norm is None):
            self.norm_num = 1.
            self.norm_num = sig / np.std(self.get_delta())
            
            self.norm = self.norm_num * L**2 / ngrid**2
        else:
            self.norm = norm
            self.norm_num = self.norm / (L**2 / ngrid**2) * L
        
        self.H0 = 100. # Hubble parameter -> units will contain "h"
        self.rhomean = 3.*omega_m*self.H0**2/(8.*np.pi*G)
        self.mass = np.ones((ngrid,ngrid), dtype=np.float64)* (self.rhomean*L**2) / ngrid**2
            
    def get_phi(self, a=1.):
        return self.norm_num * np.real(np.fft.ifftn(self.phik))
    def get_delta(self, a=1.):
        return a * self.norm_num * np.real(np.fft.ifftn(-self.phik * self.kabs**2))
    def get_s(self, a=1):
        return self.norm_num * np.real(np.fft.ifftn((- self.phik * 1j)[...,np.newaxis] * self.ki, axes=(0,1))) * a
    def get_H(self, a=1.):
        return self.norm_num * np.real(np.fft.ifftn((- self.phik)[...,np.newaxis,np.newaxis] * self.ki[...,np.newaxis,:]* self.ki[...,:,np.newaxis], axes=(0,1))) * a
    def get_x(self, a = 1.0):
        return (self.qi + self.get_s(a=a)) % self.L
    def hubble(self, a = 1.):
        return 100.* a**(-3./2.)
    def get_v(self, a = 1.0):
        return self.get_s(a=a) * self.hubble(a) * a**2
    def get_particles(self, mode="xvm", a=1.):
        """Get particles from the Zel'dovich approximation"""
        res = []
        if "x" in mode:
            res.append(self.get_x(a))
        if "v" in mode:
            res.append(self.get_v(a))
        if "m" in mode:
            res.append(self.mass)
            
        if len(res) == 1:
            return res[0]
        else:
            return res
        
    def linear_adhesion_tess(self, a=1., wrapto=None):
        """Return the Adhesion tesselation, periodic boundaries are not correctly handled,
        but you can wrap the domain of interest to the center by setting wrapto
        
        returns simplices, positions and masses
        
        Note that in the Adhesion solution each Lag. simplex maps to a single position and mass
        """
        
        phi = self.get_phi(a=1.)*a - np.sum(0.5*self.qi**2, axis=-1)
        from scipy.spatial import ConvexHull

        if wrapto is None:
            qi = self.qi
        else:
            qi = ((self.qi[...,0:2] - np.array(wrapto) + self.L/2.) % self.L)   + np.array(wrapto)

        x3d = np.stack((self.qi[...,0], self.qi[...,1], phi), axis=-1).reshape(-1,3)

        qhull = ConvexHull(x3d)
        norm = qhull.equations
        simp = qhull.simplices

        def wrap(dx):
            return ((dx + self.L/2.) % self.L) - self.L / 2.

        dx1 = wrap(x3d[simp[...,1],0:2] - x3d[simp[...,0],0:2])
        dx2 = wrap(x3d[simp[...,2],0:2] - x3d[simp[...,0],0:2])
        area = 0.5*np.abs(dx1[:,0]*dx2[:,1] - dx1[:,1]*dx2[:,0])

        xadh = np.stack((norm[:,0] / norm[:,2], norm[:,1] / norm[:,2]), axis=-1)

        return simp, xadh, area*self.rhomean
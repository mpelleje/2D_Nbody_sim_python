import numpy as np
from .ic import get_kmesh
from ..math import uniform_grid_nd, tesselation2d, stick_through_idptr
#from ..unfold import unfold2d
import time

def deposit2d(pos, ngrid, L, mass=1., mode="cic", norm="density"):
    """Calculates a density field by assigning particles to a mesh"""
    
    mass = mass * np.ones(pos.shape[:-1])
    
    bins = np.linspace(0., L, ngrid+1)
    if mode == "ngp":
        rhogrid,_ = np.histogramdd(pos.reshape(-1,2) % L, bins=(bins, bins), weights=mass.reshape(-1))
    elif mode == "cic":
        xred = (pos % L) / (L / ngrid)
        ired = np.int64(np.floor(xred))
        dx = xred - ired # this will be in 0...1

        bins = np.arange(0, ngrid+1)

        rhogrid = 0.
        for ix in (0,1):
            for iy in (0,1):
                weight = np.abs(1-ix-dx[...,0])*np.abs(1-iy-dx[...,1]) * mass
                idep = (xred + np.array((ix, iy))) % ngrid
                rhonew,_ = np.histogramdd(idep.reshape(-1,2), bins=(bins, bins), weights=weight.reshape(-1))
                rhogrid += rhonew
    else:
        raise ValuError("Unknown deposit mode: %s" % mode)
        
    if norm == "sum":
        return rhogrid
    elif norm == "density":
        dV = (bins[1] - bins[0])**2
        return rhogrid / dV
    else:
        raise ValuError("Unknown mode for norm: %s" % norm)
        
def fourier_solve_poisson2d(rho, L, G=43.0071057317063e-10, deconv_cic = False):
    """Calculates the potential and force field from a densiy field"""
    
    ngrid = rho.shape[0]
    
    rhok = np.fft.rfftn(rho)
    ki = get_kmesh((ngrid, ngrid), L, real=True)
    kabs = np.sqrt(np.sum(ki**2, axis=-1))
    
    phik = - 4.*np.pi*G* rhok / np.clip(kabs, 1e-2*np.pi/L, None) ** 2
    phik[kabs==0.] = 0.
    if(deconv_cic):
        kred = ki*L / (2.*phik.shape[0])
        kred[np.abs(kred) < 1e-10] = 1e-10
        phik /= np.clip(np.product((np.sin(kred)/kred)**2, axis=-1), 0., 1)

    phigrid = np.fft.irfftn(phik)
    accgrid = np.fft.irfftn(-phik[...,np.newaxis] * 1j* ki, axes=(0,1))
    
    return phigrid, accgrid
    
def linear_interp2d(fgrid, x, L):
    "periodic linear interpolation in two dimensions"
    ngrid = fgrid.shape[0]

    xred = ((x%L) / (L / ngrid))
    i0 = np.int64(np.floor(xred))
    i0 = i0 % ngrid
    i1 = (i0 + 1) % ngrid
    
    dx = xred - i0
    
    w0 = 1. - (xred - i0)
    w1 = dx
    
    outshape = x.shape[:-1] + (1,)*len(fgrid.shape[2:])
    
    f =  fgrid[i0[...,0], i0[...,1]] * (w0[...,0] * w0[...,1]).reshape(outshape)
    f += fgrid[i1[...,0], i0[...,1]] * (w1[...,0] * w0[...,1]).reshape(outshape)
    f += fgrid[i0[...,0], i1[...,1]] * (w0[...,0] * w1[...,1]).reshape(outshape)
    f += fgrid[i1[...,0], i1[...,1]] * (w1[...,0] * w1[...,1]).reshape(outshape)
    
    return f

class PM2DPotentialField():
    def __init__(self, ngrid=128, L=100., depositmode="cic", G=43.0071057317063e-10, deconv_cic = True):
        self.G = G
        self.L = L
        self.ngrid = ngrid

        self.depositmode = depositmode
        
        self.xi = uniform_grid_nd((ngrid,ngrid), L)
        self.deconv_cic = deconv_cic
        
        super().__init__()
        
    def deposit(self, pos, mass=1.):
        self.rhogrid = deposit2d(pos, ngrid=self.ngrid, L=self.L, mass=mass, mode=self.depositmode, norm="density")
        
        self.phigrid, self.accgrid = fourier_solve_poisson2d(self.rhogrid, self.L, G=self.G, deconv_cic=self.deconv_cic)
        
    def calculate_forces(self, pos, mass):
        self.deposit(pos, mass)
        
        return self.acc(pos)

    def rho(self,x):
        return linear_interp2d(self.rhogrid, x, self.L)
        
    def phi(self,x):
        return linear_interp2d(self.phigrid, x, self.L)
    
    def acc(self,x):
        return linear_interp2d(self.accgrid, x, self.L)
    
class SimulationCallback():
    def before_step(self, sim, a0, a1):
        pass
    def after_step(self, sim, a0, a1):
        pass
    def after_integration(self, sim):
        pass
    
class LoggerCallback(SimulationCallback):
    def __init__(self, alog=[]):
        self.alog = np.array(alog)
        
        self.ai = []
        self.posi = []
        self.veli = []
        self.massi = []
    
    def after_step(self, sim, a0, a1):
        for asnap in self.alog:
            if ((a0 < asnap) & (a1 >= asnap)) | ((a0 == a1) & (a1 == asnap)):
                self.add_snapshot(sim, a1)
    
    def add_snapshot(self, sim, a):
        self.ai.append(a)
        self.posi.append(np.copy(sim.pos))
        self.veli.append(np.copy(sim.vel))
        self.massi.append(np.copy(sim.mass))
    
    def get_log(self, mode="tx"):
        res = []
        if "t" in mode:
            res.append(np.array(self.ai))
        if "x" in mode:
            res.append(np.array(self.posi))
        if "v" in mode:
            res.append(np.array(self.veli))
        if "m" in mode:
            res.append(np.array(self.massi))
        
        return res
    
class StickinessCallback(SimulationCallback):
    def __init__(self, mode=1):
        self.initialized = False
        self.mode = mode
        
    def initialize(self, ngrid):
        self.tri = tesselation2d(ngrid)
        self.idptr = np.arange(0, ngrid*ngrid)
        self.initialized = True
    
    def after_step(self, sim, a0, a1):
        ngrid = sim.pos.shape[0]
        if not self.initialized:
            self.initialize(ngrid)
        
        # Position and masses are stuck together directly
        pos, mass, self.tri, newidptr = unfold2d(sim.pos, L=sim.ics.L, mass=sim.mass, mode=self.mode, simp=self.tri, output_flat=False)
        # Velocities we can handle afterwards through knowledge of the idptr
        pos2,newvel,mass2 = stick_through_idptr(newidptr, sim.pos, vel=sim.vel, mass=sim.mass, L=sim.ics.L, contract_idptr=True)
        
        # print(np.max(np.abs(wrap(pos-pos2)[valid])))
        
        # newidptr only includes recent stickiness relations
        # to keep track of the stickiness history we also update a direct idptr
        self.idptr = newidptr.flat[self.idptr]
        while(np.any(self.idptr != self.idptr[self.idptr])):
            self.idptr = self.idptr[self.idptr]

        sim.vel = newvel
        sim.pos = pos
        sim.mass = mass
    
class CosmologicalSimulation2d():
    def __init__(self, ics, aic=0.05, ngrid_pm=128, dafac_max=0.05, da_max=0.02, callbacks=[],  verbose=1, alog=[], sticky=False):
        self.verbose = verbose
        self.ics = ics
        
        self.potfield = PM2DPotentialField(ngrid=ngrid_pm, L=self.ics.L)
        
        self.pos, self.vel, self.mass = ics.get_particles(mode="xvm", a=aic)
        self.a = aic

        self.callbacks = [] + callbacks
        
        if len(alog) >= 0:
            self.logger = LoggerCallback(alog)
            self.callbacks.append(self.logger)
        else:
            self.logger = None
        
        if sticky:
            self.stickiness = StickinessCallback(mode=1)
            self.callbacks.append(self.stickiness)
        else:
            self.stickiness = None
        
        self.dafac_max, self.da_max = dafac_max, da_max
        
        self.Omega_m = 1.0
        self.Omega_l = 0.0
        
        self.acc = None

        self.integration_step(0.)
        
        self.alog = None

    def hubble(self, a=1.):
        om = self.Omega_m * a**-3
        ol = self.Omega_l
        
        return 100. * (om + ol)**(1./2.)

    def integration_step(self, da):
        a = self.a
        H = self.hubble(a)
        
        driftfac = da/a**2 / (a*H)
        kickfac = da/a / (a*H)
        
        for cb in self.callbacks:
            cb.before_step(self, a, a+da)
        
        if self.acc is None:
            self.acc = self.potfield.calculate_forces(self.pos, self.mass)
        
        self.vel += (0.5 * kickfac) * self.acc
        
        self.pos += driftfac * self.vel
        
        self.acc = self.potfield.calculate_forces(self.pos, self.mass)
        self.vel += (0.5 * kickfac) * self.acc
        
        self.a += da
        
        for cb in self.callbacks:
            cb.after_step(self, self.a-da, self.a)
        
    def get_phi(self):
        return self.potfield.phi(self.pos)
    
    def _define_timestep(self):
        return min(self.dafac_max * self.a, self.da_max)

    def integrate_till(self, afinal, maxsteps=100000):
        assert afinal >= self.a
        
        if self.verbose:
            t0 = time.time()
            da = self._define_timestep()
            print("integrate time %.4e -> %.4e  with da=%.2g  (%d steps)" % (self.a, afinal, da, (afinal - self.a)/da))
        
        for i in range(0, maxsteps):
            daleft = afinal - self.a
            
            da = np.min((daleft, self._define_timestep()))
            
            self.integration_step(da)
            
            if self.a >= afinal:
                break
                if self.verbose:
                    print("Done after %d steps in %.2f seconds" % (i, time.time()-t0))

        if self.a < afinal:
            print("I reached the maxsteps of %d" % maxsteps)
            
        for cb in self.callbacks:
            cb.after_integration(self)
            
    def get_log(self, mode="tx"):
        assert self.logger is not None, "Have not defined any snapshots when creating the simulation"
        
        return self.logger.get_log(mode)


import numpy as np
import sheet_unfolding as su

import time
import argparse

parser = argparse.ArgumentParser(
                    prog='Nbody2D',
                    description="Runs a small 2D N-body simulation. Example Usage: python nbody2d.py -n 128",
                    epilog='')

#parser.add_argument('-d', '--defaultsetup')
parser.add_argument('-n', '--npart', help="particles per dim. (performance critical)")
parser.add_argument('--pmgrid', help="resolution of force-grid (performance critical)")
parser.add_argument('--da', help="max. simulation time-step size (performance critical)")
parser.add_argument('--aic', help="initial condition time")
parser.add_argument('--afin', help="final time")
parser.add_argument('--boxsize', help="Physical boxsize in Mpc")
parser.add_argument('--framepause', help="min. delay between frames")
parser.add_argument('-s', '--drawsteps', help="total number of frames displayed")
parser.add_argument('-a', '--aspectratio', help="frame aspect ratio")
parser.add_argument('-w', '--width', help="frame width in inches")
parser.add_argument('-o', '--outdir', help="If set, save images to directory for creating a movie")
parser.add_argument('-v', '--vismode', help="can be one of 'point', 'tri', 'sheet' or 'none'")
parser.add_argument('--no-overlay', action="store_true", help="set to not overlay scalefactor a=...")
parser.add_argument('-p', '--pointsize', help="The size at which particles are drwan (default 2)")
parser.add_argument('--verbose', action="store_true", help="print timing info")
parser.add_argument('-r', '--repeat', action="store_true", help="keep repeating simulation")

args = parser.parse_args()

c = {"npart": 128, "boxsize": 100., "drawsteps": 100, "framepause": 1e-4, "pmgrid":128, "outdir":None, "aic":0.05, "afin":1.0, "da":0.01, "aspectratio":1.5, "width": 10.0, "no_overlay": False, "vismode": "point", "pointsize":2.0, "verbose":False, "repeat":False}
kwargs = vars(args)
for key in kwargs:
    if kwargs[key] is not None:
        if type(c[key]) == int:
            c[key] = int(kwargs[key])
        elif type(c[key]) == float:
            c[key] = float(kwargs[key])
        else:
            c[key] = kwargs[key]

if c["verbose"]:
    print("Using config", c)

myic = su.sim.IC2DCosmo(ngrid = c["npart"], L=c["boxsize"], rs = 0.1, norm=2e7)
sim = su.sim.CosmologicalSimulation2d(myic, aic=c["aic"], ngrid_pm=c["pmgrid"], verbose=0, da_max=c["da"], dafac_max=0.05)

if not c["vismode"] == "none":
    import matplotlib.pyplot as plt
    if c["vismode"] in ("tri", "sheet"):
        tri = su.math.tesselation2d(c["npart"])

    fig, ax = plt.subplots(1,1, figsize=(c["width"],c["width"]/c["aspectratio"]))
    #plt.tight_layout()
    fig.subplots_adjust(left=0., top=1.0, bottom=0., right=1.0)

    if c["aspectratio"] >= 1:
        ax.set_xlim(0,sim.ics.L)
        ax.set_ylim(0,sim.ics.L / c["aspectratio"])
    else:
        ax.set_xlim(0,sim.ics.L * c["aspectratio"])
        ax.set_ylim(0,sim.ics.L)
        
    scatter, trip = None, None

    #ax.set_xlabel("x [Mpc]")
    #ax.set_ylabel("y [Mpc]")

if c["outdir"] is not None:
    import os
    os.makedirs(c["outdir"], exist_ok=True)
    
done = False
while not done:
    ai = np.linspace(c["aic"], c["afin"], c["drawsteps"])
    sim = su.sim.CosmologicalSimulation2d(myic, aic=c["aic"], ngrid_pm=c["pmgrid"], verbose=0, da_max=c["da"], dafac_max=0.05)


    for i in range(c["drawsteps"]):
        t0 = time.time()
        sim.integrate_till(ai[i])
        t1 = time.time()

        pos = sim.pos % sim.ics.L


        if (not c["no_overlay"]) & (c["vismode"] != "none"):
            ax.set_title("a = %.2f" % sim.a, y=1.0, pad=-35, color="blue", fontsize=20)

        if c["vismode"] == "none":
            pass
        elif c["vismode"] == "point":
            if scatter is not None:
                scatter.remove()
            scatter = ax.scatter(pos[...,0].flat, pos[...,1].flat, marker=".", s=c["pointsize"], alpha=0.5, color="black")
        elif c["vismode"] == "tri":
            if scatter is not None:
                scatter.remove()
            scatter = ax.scatter(pos[...,0].flat, pos[...,1].flat, marker=".", s=c["pointsize"], alpha=0.5, color="blue")

            if trip is not None:
                for l in ax.lines:
                    l.remove()
                #ax.patches[-1].remove()

            pos = sim.pos[...,0:2].reshape(-1,2) % sim.ics.L
            trimask = su.math.trimask(pos, tri, maxsize=sim.ics.L * 0.4)
            trip = ax.triplot(pos[:,0], pos[:,1], tri[trimask], alpha=1.0, linewidth=0.25, color="black")
            #scatter = ax.scatter(pos[...,0].flat, pos[...,1].flat, marker=".", s=2, alpha=0.5, color="black")
        elif c["vismode"] == "sheet":
            if trip is not None:
                trip.remove()

            pos = sim.pos[...,0:2].reshape(-1,2) % sim.ics.L
            trimask = su.math.trimask(pos, tri, maxsize=sim.ics.L * 0.4)

            vol = su.math.triangle_area(tri, pos, sim.ics.L) / (0.5 * (sim.ics.L / c["npart"])**2)
            color = 1./np.abs(vol)

            trip = ax.tripcolor(pos[:,0], pos[:,1], tri[trimask], facecolors=color[trimask], shading="flat", cmap="Greys", vmin=0., vmax=2., alpha=0.25)
        else:
            raise ValueError("Unknown vismode: %s" % c["vismode"]) 

        t2 = time.time()

        if c["outdir"] is not None:
            fig.savefig("%s/image_%04d.png" % (c["outdir"], i))
            #plt.show()
        elif c["vismode"] != "none":
            plt.pause(max(c["framepause"], 1e-5))

        t3 = time.time()

        if c["verbose"]:
            print("frame %d, a=%.2f, integrate = %.0fms draw = %.0fms display = %.0fms" % (i, sim.a, 1000.*(t1-t0), 1000.*(t2-t1), 1000.*(t3-t2)))

    if c["outdir"] is not None:
        print("To create the video, paste the following line:")
        print('ffmpeg -framerate 20 -i "%s/image_%s.png" -c:v libx264 -pix_fmt yuv420p %s.mp4' % (c["outdir"], "%04d", c["outdir"]))
        
    if not c["repeat"]:
        done = True

if c["vismode"] != "none":
    plt.show()


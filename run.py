import SebMC
from SebMC import Vector3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

u235 = SebMC.Nuclide("U235", 235, 0.1e23)
u235.xs_fission_0 = 92.5
u238 = SebMC.Nuclide("U238", 238,  1.93e23)
u238.xs_absorbtion_0 = 85
u238.E0 = 2010
u238.Gamma = 25
graphite = SebMC.Nuclide("graphite", 12,  2e23)
graphite.xs_scatter_0 = 4.8
hom = SebMC.Material([u235, u238, graphite],300)

s1 = SebMC.Cell(1,0,1,hom,Vector3D(0, 0, 0),1)
s2 = SebMC.Cell(2,0,1,hom,Vector3D(3,0,0),0.5)
s3 = SebMC.Cell(3,0,1,hom,Vector3D(0,3,0),6)
s4 = SebMC.Cell(4,0,1,hom,Vector3D(0.5,3.5,0),2)
s5 = SebMC.Cell(5,0,1,hom,Vector3D(-5.5,-4,0.5),1.5)
s6 = SebMC.Cell(73,0,1,hom,Vector3D(0.5,3.5,0),1)
s7 = SebMC.Cell(41,0,1,hom,Vector3D(0,0,0),0.9)
bound = SebMC.Cell(0,1,1,hom,Vector3D(0,0,0),25)

geom = SebMC.Geometry([bound])
geom.plot_slice('xy',0)
run = SebMC.Run(geom)

spos = Vector3D(0,0,0)
sen = 5e3

# neutron_track = SebMC.Track(geom, spos, sen)
# history1 = neutron_track.track()

N=30
source = list(np.zeros(N))
for i in range(N):
    source[i] = spos # run.IsotropicSphereSource(run.geometry.root.radius)

#run.history_quiver(run.cycle(N,source))

run.run(1000,10,50)
run.plot_k_eff()
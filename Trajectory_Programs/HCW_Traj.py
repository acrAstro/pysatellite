import numpy as np
import cvxpy as cvx
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from RelativeMotion import *
print cvx.installed_solvers()

mu = 3.986*10**14
a = 6678*10**3
n = np.sqrt(mu/a**3)
nu = 6
ny = 6

S = RelativeMotion()
Ac = S.hcw(n)
Bc = S.inputMats(nu)
method = 'zoh'
Ts = 10
sysc,sysd =  S.discretizeSystem(n,nu,ny,Ts,method)
t0 = 0
tf = int((2*np.pi/n))

x0 = 100
X0 = np.matrix([[x0,-100,300,0,-2*n*x0,0]])
Xf = np.matrix([[0,0,0,0,0,0]])
Umax= 0.1
sysc,sysd = S.discretizeSystem(n,nu,ny,Ts,method)

X = S.discHCWSim_initial(X0,t0,tf,Ts,n,nu,ny,method)

Xc, Uc = S.minFuelHCW(np.transpose(X0),np.transpose(Xf),t0,tf,Ts,n,nu,ny,method,Umax)

print X.shape, Xc.shape

fig,(ax1,ax2) = plt.subplots(1,2, figsize = (12,4), subplot_kw = {'projection':'3d'})
for ax,xyz,c in [(ax1,X,'b'),(ax2,Xc,'r')]:
    ax.plot(xyz[0,:],xyz[1,:],xyz[2,:],c)
    ax.set_xlabel('$x$',fontsize = 16)
    ax.set_ylabel('$y$',fontsize = 16)
    ax.set_zlabel('$z$',fontsize = 16)

plt.show()


#fig3d = plt.figure()
#ax = fig3d.add_subplot(111,projection = '3d')
#ax.plot(X[0,:],X[1,:],X[2,:],'r')
#ax.plot(Xc[0,:],Xc[1,:],Xc[2,:])
#plt.show()

import numpy as np
import control as ctrl
import cvxpy as cvx

class RelativeMotion:

    def __init__(self):
        self.data = []
    
    def hcw(self,n):
        self.n = n
        n2 = n**2
        A = np.matrix([[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[3*n2,0,0,0,2*n,0],[0,0,0,-2*n,0,0],[0,0,-n2,0,0,0]])

        return A

    def inputMats(self,nu):
        self.nu = nu
        if nu == 2:
            B = np.matrix([[0,0],[0,0],[0,0],[0,0],[1,0],[0,1]])
        elif nu == 3:
            B = np.matrix([[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
        elif nu == 4:
            B = np.matrix([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,1,0,0,-1,0],[0,0,1,0,0,-1]])
        elif nu == 6:
            B = np.matrix([[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[1,0,0,-1,0,0],[0,1,0,0,-1,0],[0,0,1,0,0,-1]])
        else:
            B = 'Define new input matrix attribute'

        return B

    def observer(self,ny):
        if ny == 3:
            C = np.matrix([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
        elif ny == 6:
            C = np.matrix([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        else:
            C = 'Define proper observer'

        return C
    
    def feedForward(self,nu,ny):
        self.nu = nu
        self.ny = ny
        D = np.zeros((ny,nu))

        return D
    
    def discretizeSystem(self,n,nu,ny,Ts,method):
        self.Ts = Ts
        self.method = method
        
        sysc = ctrl.ss(self.hcw(n),self.inputMats(nu),self.observer(ny),self.feedForward(ny,nu))
        sysd = ctrl.matlab.c2d(sysc,Ts,method)

        return sysc,sysd

    def discHCWSim_initial(self,X0,t0,tf,Ts,n,nu,ny,method):
        self.X0 = X0
        self.t0 = t0
        self.tf = tf
        self.Ts = Ts
        self.n = n
        self.nu = nu
        self.ny = ny
        self.method = method

        sysc,sysd = self.discretizeSystem(n,nu,ny,Ts,method)

        A = sysd.A
        B = sysd.B
        C = sysd.C
        D = sysd.D

        time = np.arange(t0,tf,Ts)
        numSteps = len(time)

        X = np.zeros((6,numSteps+1))
        X[:,0] = X0

        for ii in range(numSteps):
            X[:,ii+1] = A.dot(X[:,ii])

        return X

    def minFuelHCW(self,X0,Xf,t0,tf,Ts,n,nu,ny,method,Umax):
        self.X0 = X0
        self.Xf = Xf
        self.t0 = t0
        self.tf = tf
        self.Ts = Ts
        self.n  = n
        self.nu = nu
        self.ny = ny
        self.method = method
        #self.solver = solver

        time = np.arange(t0,tf,Ts)
        numSteps = len(time)
        
        sysc,sysd = self.discretizeSystem(n,nu,ny,Ts,method)

        A = sysd.A
        B = sysd.B

        x = cvx.Variable(6,numSteps+1)
        u = cvx.Variable(nu,numSteps)
        
        states = []
        umin = np.zeros((nu,1))
        umax = Umax*np.ones((nu,1))
        for i in range(numSteps):
            cost = cvx.norm(u[:,i],1)
            constraints = [x[:,i+1] == A*x[:,i] + B*u[:,i],
                           u[0:nu/2-1,i] <= umax,u[nu/2:end,i] <= umax]
            states.append(cvx.Problem(cvx.Minimize(cost),constraints))

        prob = sum(states)
        prob.constraints += [x[:,0] == X0, x[:,numSteps] == Xf]
        prob.solve(verbose = True, solver = 'GUROBI')

        X = x.value
        U = u.value
        
        return X, U

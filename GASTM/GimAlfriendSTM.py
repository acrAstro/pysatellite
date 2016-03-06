import numpy as np

class GimAlfriendSTM:
# Class to compute the time evolution of the relative motion of two satellites
# under the influence of J2
    def __init__(self):
        # Instantiate class
        self.data = []
    
    def theta2lam(self,a,theta,q1,q2):
        # Calculates mean longitude from true longitude
        
        # Inputs
        # a = semi-major axis
        # theta = true longitude
        # q1 = e*cos(w)
        # q2 = e*sin(w)
        
        self.a = a
        self.theta = theta
        self.q1 = q1
        self.q2 = q2

        eta = np.sqrt(1 - q1**2 - q2**2)        
        beta = 1/(eta*(1+eta))
        R = a*eta**2/(1 + q1*np.cos(theta) + q2*np.sin(theta))

        x1 = R*(1+beta*q1**2)*np.sin(theta) - beta*R*q1*q2*np.cos(theta) + a*q2
        x2 = R*(1+beta*q2**2)*np.cos(theta) - beta*R*q1*q2*np.sin(theta) + a*q1

        F = np.arctan2(x1,x2)

        lam = F - q1*np.sin(F) + q2*np.cos(F)
        
        while lam < 0:
            lam = lam + 2*np.pi
        while lam >= 2*np.pi:
            lam = lam - 2*np.pi
        
        if theta < 0:
            kkPlus = 0
            quadPlus = 0
            while theta < 0:
                kkPlus = kkPlus + 1
                theta = theta + 2*np.pi
            if theta < np.pi/2 && lam > np.pi:
                quadPlus = 1
            elif lam < np.pi/2 && theta > np.pi:
                quadPlus = -1
            lam = lam - (kkplud+quadPlus)*2*np.pi
        else:
            kkMinus = 0
            quadMinus = 0
            while theta >= 2*np.pi:
                kkMinus = kkMinus + 1
                theta = theta - 2*np.pi
            if theta < np.pi/2 && lam > np.pi:
                quadMinus = -1
            elif lam < np.pi/2 && theta > np.pi:
                quadMinus = 1
            lam = lam + (kkMinus+quadMinus)*2*np.pi
        return lam
    
    def MeanElemsSTM(self,J2,t,ICSc,Re,mu,tol):
        # State transition matrix of mean orbital elements due to J2
        
        # Inputs: time, initial non-singular relative orbital elements
        # t[0] = t_0
        # t[1] = t_1
        
        # ICSc[0] = a0
        # ICSc[1] = argLat0
        # ICSc[2] = inc0
        # ICSc[3] = q10
        # ICSc[4] = q20
        # ICSc[5] = RAAN0
        
        # Outputs:
        # Phi_J2 = 6x6 state transition matrix for mean orbital elements
        # cond_c = final elements after time step
        
        self.J2 = J2
        self.t = t
        self.ICSc = ICSc
        self.Re = Re
        self.mu= mu
        self.tol = tol        
        
        Re2 = Re*Re
        gamma = 3*J2*Re2
        
        t0 = t[0]
        t = t[1]
        
        a0 = ICSc[0]
        argLat0 = ICSc[1]
        inc0 = ICSc[2]
        q10 = ICSc[3]
        q20 = ICSc[4]
        RAAN0 = ICSc[5]
        
        n0 = np.sqrt(mu/a0**3)
        p0 = a0*(1 - q10**2 - q20**2)
        R0 = p0/(1 + q10*np.cos(argLat0) + q20*np.sin(argLat0))
        Vr0 = np.sqrt(mu/p0)*(q10*np.sin(argLat0) - q20*np.cos(argLat0))
        Vt0 = np.sqrt(mu/p0)*(1 + q10*np.cos(argLat0) + q20*np.sin(argLat0))
        eta0 = np.sqrt(1 - q10**2 - q20**2)
        
        lam0 = self.theta2lam(a0,argLat0,q10,q20)
        # Secular variations caused by J2
        aDot = 0
        incDot = 0
        argPerDot = gamma*(1/4)*(n0/p0**2)*(5*np.cos(inc0)**2 - 1)
        sDot = np.sin(argPerDot*(t-t0))
        cDot = np.cos(argPerDot*(t-t0))
        
        lamDot = n0+ gamma*(1/4)*(n0/p0**2)*((5+3*eta0)*np.cos(inc0)**2 - (1+eta0))
        RAANDot = -gamma*(1/2)*(n0/p0**2)*np.cos(inc0)
        
        a = a0 + aDot*(t-t0)
        inc = inc0 + incDot*(t-t0)
        Omega = RAAN0 + RAANDot*(t-t0)
        q1 = q10*np.cos(argPerDot*(t-t0)) - q20*np.sin(argPerDot*(t-t0))
        q2 = q10*np.sin(argPerDot*(t-t0)) + q20*np.cos(argPerDot*(t-t0))
        
        lam = lam0 + lamDot*(t-t0)
        
        theta = self.lam2theta(lam,q1,q2,tol)
        
        n = np.sqrt(mu/a**3)
        p = a*(1 - q1**2 - q2**2)
        R = p/(1 + q1*np.cos(theta) + q2*np.sin(theta))
        Vr = np.sqrt(mu/p)*(q1*np.sin(theta) - q2*np.cos(theta))
        Vt = np.sqrt(mu/p)*(1 + q1*np.cos(theta) + q2*np.sin(theta))
        eta = np.sqrt(1 - q1**2 - q2**2)        
        
        G_theta = n*R/Vt
        G_theta0 = -n0*R0/Vt0
        G_q1 = (q1*Vr)/(eta*Vt) + q2/(eta*(1+eta)) - eta*R*(a+R)*(q2+np.sin(theta))/(p**2)
        G_q10 = -(q10*Vr0)/(eta0*Vt0) - q20/(eta0*(1+eta0)) + eta0*R0*(a0+R0)*(q20+np.sin(argLat0))/(p0**2)
        G_q2 = (q2*Vr)/(eta*Vt) - q1/(eta*(1+eta)) + eta*R*(a+R)*(q1+np.cos(theta))/(p**2)
        G_q20 = -(q20*Vr0)/(eta0*Vt0) + q10/(eta0*(1+eta0)) - eta0*R0*(a0+R0)*(q10+np.cos(argLat0))/(p0**2)
        K = 1 + G_q1*(q10*sDot+q20*cDot) - G_q2*(q10*cDot-q20*sDot)

        # Transformation Matrix A
        phi11 = 1
        phi12 = 0
        phi13 = 0
        phi14 = 0
        phi15 = 0
        phi16 = 0
        
        phi21 = -((t-t0)/G_theta)*((3*n0/(2*a0)) + (7*gamma/8)*(n0/(a0*p0**2))*(eta0*(3*np.cos(inc0)**2-1) + K*(5*np.cos(inc0)**2-1)))
        phi22 = -(G_theta0/G_theta)
        phi23 = -((t-t0)/G_theta)*(gamma/2)*(n0*np.sin(inc0)*np.cos(inc0)/p0**2)*(3*eta0+5*K)
        phi24 = -((G_q10+cDot*G_q1+sDot*G_q2)/G_theta) + ((t-t0)/G_theta)*(gamma/4)*(n0*a0*q10/p0**3)*(3*eta0*(3*np.cos(inc0)**2-1) + 4*K*(5*np.cos(inc0)**2-1))
        phi25 = -((G_q20-sDot*G_q1+cDot*G_q2)/G_theta) + ((t-t0)/G_theta)*(gamma/4)*(n0*a0*q20/p0**3)*(3*eta0*(3*np.cos(inc0)**2-1) + 4*K*(5*np.cos(inc0)**2-1))
        phi26 = 0
        
        phi31 = 0
        phi32 = 0
        phi33 = 1
        phi34 = 0
        phi35 = 0
        phi36 = 0
        
        phi41 = (7*gamma/8)*(n0*(q10*sDot+q20*cDot)*(5*np.cos(inc0)**2-1)/(a0*p0**2))*(t-t0)
        phi42 = 0
        phi43 = (5*gamma/2)*(n0*(q10*sDot+q20*cDot)*(np.sin(inc0)*np.cos(inc0))/p0**2)*(t-t0)
        phi44 = cDot - gamma*(n0*a0*q10*(q10*sDot+q20*cDot)*(5*np.cos(inc0)**2-1)/p0**3)*(t-t0)
        phi45 = -sDot - gamma*(n0*a0*q20*(q10*sDot+q20*cDot)*(5*np.cos(inc0)**2-1)/p0**3)*(t-t0)
        phi46 = 0
        
        phi51 = -(7*gamma/8)*(n0*(q10*cDot-q20*sDot)*(5*np.cos(inc0)**2-1)/(a0*p0**2))*(t-t0)
        phi52 = 0
        phi53 = -(5*gamma/2)*(n0*(q10*cDot-q20*sDot)*(np.sin(inc0)*np.cos(inc0))/p0**2)*(t-t0)
        phi54 = sDot + gamma*(n0*a0*q10*(q10*cDot-q20*sDot)*(5*np.cos(inc0)**2-1)/p0**3)*(t-t0)
        phi55 = cDot + gamma*(n0*a0*q20*(q10*cDot-q20*sDot)*(5*np.cos(inc0)**2-1)/p0**3)*(t-t0)
        phi56 = 0
        
        phi61 = (7*gamma/4)*(n0*np.cos(inc0)/(a0*p0**2))*(t-t0)
        phi62 = 0
        phi63 = (gamma/2)*(n0*np.sin(inc0)/p0**2)*(t-t0)
        phi64 = -(2*gamma)*(n0*a0*q10*np.cos(inc0)/p0**3)*(t-t0)
        phi65 = -(2*gamma)*(n0*a0*q20*np.cos(inc0)/p0**3)*(t-t0)
        phi66 = 1
        
        Phi_J2 = np.matrix([[phi11,phi12,phi13,phi14,phi15,phi16],[phi21,phi22,phi23,phi24,phi25,phi26],[phi31,phi32,phi33,phi34,phi35,phi36],[phi41,phi42,phi43,phi44,phi45,phi46],[phi51,phi52,phi53,phi54,phi55,phi56],[phi61,phi62,phi63,phi64,phi65,phi66]])
        cond_c = np.matrix([a, theta, inc, q1, q2, Omega]).T
        return Phi_J2, cond_c
        
        
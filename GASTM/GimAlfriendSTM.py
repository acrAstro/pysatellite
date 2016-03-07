import numpy as np

class GimAlfriendSTM:
# Class to compute the time evolution of the relative motion of two satellites
# under the influence of J2
    def __init__(self):
        # Instantiate class
        self.data = []
    
    def lam2theta(self,lam,q1,q2,tol):
        # Calculates true longitude from mean longitude
        
        # Inputs:
        # lambda = mean longitude, M + w
        # q1 = e*cos(w)
        # q2 = e*sin(w)
        # tol = tolerance for solving Kepler's equation
        
        # Outputs:
        # theta = true longitude, f + w
        # F = eccentric longitude, E + w
        
        eta = np.sqrt(1 - q1**2 - q2**2)
       
        # Solve Kepler's Equation via Newton-Raphson      
        F = lam
        FF = 1
        while abs(FF) > tol:
            FF = lam - (F - q1*np.sin(F) + q2*np.cos(F))
            dFFdF = -(1 - q1*np.cos(F) - q2*np.sin(F))
            delF = -FF/dFFdF
            F = F + delF
        
        # True longitude
        x1 = (1+eta)*(eta*np.sin(F) - q2) + q2*(q1*np.cos(F) + q2*np.sin(F))
        x2 = (1+eta)*(eta*np.cos(F) - q1) + q1*(q1*np.cos(F) + q2*np.sin(F))
        theta = np.arctan2(x1,x2)
        
        while theta < 0:
            theta = theta + 2*np.pi
        while theta >= 2*np.pi:
            theta = theta - 2*np.pi
            
        if lam < 0:
            kkPlus = 0
            quadPlus = 0
            while lam < 0:
                kkPlus = kkPlus + 1
                lam = lam + 2*np.pi
            if lam < (np.pi/2) and theta > np.pi:
                quadPlus = 1
            elif theta < np.pi/2 and lam > np.pi:
                quadPlus = -1
            theta = theta - (kkPlus+quadPlus)*2*np.pi
        else:
            kkMinus = 0
            quadMinus = 0
            while lam >= 2*np.pi:
                kkMinus = kkMinus + 1
                lam = lam - 2*np.pi
            if lam < (np.pi/2) and theta > np.pi:
                quadMinus = -1
            elif theta < np.pi/2 and lam > np.pi:
                quadMinus = 1
            theta = theta + (kkMinus + quadMinus)*2*np.pi

        return theta, F
    
    def theta2lam(self,a,theta,q1,q2):
        # Calculates mean longitude from true longitude
        
        # Inputs:
        # a = semi-major axis
        # theta = true longitude, f + w
        # q1 = e*cos(w)
        # q2 = e*sin(w)
        
        # Output:
        # lambda = mean longitude
        
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
            if theta < (np.pi/2) and lam > np.pi:
                quadPlus = 1
            elif lam < (np.pi/2) and theta > np.pi:
                quadPlus = -1
            lam = lam - (kkPlus+quadPlus)*2*np.pi
        else:
            kkMinus = 0
            quadMinus = 0
            while theta >= 2*np.pi:
                kkMinus = kkMinus + 1
                theta = theta - 2*np.pi
            if theta < (np.pi/2) and lam > np.pi:
                quadMinus = -1
            elif lam < (np.pi/2) and theta > np.pi:
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
        
    def SigmaMatrix(self,J2,elems,Re,mu):
        # Computes the perturbed linear mapping between relative orbital
        # elements and local vertical, local horizontal coordinates
        
        # Inputs:
        # elems[0] = a
        # elems[1] = theta
        # elems[2] = inc
        # elems[3] = q1
        # elems[4] = q2
        # elems[5] = Omega

        # Ouput:
        # Sigma = 6x6 mapping
        
        self.J2 = J2
        self.elems = elems
        self.Re = Re
        self.mu = mu        
        
        gamma = 3*J2*Re**2
        a= elems[0]
        argLat = elems[1]
        inc = elems[2]
        q1 = elems[3]
        q2 = elems[4]
        
        p = a*(1 - q1**2 - q2**2)
        R = p/(1 + q1*np.cos(argLat) + q2*np.sin(argLat))
        Vr = np.sqrt(mu/p)*(q1*np.sin(argLat) - q2*np.cos(argLat))
        Vt = np.sqrt(mu/p)*(1 + q1*np.cos(argLat) + q2*np.sin(argLat))
        
        A11 = R/a
        A12 = R*Vr/Vt
        A13 = 0
        A14 = -(2*a*R*q1/p) - (R**2/p)*np.cos(argLat)
        A15 = -(2*a*R*q2/p) - (R**2/p)*np.sin(argLat)
        A16 = 0
        
        A21 = -1/2*Vr/a
        A22 = np.sqrt(mu/p)*((p/R) - 1)
        A23 = 0
        A24 = (Vr*a*q1/p) + np.sqrt(mu/p)*np.sin(argLat)
        A25 = (Vr*a*q2/p) - np.sqrt(mu/p)*np.cos(argLat)
        A26 = 0
        
        A31 = 0
        A32 = R
        A33 = 0
        A34 = 0
        A35 = 0
        A36 = R*np.cos(inc)
        
        A41 = -3/2*Vt/a
        A42 = -Vr
        A43 = 0
        A44 = 3*Vt*a*q1/p + 2*np.sqrt(mu/p)*np.cos(argLat)
        A45 = 3*Vt*a*q2/p + 2*np.sqrt(mu/p)*np.sin(argLat)
        A46 = Vr*np.cos(inc)
        
        A51 = 0
        A52 = 0
        A53 = R*np.sin(argLat)
        A54 = 0
        A55 = 0
        A56 = -R*np.cos(argLat)*np.sin(inc)
        
        A61 = 0
        A62 = 0
        A63 = Vt*np.cos(argLat) + Vr*np.sin(argLat)
        A64 = 0
        A65 = 0
        A66 = (Vt*np.sin(argLat) - Vr*np.cos(argLat))*np.sin(inc)

        A = np.matrix([[A11,A12,A13,A14,A15,A16],[A21,A22,A23,A24,A25,A26],[A31,A32,A33,A34,A35,A36],[A41,A42,A43,A44,A45,A46],[A51,A52,A53,A54,A55,A56],[A61,A62,A63,A64,A65,A66]])
        
        B11 = 0
        B12 = 0
        B13 = 0
        B14 = 0
        B15 = 0
        B16 = 0
        
        B21 = 0
        B22 = 0
        B23 = 0
        B24 = 0
        B25 = 0
        B26 = 0
        
        B31 = 0
        B32 = 0
        B33 = 0
        B34 = 0
        B35 = 0
        B36 = 0
        
        B41 = 0
        B42 = 0
        B43 = -Vt*np.sin(inc)*np.cos(inc)*np.sin(argLat)**2/(p*R)
        B44 = 0
        B45 = 0
        B46 = Vt*np.sin(inc)**2*np.cos(inc)*np.sin(argLat)*np.cos(argLat)/(p*R)
        
        B51 = 0
        B52 = 0
        B53 = 0
        B54 = 0
        B55 = 0
        B56 = 0
        
        B61 = 0
        B62 = Vt*np.sin(inc)*np.cos(inc)*np.sin(argLat)/(p*R)
        B63 = 0
        B64 = 0
        B65 = 0
        B66 = Vt*np.sin(inc)*np.cos(inc)**2*np.sin(argLat)/(p*R)
        
        B = np.matrix([[B11,B12,B13,B14,B15,B16],[B21,B22,B23,B24,B25,B26],[B31,B32,B33,B34,B35,B36],[B41,B42,B43,B44,B45,B46],[B51,B52,B53,B54,B55,B56],[B61,B62,B63,B64,B65,B66]])
        Sigma = A + gamma*B
        return Sigma
        
    def SigmaInverse(self,J2,elems,Re,mu,tol):
        # Calculates the inverse linear map from Cartesian elements to relative orbital elements
        self.J2 = J2
        self.elems = elems
        self.Re = Re
        self.mu = mu
        self.tol = tol
        gamma   = 3*J2*Re**2
        a       = elems(1)
        argLat  = elems(2)
        inc     = elems(3)
        q1      = elems(4)
        q2      = elems(5)
        RAAN    = elems(6)
        
        Hamiltonian = -mu/(2*a)
        
        p = a*(1 - q1**2 - q2**2)
        R = p/(1 + q1*np.cos(argLat) + q2*np.sin(argLat))
        Vr = np.sqrt(mu/p)*(q1*np.sin(argLat) - q2*np.cos(argLat))
        Vt = np.sqrt(mu/p)*(1 + q1*np.cos(argLat) + q2*np.sin(argLat))
        
        q1tilde = q1*np.cos(RAAN) - q2*np.sin(RAAN)
        q2tilde = q1*np.sin(RAAN) + q2*np.cos(RAAN)
        p1 = np.tan(inc/2)*np.cos(RAAN)
        p2 = np.tan(inc/2)*np.sin(RAAN)
        
        if ((p1==0) and (p2==0)):
            p1p2 = p1**2 + p2**2 + tol
        else:
            p1p2 = p1**2 + p2**2
    
        InvT11 = 1
        InvT12 = 0
        InvT13 = 0
        InvT14 = 0
        InvT15 = 0
        InvT16 = 0
        
        InvT21 = 0
        InvT22 = 1
        InvT23 = 0
        InvT24 = 0
        InvT25 = p2/(p1p2)
        InvT26 = -p1/(p1p2)
        
        InvT31 = 0
        InvT32 = 0
        InvT33 = 0
        InvT34 = 0
        InvT35 = 2*p1/(np.sqrt(p1p2)*(1+p1p2))
        InvT36 = 2*p2/(np.sqrt(p1p2)*(1+p1p2))
        
        InvT41 = 0
        InvT42 = 0
        InvT43 = p1/(np.sqrt(p1p2))
        InvT44 = p2/(np.sqrt(p1p2))
        InvT45 = -p2*(p1*q2tilde-p2*q1tilde)/(p1p2**(3/2))
        InvT46 = p1*(p1*q2tilde-p2*q1tilde)/(p1p2**(3/2))
        
        InvT51 = 0
        InvT52 = 0
        InvT53 = -p2/(np.sqrt(p1p2))
        InvT54 = p1/(np.sqrt(p1p2))
        InvT55 = p2*(p1*q1tilde+p2*q2tilde)/(p1p2**(3/2))
        InvT56 = -p1*(p1*q1tilde+p2*q2tilde)/(p1p2**(3/2))
        
        InvT61 = 0
        InvT62 = 0
        InvT63 = 0
        InvT64 = 0
        InvT65 = -p2/(p1p2)
        InvT66 = p1/(p1p2)
    
        InvT = np.matrix([[InvT11,InvT12,InvT13,InvT14,InvT15,InvT16],[InvT21,InvT22,InvT23,InvT24,InvT25,InvT26],[InvT31,InvT32,InvT33,InvT34,InvT35,InvT36],[InvT41,InvT42,InvT43,InvT44,InvT45,InvT46],[InvT51,InvT52,InvT53,InvT54,InvT55,InvT56],[InvT61,InvT62,InvT63,InvT64,InvT65,InvT66]])

        InvTA11 = (1/(R*Hamiltonian))*((mu/R)*(3*a-2*R)-a*(2*Vr**2+3*Vt**2))
        InvTA12 = -a*Vr/Hamiltonian
        InvTA13 = -(Vr/Hamiltonian)*((Vt/p)*(2*a-R)-(a/(R*Vt))*(Vr**2+2*Vt**2))
        InvTA14 = (R/Hamiltonian)*((Vt/p)*(2*a-R)-(a/(R*Vt))*(Vr**2+2*Vt**2))
        InvTA15 = 0
        InvTA16 = 0
        
        InvTA21 = 0
        InvTA22 = 0
        InvTA23 = 1/R
        InvTA24 = 0
        InvTA25 = -((Vr*np.sin(argLat)+Vt*np.cos(argLat))/(R*Vt))*(np.sin(inc)/(1+np.cos(inc)))
        InvTA26 = (np.sin(argLat)/Vt)*(np.sin(inc)/(1+np.cos(inc)))
        
        InvTA31 = p*(np.cos(RAAN)*(2*Vr*np.sin(argLat)+3*Vt*np.cos(argLat)) + np.sin(RAAN)*(2*Vr*np.cos(argLat)-3*Vt*np.sin(argLat)))/(R**2*Vt)
        InvTA32 = np.sqrt(p/mu)*(np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))
        InvTA33 = (np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))*((1/R)-(Vr**2+Vt**2)/mu)-(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))*(Vr*Vt/mu)
        InvTA34 = 2*np.sqrt(p/mu)*(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))+(np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))*(R*Vr/mu)
        InvTA35 = ((q1*np.sin(RAAN)+q2*np.cos(RAAN))*(q1+np.cos(argLat))*np.sin(inc))/(p*(1+np.cos(inc)))
        InvTA36 = -((q1*np.sin(RAAN)+q2*np.cos(RAAN))*np.sin(argLat)*np.sin(inc))/(Vt*(1+np.cos(inc)))
                    
        InvTA41 = p*(np.sin(RAAN)*(2*Vr*np.sin(argLat)+3*Vt*np.cos(argLat)) - np.cos(RAAN)*(2*Vr*np.cos(argLat)-3*Vt*np.sin(argLat)))/(R**2*Vt)
        InvTA42 = -np.sqrt(p/mu)*(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))
        InvTA43 = -(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))*((1/R)-(Vr**2+Vt**2)/mu)-(np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))*(Vr*Vt/mu)
        InvTA44 = 2*np.sqrt(p/mu)*(np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))-(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))*(R*Vr/mu)
        InvTA45 = -((q1*np.cos(RAAN)-q2*np.sin(RAAN))*(q1+np.cos(argLat))*np.sin(inc))/(p*(1+np.cos(inc)))
        InvTA46 = ((q1*np.cos(RAAN)-q2*np.sin(RAAN))*np.sin(argLat)*np.sin(inc))/(Vt*(1+np.cos(inc)))

        InvTA51 = 0
        InvTA52 = 0
        InvTA53 = 0
        InvTA54 = 0
        InvTA55 = -(np.cos(RAAN)*(Vr*np.cos(argLat)-Vt*np.sin(argLat)) - np.sin(RAAN)*(Vr*np.sin(argLat)+Vt*np.cos(argLat)))/(R*Vt*(1+np.cos(inc)))
        InvTA56 = (np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))/(Vt*(1+np.cos(inc)))

        InvTA61 = 0
        InvTA62 = 0
        InvTA63 = 0
        InvTA64 = 0
        InvTA65 = -(np.sin(RAAN)*(Vr*np.cos(argLat)-Vt*np.sin(argLat))+ np.cos(RAAN)*(Vr*np.sin(argLat)+Vt*np.cos(argLat)))/(R*Vt*(1+np.cos(inc)))
        InvTA66 = (np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))/(Vt*(1+np.cos(inc)))

        InvTA = np.matrix([[InvTA11,InvTA12,InvTA13,InvTA14,InvTA15,InvTA16],[InvTA21,InvTA22,InvTA23,InvTA24,InvTA25,InvTA26],[InvTA31,InvTA32,InvTA33,InvTA34,InvTA35,InvTA36],[InvTA41,InvTA42,InvTA43,InvTA44,InvTA45,InvTA46],[InvTA51,InvTA52,InvTA53,InvTA54,InvTA55,InvTA56],[InvTA61,InvTA62,InvTA63,InvTA64,InvTA65,InvTA66]])
      
        InvTD11 = 0
        InvTD12 = 0
        InvTD13 = 0
        InvTD14 = 0
        InvTD15 = (np.sin(inc)*np.cos(inc)*np.sin(argLat)/(Hamiltonian*p*R**2))*((mu/R)*(2*a-R)-a*(Vr**2+2*Vt**2))
        InvTD16 = 0

        InvTD21 = 0
        InvTD22 = 0
        InvTD23 = -np.cos(inc)*(1-np.cos(inc))*np.sin(argLat)**2/(p*R**2)
        InvTD24 = 0
        InvTD25 = 0
        InvTD26 = 0

        InvTD31 = 0
        InvTD32 = 0
        InvTD33 = (np.cos(inc)**2*np.sin(argLat)**2/(Vt**2*R**3))*(Vr*Vt*(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))-Vt**2*(np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))) + (np.cos(inc)*np.sin(argLat)**2/(p*R**2))*(np.cos(inc)*(np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))+(q1*np.sin(RAAN)+q2*np.cos(RAAN)))
        InvTD34 = 0
        InvTD35 = (np.sin(inc)*np.cos(inc)*np.sin(argLat)/(Vt*R**3))*(Vr*(np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))+2*Vt*(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat)))
        InvTD36 = 0

        InvTD41 = 0
        InvTD42 = 0
        InvTD43 = (np.cos(inc)**2*np.sin(argLat)**2/(Vt**2*R**3))*(Vr*Vt*(np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))+Vt**2*(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))) - (np.cos(inc)*np.sin(argLat)**2/(p*R**2))*(np.cos(inc)*(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))+(q1*np.cos(RAAN)-q2*np.sin(RAAN)))
        InvTD44 = 0
        InvTD45 = -(np.sin(inc)*np.cos(inc)*np.sin(argLat)/(Vt*R**3))*(Vr*(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))-2*Vt*(np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat)))
        InvTD46 = 0

        InvTD51 = 0
        InvTD52 = 0
        InvTD53 = -(np.sin(inc)*np.cos(inc)*np.sin(argLat)/((1+np.cos(inc))*p*R**2))*(np.cos(RAAN)*np.cos(argLat)-np.sin(RAAN)*np.sin(argLat))
        InvTD54 = 0
        InvTD55 = 0
        InvTD56 = 0

        InvTD61 = 0
        InvTD62 = 0
        InvTD63 = -(np.sin(inc)*np.cos(inc)*np.sin(argLat)/((1+np.cos(inc))*p*R**2))*(np.sin(RAAN)*np.cos(argLat)+np.cos(RAAN)*np.sin(argLat))
        InvTD64 = 0
        InvTD65 = 0
        InvTD66 = 0
        
        InvTD = np.matrix([[InvTD11,InvTD12,InvTD13,InvTD14,InvTD15,InvTD16],[InvTD21,InvTD22,InvTD23,InvTD24,InvTD25,InvTD26],[InvTD31,InvTD32,InvTD33,InvTD34,InvTD35,InvTD36],[InvTD41,InvTD42,InvTD43,InvTD44,InvTD45,InvTD46],[InvTD51,InvTD52,InvTD53,InvTD54,InvTD55,InvTD56],[InvTD61,InvTD62,InvTD63,InvTD64,InvTD65,InvTD66]])        
        
        SigmaInverse = InvT*(InvTA + gamma*InvTD)
        
        return SigmaInverse
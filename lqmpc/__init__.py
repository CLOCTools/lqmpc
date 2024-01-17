'''
Closed Loop Optogenetics
SIP Lab

Jake Miller and Kyle Johnsen

Georgia Institute of Technology

Created 11/02/2023

Linear Quadratic Model Predictive Control
'''

########## IMPORTS #####################################################################################################
import osqp
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import time
########################################################################################################################

class LQMPC:
    # Linear Quadratic Model Predictive Control

    def __init__(self, t_step, A, B, C=None, D=None):
        # Initialize the LQMPC object with the system dynamics
        '''
        Parameters----- 
            t_step : System time step (s)
            A : System matrix (n x n)
            B : Control matrix (n x m)
            C : Output matrix (o x n)
            D : Feed-forward matrix (o x m)
        '''
        self.t_step = t_step
        self.A = A
        self.B = B
        self.n, self.m = self.B.shape
        if type(C) == type(None):
            C = np.eye(self.n)
        if type(D) == type(None):
            D = np.zeros((C.shape[0],self.m))
        self.C = C
        self.D = D
        self.o = self.C.shape[0] 

        if  self.A.shape[0] != self.A.shape[1]:                                  
            raise ValueError('The matrix A must be square.')
        if self.B.shape[0] != self.A.shape[1]:
            raise ValueError('The matrices A and B are of incompatible size.')
        if self.C.shape[1] != self.A.shape[0]:
            raise ValueError('The matrices A and C are of incompatible size.')
        if self.D.shape != (self.o,self.m):
            raise ValueError('The matrix D must be of size (o x m).')

        self.t = []                                                             # Time 
        self.y = []                                                             # Output
        self.yr = []                                                            # Target output
        self.u = []                                                             # Input
        self.J = []                                                             # Cost

        self.xi = None                                                          # Current state
        self.ui = None                                                          # Current input
    

    def set_control(self, Q=None, R=None, S=None, N=25, M=20):
        # Initialize the control parameters
        '''
        Parameters----- 
            Q : State penalty matrix (n x n)
            R : Input penalty matrix (m x m)
            S : Differential input penalty matrix (m x m)
            N : Prediction horizon
            M : Control horizon
        '''
        # Default values
        if type(Q) == type(None):
            Q = self.C.T@self.C                                                 
        if type(R) == type(None):
            R = sparse.eye(self.m)                                              
        if type(S) == type(None):
            S = sparse.eye(self.m)*0                                            

        # Input checks
        if  Q.shape != self.A.shape:                                            
            raise ValueError('Q must match the dimensions of matrix A.')
        if R.shape != (self.m,self.m):
            raise ValueError('R must be of dimensions m x m.')
        if S.shape != (self.m,self.m):
            raise ValueError('S must be of dimensions m x m.')
        if type(N) != int:
            raise TypeError('N must be a positive integer.')
        if type(M) != int:
            raise TypeError('M must be a positive integer.')
        if M >= N:
            raise ValueError('M must be strictly less than N.')
        
        self.Q = Q
        self.R = R
        self.S = S
        self.N = N
        self.M = M
        
        # Convert penalties into one matrix for OSQP
        Px = sparse.kron(sparse.eye(N),Q)                                       # State penalties
        Pu1 = sparse.kron(sparse.eye(M),2*S+R)                                  # Input penalties
        Pu2 = sparse.kron(sparse.eye(M,k=-1)+sparse.eye(M,k=1),-S)              # Differential input penalties
        Pu3 = sparse.block_diag([sparse.eye((M-1)*self.m)*0,-S])                # Correction to last differential input
        Pu = Pu1 + Pu2 + Pu3                                                    # All input penalties
        P = 2*sparse.block_diag([Px, Pu])                                       # Quadratic penalty matrix
        self.P = P


    def set_constraints(self, xmin=None, xmax=None, umin=None, umax=None):
        # Initialize the boundary conditions
        '''
        Parameters-----
            xmin : Minimum state value
            xmax : Maximum state value
            umin : Minimum input value
            umax : Maximum input value
        '''
        # Default values
        if type(xmin) == type(None):                                           
            xmin = np.ones(self.n)*np.inf*(-1)
        if type(xmax) == type(None):
            xmax = np.ones(self.n)*np.inf
        if type(umin) == type(None):
            umin = np.ones(self.m)*np.inf*(-1)
        if type(umax) == type(None):
            umax = np.ones(self.m)*np.inf    

        # Input checks
        if xmin.shape[0] != self.n or xmax.shape[0] != self.n:                  
            raise ValueError('Inputs xmin and xmax must be of size n.')
        if umin.shape[0] != self.m or umax.shape[0] != self.m:
            raise ValueError('Inputs umin and umax must be of size m.')
        if xmin.ndim != 1 or xmax.ndim != 1 or umin.ndim != 1 or umax.ndim != 1:
            raise ValueError('Constraints must be one-dimensional arrays.')
        
        self.xmin = xmin
        self.xmax = xmax
        self.umin = umin
        self.umax = umax

        # Convert bounds into form for OSQP
        lineq = np.hstack([np.kron(np.ones(self.N),xmin),                       # Lower inequality bound
                           np.kron(np.ones(self.M),umin)])  
        uineq = np.hstack([np.kron(np.ones(self.N),xmax),                       # Upper inequality bound
                           np.kron(np.ones(self.M),umax)])  
        Aineq = sparse.eye(self.N*self.n+self.M*self.m)                         # Inequality condition
        self.lineq = lineq
        self.uineq = uineq      
        self.Aineq = Aineq


    def __fast_update(self, x0, u0, xr):
        # Method to calculate the trajectory for the simulation step
        # Used when the simulation time step is the same as the prior step
        
        self.lb[:self.n] = -x0
        self.ub[:self.n] = -x0

        # Convert state penalty from reference to OSQP format 
        Qxr = -2*self.Q@xr[:,:self.N*self.n_sim:self.n_sim]                     # Qxr for every simulation time step
        Qxr = Qxr.flatten('F')
        qu = np.hstack([np.reshape(-2*self.S@u0,self.m),
                        np.zeros((self.M-1)*self.m)])
        qx = Qxr[:self.N*self.n]
        q = np.hstack([qx, qu])                                                 # Linear penalty term

        prob = osqp.OSQP()              
        prob.setup(self.P,q,self.Acon,self.lb,self.ub,                          # Set up OSQP solver
                   verbose=False,warm_start=False)     
        result = prob.solve()                                                   # Optimize
        if result.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')
        return result


    def __slow_update(self, x0, u0, xr):
        # Method to calculate the trajectory for the simulation step

        leq = np.hstack([-x0, np.zeros((self.N-1)*self.n)])                     # Lower equality bound             
        ueq = leq                                                               # Force equality with upper bound

        # Update x over n_sim many steps
        Axs = np.linalg.matrix_power(self.A,self.n_sim-1)                       # State multiplier            
        Aus = 0                                                                 # Input multiplier
        for i in range(self.n_sim-1): 
            Aus += np.linalg.matrix_power(self.A,i)              

        # Ax + Bu = 0
        Ax = sparse.kron(sparse.eye(self.N),-sparse.eye(self.n))+sparse.kron(sparse.eye(self.N,k=-1),Axs)
        B0 = sparse.csc_matrix((1,self.M))
        Bstep = sparse.eye(self.M)
        Bend = sparse.hstack([sparse.csc_matrix((self.N-self.M-1,self.M-1)),
                              np.ones((self.N-self.M-1,1))])
        Bu = sparse.kron(sparse.vstack([B0,Bstep,Bend]),Aus@self.B)
        Aeq = sparse.hstack([Ax, Bu])                                           # Equality condition

        Acon = sparse.vstack([Aeq, self.Aineq],format='csc')                    # Update condition             
        lb = np.hstack([leq, self.lineq])                                       # Lower bound
        ub = np.hstack([ueq, self.uineq])                                       # Upper bound
        self.Acon = Acon
        self.lb = lb
        self.ub = ub

        # Convert state penalty from reference to OSQP format 
        Qxr = -2*self.Q@xr[:,:self.N*self.n_sim:self.n_sim]                     # Qxr for every simulation time step
        Qxr = Qxr.flatten('F')
        qu = np.hstack([np.reshape(-2*self.S@u0,self.m),
                        np.zeros((self.M-1)*self.m)])
        qx = Qxr[:self.N*self.n]
        q = np.hstack([qx, qu])                                                 # Linear penalty term

        prob = osqp.OSQP()              
        prob.setup(self.P, q, Acon, lb, ub, verbose=False, warm_start=False)    # Set up OSQP solver
        result = prob.solve()                                                   # Optimize
        if result.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')
        return result


    def step(self, t_sim, x0, u0, xr, out=True):      
        # Perform one simulation step of control
        '''
        Parameters-----
            t_sim : Simulation time step (s)
            x0 : Initial state (n)
            u0 : Initial input (m)
            xr : Reference/Target state (n x N*n_sim)
            out : Print out step information
        Returns-----
            result : Optimal trajectories from OSQP
        '''     
        t1 = time.time()                                                        # Start time

        n_sim = int(t_sim/self.t_step)                                          # Number of points per simulation step
        self.n_sim = n_sim

        # Input checks
        if x0.shape[0] != self.n or x0.ndim != 1:
            raise ValueError('Array x0 must be one-dimensional with size n.')
        if u0.shape[0] != self.m or u0.ndim != 1:
            raise ValueError('Array u0 must be one-dimensional with size m.')
        if xr.shape[0] != self.n or xr.shape[1] < self.N*self.n_sim:
            raise ValueError('Step reference must have n-many rows and at least N*n_sim many columns.')
        if t_sim < self.t_step:
            raise ValueError('Simulation time step must be equal to or greater than system time step.')

        if (self.xi == x0).all() and self.t_sim == t_sim:                       # Employ fast update when possible
            result = self.__fast_update(x0, u0, xr)
        else:
            result = self.__slow_update(x0, u0, xr)
        self.t_sim = t_sim

        xi = x0
        ui = result.x[self.N*self.n:self.N*self.n+self.m]                       # Constant input over simulation step
        for i in range(self.n_sim):
            if self.t == []:
                self.t.append(0)
                self.u.append(u0)                                      
            else:
                self.t.append(self.t[-1]+self.t_step)                           # Save time value
                self.u.append(ui)                                               # Save input value
            yr = (self.C+self.D@np.linalg.pinv(self.B)@(np.eye(self.n)-self.A))@xr[:,i]           
            self.yr.append(yr)                                                  # Save target output value
            self.y.append(self.C@xi+self.D@ui)                                  # Save output value
            self.J.append(result.info.obj_val)                                  # Save cost function value
            xi = self.A@xi + self.B@ui                                          # Update current state                  
        self.xi = xi
        self.ui = ui

        t2 = time.time()                                                        # End time
        if out:
            print('Step Time:',str(t2-t1),'s')
        return result


    def simulate(self, t_sim, x0, u0, xr, L=100, T=None, out=True):
        # Run a full simulation loop
        '''
        Parameters-----
            t_sim : Simulation time step (s) as number or array of length L
            x0 : Initial state
            u0 : Initial input
            xr : Reference/Target state defined for (L+N)*n_sim points
            L : Number of simulation steps
            T : Simulation duration
            out : Print out simulation information
        Returns-----
            t : Array of time values
            y : Array of outputs
            yr : Array of target/reference outputs
            u : Array of inputs
            J : Array of cost function values
        '''
        t1 = time.time()                                                        # Start time
        if T != None:                                                           # Define either L or T
            L = int(T/t_sim)     
        if type(t_sim) == float or type(t_sim) == int:                          # Define either value or array
            t_sim = np.ones(L)*t_sim
        n_sim = int(t_sim[0]/self.t_step)                                       # Number of points per simulation step
        self.n_sim = n_sim      

        # Input checks
        if xr.shape[0] != self.n:
            raise ValueError('Simulation reference must have n-many rows.')
        if xr.shape[1] < self.n_sim*(L+self.N):
            raise ValueError('Simulation reference must have at least (L+N)*n_sim many columns.')

        for i in range(L):                                                      # Simulation loop
            if i == 0:
                self.step(t_sim[i], x0, u0,
                          xr[:,i*self.n_sim:(i+self.N)*self.n_sim], out=False)
            else:
                self.step(t_sim[i],self.xi,self.ui,
                          xr[:,i*self.n_sim:(i+self.N)*self.n_sim], out=False)            

        t2 = time.time()                                                        # End time
        if out:
            print('Simulation Time:',round(t2-t1,8),'s','\n'+
                  'Mean Step Time:',round((t2-t1)/L,8),'s','\n')  
        return self.t, self.y, self.yr, self.u, self.J


    def reset(self):
        # Basic function to clear saved variables from prior simulations
        self.t = []                                                             # Time 
        self.y = []                                                             # Output
        self.yr = []                                                            # Target output
        self.u = []                                                             # Input
        self.J = []                                                             # Cost
        self.xi = None                                                          # Current state
        self.ui = None                                                          # Current input


    def plot(self, f=None):
        # Basic function to plot output, input, and cost
        '''
        Parameters-----
            f : Optional function to transform output
        '''
        z = self.y
        zr = self.yr
        if f != None:                                                           # Transform output
            z = f(self.y)
            zr = f(self.yr)

        T = self.t[-1]
        f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        f.set_size_inches(8,6)
        ax1.plot(self.t,z)
        ax1.plot(self.t,zr,'--r')
        ax1.set_ylabel('Output')
        ax1.set_xlim([0, T])
        ax2.plot(self.t,self.u)
        ax2.set_ylabel('Input')
        ax2.set_xlim([0, T])
        ax3.plot(self.t,self.J)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Cost')
        ax3.set_xlim([0, T])
        plt.show()

########################################################################################################################
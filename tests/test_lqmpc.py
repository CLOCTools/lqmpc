'''
Closed Loop Optogenetics
SIP Lab

Jake Miller and Kyle Johnsen

Georgia Institute of Technology

Created 12/18/2023

LQMPC - Tests
'''

########## IMPORTS #####################################################################################################
import numpy as np
from scipy import sparse
import lqmpc
import pytest
########################################################################################################################

def test_init():
    A = np.ones((4,4))
    B = np.ones((4,2))
    mpc = lqmpc.LQMPC(0.001,A,B)
    assert type(mpc.C) != type(None)
    assert type(mpc.D) != type(None)
    assert mpc.xi == None
    assert mpc.ui == None

def test_init_raises():
    A = np.ones((4,4))
    B = np.ones((4,2))
    C = np.ones((3,4))
    D = np.ones((3,2))
    test_A = np.ones((4,5))
    test_B = np.ones((3,2))
    test_C = np.ones((7,3))
    test_D = np.ones((2,4))
    with pytest.raises(TypeError):
        lqmpc.LQMPC()                                                           # Must define inputs
    with pytest.raises(ValueError): 
        lqmpc.LQMPC(0.001,test_A,B,C,D)                                         # Incorrect A shape
    with pytest.raises(ValueError):
        lqmpc.LQMPC(0.001,A,test_B,C,D)                                         # Incorrect B shape
    with pytest.raises(ValueError):
        lqmpc.LQMPC(0.001,A,B,test_C,D)                                         # Incorrect C shape
    with pytest.raises(ValueError):
        lqmpc.LQMPC(0.001,A,B,C,test_D)                                         # Incorrect D shape

def test_control():
    mpc = lqmpc.LQMPC(0.001,np.ones((2,2)),np.ones((2,1)))
    mpc.set_control()
    assert type(mpc.Q) != type(None)
    assert type(mpc.R) != type(None)
    assert type(mpc.S) != type(None)
    assert type(mpc.P) != type(None)
    mpc.set_control(Q=3*np.eye(2),R=np.eye(1),S=5*np.eye(1),N=3,M=2)
    Px = 6*np.eye(2*3)
    Pu = np.array([[22,-10],[-10,12]])
    P = sparse.block_diag([Px, Pu])
    assert (mpc.P.A == P.A).all()

def test_control_raises():
    mpc = lqmpc.LQMPC(0.001,np.ones((3,3)),np.ones((3,2)))
    with pytest.raises(ValueError):
        mpc.set_control(Q=np.ones((4,3)))                                       # Incorrect Q shape
    with pytest.raises(ValueError):
        mpc.set_control(R=np.ones((3,2)))                                       # Incorrect R shape
    with pytest.raises(ValueError):
        mpc.set_control(S=np.ones((3,2)))                                       # Incorrect S shape
    with pytest.raises(TypeError):
        mpc.set_control(N=25.0)                                                 # Float instead of integer
    with pytest.raises(TypeError):
        mpc.set_control(M=0.2)                                                  # Float instead of integer
    with pytest.raises(ValueError):
        mpc.set_control(N=10,M=10)                                              # Incorrect: N = M
    with pytest.raises(ValueError):
        mpc.set_control(N=10,M=11)                                              # Incorrect: N < M

def test_constraints():
    mpc = lqmpc.LQMPC(0.001,np.ones((3,3)),np.ones((3,2)))
    with pytest.raises(AttributeError):
        mpc.set_constraints()                                                   # Define prerequisite parameters
    mpc.set_control()
    mpc.set_constraints()
    assert type(mpc.xmin) != type(None)
    assert type(mpc.xmax) != type(None)
    assert type(mpc.umin) != type(None)
    assert type(mpc.umax) != type(None)
    assert type(mpc.lineq) != type(None)
    assert type(mpc.uineq) != type(None)
    assert type(mpc.Aineq) != type(None)

def test_constraints_raises():
    mpc = lqmpc.LQMPC(0.001,np.ones((3,3)),np.ones((3,2)))
    mpc.set_control()
    with pytest.raises(ValueError):
        mpc.set_constraints(xmin=np.ones(4))                                    # Incorrect xmin size
    with pytest.raises(ValueError):
        mpc.set_constraints(xmax=np.ones(4))                                    # Incorrect xmax size
    with pytest.raises(ValueError):
        mpc.set_constraints(umin=np.ones(3))                                    # Incorrect umin size
    with pytest.raises(ValueError):
        mpc.set_constraints(umax=np.ones(3))                                    # Incorrect umax size
    with pytest.raises(ValueError):
        mpc.set_constraints(xmin=np.ones((3,1)))                                # Incorrect xmin shape
    with pytest.raises(ValueError):
        mpc.set_constraints(xmax=np.ones((3,1)))                                # Incorrect xmax shape
    with pytest.raises(ValueError):
        mpc.set_constraints(umin=np.ones((2,1)))                                # Incorrect umin shape
    with pytest.raises(ValueError):
        mpc.set_constraints(umax=np.ones((2,1)))                                # Incorrect umax shape

def test_step_inputs():
    mpc = lqmpc.LQMPC(0.001,np.ones((3,3)),np.ones((3,2)))
    with pytest.raises(AttributeError):
        mpc.step(0.002,np.zeros(3),np.zeros(2),np.ones((3,6)),out=False)        # Define prerequisite parameters
    mpc.set_control(N=3,M=2)
    mpc.set_constraints()
    with pytest.raises(ValueError):
        mpc.step(0.002,np.zeros(4),np.zeros(2),np.ones((3,6)),out=False)        # Incorrect length of x0
    with pytest.raises(ValueError):
        mpc.step(0.002,np.zeros(3),np.zeros(1),np.ones((3,6)),out=False)        # Incorrect length of u0
    with pytest.raises(ValueError):
        mpc.step(0.002,np.zeros((3,1)),np.zeros(2),np.ones((3,6)),out=False)    # Incorrect shape of x0
    with pytest.raises(ValueError):
        mpc.step(0.002,np.zeros(3),np.zeros((2,1)),np.ones((3,6)),out=False)    # Incorrect shape of u0
    with pytest.raises(ValueError):
        mpc.step(0.002,np.zeros(3),np.zeros(2),np.ones((3,5)),out=False)        # Incorrect column length of xr 
    with pytest.raises(ValueError):
        mpc.step(0.002,np.zeros(3),np.zeros(2),np.ones((2,6)),out=False)        # Incorrect row length of xr
    with pytest.raises(ValueError):
        mpc.step(0.0001,np.zeros(3),np.zeros(2),np.ones((3,6)),out=False)       # Simulation time step too small
        
def test_step():
    mpc = lqmpc.LQMPC(1,3*np.ones((1,1)),5*np.ones((1,2)))
    mpc.set_control(N=4,M=2)
    mpc.set_constraints(np.array([-11]),np.array([11]),
                        np.array([0,1]),np.array([19,21]))
    test_Acon = np.array([[-1, 0, 0, 0, 0, 0, 0, 0],
                        [9, -1, 0, 0, 20, 20, 0, 0],
                        [0, 9, -1, 0, 0, 0, 20, 20],
                        [0, 0, 9, -1, 0, 0, 20, 20],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]])
    test_lb = np.array([-7,0,0,0,-11,-11,-11,-11,0,1,0,1])
    test_ub = np.array([-7,0,0,0,11,11,11,11,19,21,19,21])
    with pytest.raises(ValueError):
        mpc.step(3,7*np.ones(1),np.zeros(2),np.ones((1,12)))                    # Error when OSQP does not solve
    assert (mpc.Acon.A == test_Acon).all()
    assert (mpc.lb == test_lb).all()
    assert (mpc.ub == test_ub).all()

def test_simulate():
    A = np.array([[1, -6.66e-13, -2.03e-9, -4.14e-6],                              
              [9.83e-4, 1, -4.09e-8, -8.32e-5],
              [4.83e-7, 9.83e-4, 1, -5.34e-4],
              [1.58e-10, 4.83e-7, 9.83e-4, .9994]])
    B = np.array([[9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]]).T                      
    C = np.array([[-.0096, .0135, .005, -.0095]])           
    def z_to_y(z):
        return (np.log(z)+5.468)/61.4             
    yr = z_to_y(0.2)
    yr = np.ones((1,(100+25)*250))*yr
    I = np.eye(4)
    xr = np.linalg.inv(I-A)@B@(np.linalg.pinv(C@np.linalg.inv(I-A)@B)@yr)
    xr_test1 = xr[0:3,:]
    xr_test2 = xr[:,0:(100+25)*250-1]
    mpc = lqmpc.LQMPC(0.001,A,B,C)
    mpc.set_control()
    mpc.set_constraints()
    with pytest.raises(ValueError):
        mpc.simulate(0.25,np.zeros(4),np.zeros(1),xr_test1)                     # Incorrect number of rows
    with pytest.raises(ValueError):
        mpc.simulate(0.25,np.zeros(4),np.zeros(1),xr_test2)                     # Incorrect number of columns                                  

def test_reset():
    A = np.array([[1, -6.66e-13, -2.03e-9, -4.14e-6],                              
              [9.83e-4, 1, -4.09e-8, -8.32e-5],
              [4.83e-7, 9.83e-4, 1, -5.34e-4],
              [1.58e-10, 4.83e-7, 9.83e-4, .9994]])
    B = np.array([[9.83e-4, 4.83e-7, 1.58e-10, 3.89e-14]]).T                      
    C = np.array([[-.0096, .0135, .005, -.0095]])           
    def z_to_y(z):
        return (np.log(z)+5.468)/61.4             
    yr = z_to_y(0.2)
    yr = np.ones((1,(100+25)*250))*yr
    I = np.eye(4)
    xr = np.linalg.inv(I-A)@B@(np.linalg.pinv(C@np.linalg.inv(I-A)@B)@yr)

    mpc = lqmpc.LQMPC(0.001,A,B,C)
    mpc.set_control()
    mpc.set_constraints()
    mpc.simulate(0.25,np.zeros(4),np.zeros(1),xr)
    assert len(mpc.t) == 100*250
    mpc.simulate(0.25,np.zeros(4),np.zeros(1),xr)
    assert len(mpc.t) == 100*250*2                                              # Without clearing prior simulation
    mpc.reset()                                                                 
    mpc.simulate(0.25,np.zeros(4),np.zeros(1),xr)
    assert len(mpc.t) == 100*250                                                # With clearing prior simulation
    mpc.step(0.1,np.zeros(4),np.zeros(1),xr)
    assert len(mpc.t) == 100*250 + 1*100                                        # With one step
    mpc.step(0.01,np.zeros(4),np.zeros(1),xr)
    assert len(mpc.t) == 100*250 + 1*100 + 1*10                                 # With another step
    mpc.reset()
    mpc.step(0.20,np.zeros(4),np.zeros(1),xr)
    assert len(mpc.t) == 1*200                                                  # Step after reset
    mpc.simulate(0.25,np.zeros(4),np.zeros(1),xr)
    assert len(mpc.t) == 1*200 + 100*250                                        # Step then simulate
    mpc.reset()
    assert len(mpc.t) == 0                                                      # After reset
    mpc.step(0.20,np.zeros(4),np.zeros(1),xr)
    mpc.reset()
    mpc.step(0.20,np.zeros(4),np.zeros(1),xr)                                   # Use slow step even with same t_sim

import numpy as np
from context import magnetorquer_detumble

skew = magnetorquer_detumble.practical.skew
PC = magnetorquer_detumble.practical.PracticalController


#quaternion stuff
#Left muliply quaternion matrix    
def L(q): 
    #scalar part of quat
    qs = q[0]

    #vector part of quat (dimension 1x3)
    qv = np.expand_dims(q[1:], axis=0)
    
    L = np.block([
    [q[0], -qv], 
    [qv.T, q[0]*np.identity(3)+hat(q[1:])]            
    ])
    
    return L

#Right multiply quaternion matrix
#R_ to not overload with the measurement noise R
def R(q): 

    #scalar part of quat
    qs = q[0]

    #vector part of quat (dimension 1x3)
    qv = np.expand_dims(q[1:], axis=0)
    
    R = np.block([
    [q[0], -qv], 
    [qv.T, q[0]*np.identity(3)-hat(q[1:])]            
    ])
    
    return R

#operator that converts a 3 parameter vector to a quaterion w zero real part
H = np.vstack((np.zeros((1,3)), np.identity(3)))

#Conjugate of a quaternion. Negate the vector part

def conj_q(q):

    invq = np.hstack((q[0], -1*q[1:]))

    return invq

def G(q): 

    G = L(q) @ h

    return G

#Quaternion to Rotation Matrix
def quaternion_to_rotmatrix(q):

    return H.T@L(q)@R(q).T@H

#assume B lives on a sinusioud
def B(t): 
    Bmag = 50e-6
    f = 1/(90/60)
    return (Bmag/sqrt(2))*np.array([np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t), 1])


def Bdot(t):
    
    Bmag = 50e-6
    f = 1/(90/60)
    return (Bmag/sqrt(2))*2*np.pi*np.array([np.cos(2*np.pi*f*t), -np.sin(2*np.pi*f*t), 0])


#TODO: FINISH
def dynamics(t,x): 

    #quaternion
    q = x[0:4]

    q = q/np.linalg.norm(q)

    return x_dot


#test one step of the get control function to check if the logic in practical.py is correct
mag_data = np.array([1.0, 2.0, 3.0])
gyro_data = np.array([-0.1, 0.1, 0.1])
sun_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

controller = PC(np.array([10.0, 11.0, 12.0]), np.array([0.5, 0.6, 0.3]), mag_data, gyro_data, sun_data, 6*np.pi, which_controller=1)

control = controller.get_control(0.2)

print(control)
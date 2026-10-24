"""
General file for useful methods from Computational Physics with Python
from Eric Ayars, Caltech
"""

def euler(y, t, dt, derivs):
    """
    A routine that implements Euler's method to find the new state of y

    Parameters
    -----    
    y : vector, current state
    t : time
    dt : time step
    derivs : function that returns the derivatives of vector y, defines the system

    Returns
    ----- 
    y_next : vector, next state

    """

    y_next = y + derivs(y,t) * dt
    return y_next
    

def rk2(y, time, dt, derivs):
    """
    Moves the value of 'y' forward by a step of size 'dt' using a second-order Runge-Kutta algorithm.
    Equivalent to finding the average of the slope at time t and at time (t+dt), and using the average 
    slope to find the new value of y.

    Parameters
    -----    

    y : vector, current state
    t : time
    dt : time step
    derivs : function that returns the derivatives of vector y, defines the system

    Returns
    ----- 
    y_next : vector, next state

    """

    k0 = dt * derivs(y, time)
    k1 = dt * derivs(y + k0, time + dt)

    y_next = y + 0.5 * (k0 + k1) 

    return y_next



def rk4(y, time, dt, derivs):
    """
    Moves the value of 'y' forward by a step of size 'dt' using a fourth-order Runge-Kutta algorithm.
    Equivalent to finding the average of the slope at time t and at time (t+dt), and using the average 
    slope to find the new value of y.

    Parameters
    -----    
    y : vector, current state
    t : time
    dt : time step
    derivs : function that returns the derivatives of vector y, defines the system

    Returns 
    ----- 
    y_next : vector, next state

    """

    k1 = dt * derivs(y, time)
    k2 = dt * derivs(y + 0.5*k1, time + 0.5*dt)
    k3 = dt * derivs(y + 0.5*k2, time + 0.5*dt)
    k4 = dt * derivs(y + k3, time + dt)

    y_next = y + (1/6)*(k1 + 2*k2 + 2*k3 + k4) 
    
    return y_next
 

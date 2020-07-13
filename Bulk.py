import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import simps
from scipy.signal import savgol_filter
import pandas as pd
from decimal import *
from scipy.integrate import solve_bvp

def Bulk_Scalar(H, V, r0, LamB=0.0, k: float = 1, m: float = 0.2, visualise: bool = True):
    c=np.sqrt(4+(np.power(m,2)/np.power(k,2)))
    E=np.sqrt(4+(np.power(m,2)/np.power(k,2)))-2
    
    #the EL equation of motion that must be solved (Numerically)    
    def func2(x, U):
        # Here U is a vector such that phi=U[0] and phi'=U[1]. This function should return [phi', phi'']
        return np.vstack((U[1], 4*k*U[1] + m*m*U[0]+LamB*U[0]*U[0]*U[0]))
    
    def New_solve_bvp(Func, r0):
    
        def bc(ya, yb):
            return np.array([ya[0]-H, yb[0]-V])

        xs = np.linspace(0, r0, round((2**10)*r0*r0))
        y_a = np.zeros((2, xs.size))
        y_a[1]=1
        y_b = np.zeros((2, xs.size))

        #xs2 = np.linspace(0, r0, (2**12)*r0)

        res_a = solve_bvp(Func, bc, xs, y_a, tol=1e-9)
        y0 = res_a.sol(xs)[0]
        d0 = res_a.sol(xs)[1]

        return([xs, y0], [xs, d0])

    
    phi, d_phi = New_solve_bvp(func2, r0)  
    ys=phi[1]; ds=d_phi[1]
    xs=phi[0]
    
    # BStiff =(H*(np.exp((2+c)*k*r0))-V)/(np.exp((2+c)*k*r0) - np.exp((2-c)*k*r0))
    # AStiff=H-BStiff

    AStiff = ((V * np.exp((c-2)*k*r0)) - H)/(np.exp(2*c*k*r0)-1)

    BStiff = H-AStiff


    def Phi(A0, B0, r):
        return(np.exp(2*k*r)*((A0*np.exp(c*k*r))+(B0*np.exp(-c*k*r))))
    
    
    
    if visualise:
        plt.plot(xs, ys, label = "With Coupling")
        plt.plot(xs, Phi(AStiff, BStiff, xs), label = "Analytic Limit")
        plt.legend()
        plt.show()
        
    return(phi, d_phi)

def Bulk_Potential(H: float, V: float, Ymin: float, Ymax: float, LamB=0.0, k: float = 1, m: float = 0.2, visualise: bool = True): 
    Potential=[]
    Y = []
    y = Ymin
    while y <= Ymax:
        
        phi, d_phi = Bulk_Scalar(H, V, y, LamB=LamB, k=k, m=m, visualise=False)  
        ys=phi[1]; ds=d_phi[1]
        xs=phi[0]

        #the values of the field and its derivative for a given r0 are plugged into the Lagrangian
        e=np.exp(-4*k*xs)
        L=e*((ds*ds)+(m*m*ys*ys)+(LamB*ys*ys*ys)) #our lagrangean
        Potential=np.append(Potential, simps(L, xs))
        Y=np.append(Y, y)

        y=y+0.01

        print(y, end="\r")

    if visualise:
        
        A = ((V * np.exp((c-2)*k*Y)) - H)/(np.exp(2*c*k*Y)-1)

        B = H-A
        
        
        Pot=(A*A*k*(2+c)*(np.exp(2*c*k*Y)-1) + k*(c-2)*B*B*(1-np.exp(-2*c*k*Y)))
        

        
        
        plt.plot(Y, Potential, label = "Potential")
        plt.plot(Y, Pot)
        #plt.plot(xs, Phi(AStiff, BStiff, xs), label = "Analytic Limit")
        plt.legend()
        plt.show()
        
        print("Minima,  ", "y: ", Y[Potential.argmin()], " V: ", Potential.min())
        print("Analytical Min, y:", Y[Pot.argmin()], "V:", Pot.min())


    return [Y, Potential]

# c=np.sqrt(4+(np.power(m,2)/np.power(k,2)))
# E=np.sqrt(4+(np.power(m,2)/np.power(k,2)))-2

# #the EL equation of motion that must be solved (Numerically)    
# def dU_dx(U, x):
#     # Here U is a vector such that phi=U[0] and phi'=U[1]. This function should return [phi', phi'']
#     return [U[1], 4*k*U[1] + m*m*U[0]+LamB*U[0]*U[0]*U[0]]

# def Phi(A0, B0, r):
    
#     return(np.exp(2*k*r)*((A0*np.exp(c*k*r))+(B0*np.exp(-c*k*r))))



# def V_lim(rc, V, H):
#     A=np.float64((V-H*np.exp((2-c)*k*rc))/(np.exp((2+c)*k*rc)-np.exp((2-c)*k*rc)))
#     B=np.float64(H-A)
#     V=np.float64((A*A*k*(2+c)*(np.exp(2*c*k*rc)-1) + k*(c-2)*B*B*(1-np.exp(-2*c*k*rc))))
#     return V

# def solve_bvp_HM(Func, r0):

#     BStiff=((V*(np.exp(-(2+c)*k*r0))-H)/((np.exp(-2*k*c*r0))-1))
#     AStiff=H-BStiff

#     #creates an array of 9999 points equally spaced from 0 to r0 (the position of the visible brane)   
#     xs = np.linspace(r0, 0, 9999)


#     it=0 #counts number of interations in th shooting method
#     abserr=50 #sets the "initial" error so the the while loop can functiond
#     DBH0=-10 #estimate for the initial gradient that the while loop will iteratively improve 
#     shoot_J=DBH0#sets the intial interval over which the shooting method varies
#     BH0=H#sets the intial condition of phi (unchanged by shooting method)

#     # the while loop will interate, improving upon the initial conditions until it has produced the 
#     # boundary condition the the visible boundary to the error as described in the while loop condition
#     while(abserr>1e-17):

#         U0 = [V, DBH0]
#         Us0 = odeint(Func, U0, xs)
#         y0 = Us0[:,0] 
#         d0 = Us0[:,1]

#         trunc=0

#         for i in range(0, y0.size):

#             if(((y0[i]>2*H) or (y0[i]<0)) and trunc==0):
#                 trunc=1
#                 cont=y0[i]
#             if(trunc==1):
#                 y0[i]=cont

#         #print(y0[0], y0[-1])        
#         BV1=y0[-1]#finds y value at 0

#         #finds the difference between the visible boundary condition, and the value of this iterations
#         #phi at the visible boundary. the initial conditions are then adjusted accordingly, by bisecting
#         #the interval until it the error on the visible boundary is small enough 
#         err=BV1-H
#         abserr=abs(err)
#         if(err>0):
#                 DBH0=DBH0-shoot_J
#         if(err<0):
#                 DBH0=DBH0+shoot_J
#         if(it>=500):
#                 break
#         shoot_J=shoot_J*0.5
#         it=it+1
#         print(DBH0, err)
        
#     return([xs, y0], [xs, d0])


# #the EL equation of motion that must be solved (Numerically)    
# def func2(x, U):
#     global LamB
#     # Here U is a vector such that phi=U[0] and phi'=U[1]. This function should return [phi', phi'']
#     return np.vstack((U[1], 4*k*U[1] + m*m*U[0]+LamB*U[0]*U[0]*U[0]))

# from scipy.integrate import solve_bvp

# def New_solve_bvp(Func, r0):
    
#     def bc(ya, yb):
#         return np.array([ya[0]-H, yb[0]-V])
    
#     xs = np.linspace(0, r0, round((2**10)*r0*r0))
#     y_a = np.zeros((2, xs.size))
#     y_a[1]=1
#     y_b = np.zeros((2, xs.size))
    
#     #xs2 = np.linspace(0, r0, (2**12)*r0)

#     res_a = solve_bvp(Func, bc, xs, y_a, tol=1e-8)
#     y0 = res_a.sol(xs)[0]
#     d0 = res_a.sol(xs)[1]
    
#     return([xs, y0], [xs, d0])

# def Phi(A0, B0, r):
#     return(np.exp(2*k*r)*((A0*np.exp(c*k*r))+(B0*np.exp(-c*k*r))))

# def NumV(Func, Ymin, Ylim): 
#     global LamB
#     Potential=[]
#     Y=[]
#     while(Ymin < Ylim):
        
#         phi, d_phi = New_solve_bvp(Func, Ymin)  
#         ys=phi[1]; ds=d_phi[1]
#         xs=phi[0]
#         print(Ymin)

#         #the values of the field and its derivative for a given r0 are plugged into the Lagrangian
#         e=np.exp(-4*k*xs)
#         L=e*((ds*ds)+(m*m*ys*ys)+(LamB*ys*ys*ys)) #our lagrangean
        
#         V=simps(L, xs)
#         Potential=np.append(Potential, V)
#         Y=np.append(Y, Ymin)
#         Ymin=Ymin+0.05
#     return [Y, Potential]


import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import simps
from scipy.signal import savgol_filter
import pandas as pd
from decimal import *
from scipy.integrate import solve_bvp

def Bulk_Scalar(H, V, r0, LamB=0.0, LamV=None, LamH=None, k: float = 1, m: float = 0.2, visualise: bool = True):
    c=np.sqrt(4+(np.power(m,2)/np.power(k,2)))
    E=np.sqrt(4+(np.power(m,2)/np.power(k,2)))-2
    
    def dU_dx(U, x, r0, k):
        # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
        return [U[1], 4*k*U[1] + m*m*U[0]+LamB*U[0]*U[0]*U[0]]

    def func2(x, U):
        # Here U is a vector such that phi=U[0] and phi'=U[1]. This function should return [phi', phi'']
        return np.vstack((U[1], 4*k*U[1] + m*m*U[0]+LamB*U[0]*U[0]*U[0]))
    
    def boundaries(phi, Lam, B):
        return(2*Lam*phi*((phi*phi)-(B*B)))
    
    def no_bc_bvp(Func, r0):
    
        def bc(ya, yb):

            #return np.array([ya[1]-(lamH*ya[0]*(ya[0]**2-H**2)), yb[1]-(lamH*yb[0]*(yb[0]**2-V**2))])
            return np.array([ya[0]-H, yb[0]-V])

        xs = np.linspace(0, r0, round((2**10)*r0*r0))
        y_a = np.zeros((2, xs.size))
        y_a[1]=-0.01396517
        y_b = np.zeros((2, xs.size))

        #xs2 = np.linspace(0, r0, (2**12)*r0)

        res_a = solve_bvp(Func, bc, xs, y_a, tol=1e-9)
        y0 = res_a.sol(xs)[0]
        d0 = res_a.sol(xs)[1]

        return([xs, y0], [xs, d0])

    def bc_bvp(Func, BCs, r0):
        
        #creates an array of 9999 points equally spaced from 0 to r0 (the position of the visible brane)   
        xs = np.linspace(r0, 0, 9999)


        it=0 #counts number of interations in th shooting method
        abserr=50 #sets the "initial" error so the the while loop can functiond
        
        BH0=V+0.1 #estimate for the initial boundary that the while loop will iteratively improve
        DBH0=BCs(BH0, LamV, V) #estimate for the initial gradient
        shoot_J=V#sets the intial interval over which the shooting method varies

        # the while loop will interate, improving upon the initial conditions until it has produced the 
        # boundary condition the the visible boundary to the error as described in the while loop condition
        while(abserr>1e-14):
            
            #print(BH0)

            DBH0=-BCs(BH0, LamV, V)
            U0 = [BH0, DBH0]
            Us0 = odeint(dU_dx, U0, xs, args=(r0, k))
            y0 = Us0[:,0] 
            d0 = Us0[:,1]
            
            #print(y0)

            trunc=0

            for i in range(0, y0.size):

                if(((y0[i]>2*H) or (y0[i]<0)) and trunc==0):
                    trunc=1
                    cont=y0[i]
                    cont2=d0[i]
                if(trunc==1):
                    y0[i]=cont
                    d0[i]=cont2
                    
            BV1=y0[-1]#finds y value at r0
            dsBC=BCs(BV1, LamH, H)
            DBV1=d0[-1]


            #for a given gradient, the following lines find the largest non imaginary solution for the boundaries
            temp=[]
            coeff=[-2*LamH, 0, 2*LamH*H*H, +DBV1]
            solutions=np.roots(coeff)
            for x in solutions:
                if (np.iscomplex(x)==False):
                    temp=np.append(temp, x)


            solution=np.real(np.amax(temp))


            err=solution-BV1
            abserr=abs(err)


            if(err>0):
                    BH0=BH0+shoot_J
            if(err<0):
                    BH0=BH0-shoot_J
            if(it>=500):
                    break
            shoot_J=shoot_J*0.5
            it=it+1

        #print("uncertainty on boundary: ", abserr)
        return([np.flip(xs), np.flip(y0)], [np.flip(xs), np.flip(d0)])

    if(LamV and LamH):
        phi, d_phi = bc_bvp(dU_dx, boundaries, r0)
    
    else:
        phi, d_phi = no_bc_bvp(func2, r0)


    ys=phi[1]; ds=d_phi[1]
    xs=phi[0]
    
    # BStiff =(H*(np.exp((2+c)*k*r0))-V)/(np.exp((2+c)*k*r0) - np.exp((2-c)*k*r0))
    # AStiff=H-BStiff

    AStiff = ((V * np.exp((c-2)*k*r0)) - H)/(np.exp(2*c*k*r0)-1)

    BStiff = H-AStiff


    def Phi(A0, B0, r):
        return(np.exp(2*k*r)*((A0*np.exp(c*k*r))+(B0*np.exp(-c*k*r))))
    
    
    
    if visualise:
        fig = plt.figure(figsize=(15,8))
        plt.plot(xs, ys, label = "With Coupling")
        plt.plot(xs, Phi(AStiff, BStiff, xs), label = "Analytic Limit")
        plt.xlabel("$y$", fontsize=13)
        plt.ylabel("$\Phi(y)$", fontsize=13)
        plt.legend()
        plt.show()
        
    return(phi, d_phi)



def Bulk_Potential(H: float, V: float, Ymin: float, Ymax: float, LamH=None, LamV=None, LamB=0.0, k: float = 1, m: float = 0.2, visualise: bool = True, value_returned=None): 
    c=np.sqrt(4+(np.power(m,2)/np.power(k,2)))

    if(Ymin==0):
        print("Please do not set Ymin to 0, as you can't intergrate over 0 distance!!!")

    E=np.sqrt(m**2)/(4*k**2)
    #E=np.sqrt(4+(np.power(m,2)/np.power(k,2)))-2
    Potential=[]
    Y = []
    y = Ymin
    while y <= Ymax:
        
        phi, d_phi = Bulk_Scalar(H, V, y, LamB=LamB, LamH=LamH, LamV=LamV, k=k, m=m, visualise=False)  
        ys=phi[1]; ds=d_phi[1]
        xs=phi[0]

        #the values of the field and its derivative for a given r0 are plugged into the Lagrangian
        e=np.exp(-4*k*xs)
        L=e*((ds*ds)+(m*m*ys*ys)+(LamB*ys*ys*ys)) #our lagrangean

        value = simps(L, xs)

        if(LamV and LamH):
            value = value + LamH*((ys[0]*ys[0]-H*H)*(ys[0]*ys[0]-H*H))+LamV*(np.exp(-4*k*y)*((ys[-1]*ys[-1]-V*V)*(ys[-1]*ys[-1]-V*V)))

        Potential=np.append(Potential, value)
        Y=np.append(Y, y)

        y=y+0.01

        print(y, end="\r")

    A = ((V * np.exp((c-2)*k*Y)) - H)/(np.exp(2*c*k*Y)-1)

    B = H-A
    
    Pot=(A*A*k*(2+c)*(np.exp(2*c*k*Y)-1) + k*(c-2)*B*B*(1-np.exp(-2*c*k*Y)))

    min_position_numerical=Potential.argmin()

    second_deriv_num=np.gradient(np.gradient(Potential, Y), Y)

    ym = Y[Potential.argmin()]
    yc = Y[Pot.argmin()]

    mass = second_deriv_num[min_position_numerical]*(np.exp(2*k*(yc+ym))/(8*k*k*k*E*E*V*V))

    if visualise:
        fig= plt.figure(figsize=(15,8))
        plt.plot(Y, Potential, label = "With Coupling")
        plt.plot(Y[1:], Pot[1:], label = "Analytic Limit")
        #print()
        plt.ylim(0, 2*Potential[0])
        #plt.plot(xs, Phi(AStiff, BStiff, xs), label = "Analytic Limit")
        plt.ylabel("$V_{GW}$", fontsize=13)
        plt.xlabel("Brane Sepration $y_0$", fontsize=13)
        plt.legend()
        plt.show()



        #mass1= d2Npot_dY2[NPotMinPos]*(np.exp(2*k*(Y[NPotMinPos]+Smpos))/(8*k*k*k*E*E*V*V))
        
        print("mr/m0: ", np.sqrt(mass))
        print("Minima,  ", "y: ", ym, " V: ", Potential.min())
        print("Analytical Min, y:", yc, "V:", Pot.min())
        print("V(0): ", Potential[0], " Analytical Value: ", ((LamH*LamV)/(LamH+LamV))*(V*V - H*H)*(V*V - H*H) )
    
    if(value_returned == "mass"):
        return mass
    if(value_returned == "depth"):
        return Potential[-1] - Potential.min()
    else:
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


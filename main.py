import numpy as np
import numpy.random 
import scipy.special
import random 
import math
import matplotlib.pyplot as plt
import Functions as F

X=F.random_2d_points_lam(10, 20, 10, 20, 2)


t_range=np.linspace(0, 2, 25)

Kt=[]
for t in t_range:
    Kt=np.append(Kt, F.K_function(t, X, [10, 10, 20, 20]))

Fr=[]
for t in t_range:
    Fr=np.append(Fr, F.F_function(t, X, [10, 10, 20, 20]))

plt.plot(t_range, Kt)
p=np.pi
plt.plot(t_range, 1*t_range*t_range*p )

plt.show()

plt.plot(t_range, Fr)
plt.plot(t_range, F.contact_dist_2d(1, t_range))

plt.show()

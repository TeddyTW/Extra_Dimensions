import numpy as np
import numpy.random 
import scipy.special
import random 
import math
import matplotlib.pyplot as plt
import Functions as F

X=F.random_2d_points_lam(5, 15, 5, 15, 1)


t_range=np.linspace(0, 1, 20)

Kt=[]
for t in t_range:
    Kt=np.append(Kt, F.K_function(t, X, [5, 5, 15, 15]))

Fr=[]
for t in t_range:
    Fr=np.append(Fr, F.F_function(t, X, [5, 5, 15, 15]))

plt.plot(t_range, Kt)
p=np.pi
plt.plot(t_range, 1*t_range*t_range*p )

plt.show()

plt.plot(t_range, Fr)
plt.plot(t_range, F.contact_dist_2d(1, t_range))

plt.show()
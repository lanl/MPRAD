from math import pi, sqrt, sin, cos
import numpy as np

n = 800 
data = np.zeros((n,n),np.uint16)
for i in range(n):
    for j in range(n):
        x = 2.0*(i+0.5)/n-1.0
        y = 2.0*(j+0.5)/n-1.0
        if x**2+y**2<1.0: data[i,j]=1
np.savez_compressed("disk_800.npz", data=data)

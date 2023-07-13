from cProfile import label
from ..util.kernel import *
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init()

h = 1.0 # support radius
n = 200 # number of divisions
has_neg = False

r_arr = np.array([2.0 * h / n * i - h for i in range(n + 1)]) if has_neg else np.array([1.0 * h / n * i for i in range(n + 1)])
r = ti.field(float, n + 1)
r.from_numpy(r_arr)
val = ti.field(float, n + 1)

@ti.kernel
def calcVal(ker_name: ti.template()):
    for i in range(n + 1):
        val[i] = ker_name(r[i], h)

fig, ax = plt.subplots()
toPlot = [W, WP, W_grad_norm, WP_grad_norm]
for ker in toPlot:
    calcVal(ker) # calculate kernel  
    ax.plot(r_arr, val.to_numpy(),label=ker.__name__)

ax.legend()

plt.show()

# q = 0.5
# print(1.0 / 60 * (192 * q ** 6 - 288 * q ** 5 + 160 * q ** 3 + 84 * q + 30))
# print(-8.0 / 15 * (2 * q ** 6 - 9 * q ** 5 + 15 * q ** 4 - 10 * q ** 3 + 3 * q - 1))
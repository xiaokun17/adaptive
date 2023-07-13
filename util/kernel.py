import taichi as ti
import math
@ti.func
def W(r, h):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (q ** 3 - q ** 2) + 1
    elif q > 0.5 and q < 1:
        tmp = 2 * (1 - q) ** 3
    tmp *= 8 / math.pi / h ** 3
    return tmp

@ti.func
def W_grad(x, h):
    r = x.norm()
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (3 * q ** 2 - 2 * q)
    elif q > 0.5 and q < 1:
        tmp = -6 * (1 - q) ** 2
    tmp *= 8 / math.pi / h ** 3 / h
    return tmp * x / r

@ti.func
def W_grad_norm(r, h):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 6 * (3 * q ** 2 - 2 * q)
    elif q > 0.5 and q < 1:
        tmp = -6 * (1 - q) ** 2
    tmp *= 8 / math.pi / h ** 3 / h
    return tmp

@ti.func
def WP(r, h):
    p = h - r
    tmp = 0.0
    if 1e-7 <= r <= h:
        tmp = 15 / math.pi / h ** 6 * p ** 3
    return tmp

@ti.func
def WP_grad(x, h):
    r = x.norm()
    p = h - r
    tmp = 0.0
    if 1e-7 <= r <= h:
        tmp = -45 / math.pi / h ** 6 * p ** 2
    elif r < 1e-7:
        r = h
    return tmp * x / r

@ti.func
def WP_grad_norm(r, h):
    p = h - r
    tmp = 0.0
    if 1e-7 <= r <= h:
        tmp = -45 / math.pi / h ** 6 * p ** 2
    return tmp

@ti.func
def W_lap(x, h, V_j, value):
    return 10.0 * V_j * W_grad(x, h) * value.dot(x) / (0.01 * h ** 2 + x.norm_sqr())

# for adaptive
@ti.func
def W_derivative_to_h(r, h):
    q = r / h
    tmp = 0.0
    tmp_1 = 0.0
    if q <= 0.5:
        tmp = 6 * (q ** 3 - q ** 2) + 1
        tmp_1 = 18 * q ** 2 - 12 * q
    elif q > 0.5 and q < 1:
        tmp = 2 * (1 - q) ** 3
        tmp_1 = -6 * (1 - q) ** 2
    tmp *= -24 / math.pi / h ** 4
    tmp_1 *= -8 * r / math.pi / h ** 5
    return -(tmp + tmp_1)
    
# for SDF boundary handling
@ti.func
def lamb(d, h):
    q = abs(d) / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 1.0 / 60 * (192 * q ** 6 - 288 * q ** 5 + 160 * q ** 3 - 84 * q + 30)
    elif q > 0.5 and q < 1:
        tmp = -8.0 / 15 * (2 * q ** 6 - 9 * q ** 5 + 15 * q ** 4 - 10 * q ** 3 + 3 * q - 1)
    if d < 0:
        tmp = 1 - tmp
    return tmp

# for SDF boundary handling
@ti.func
def lamb_grad_norm(d, h):
    q = abs(d) / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 1.0 / 5 * (96 * q ** 5 - 120 * q ** 4 + 40 * q ** 2 - 7)
    elif q > 0.5 and q < 1:
        tmp = -8.0 / 5 * (4 * q ** 5 - 15 * q ** 4 + 20 * q ** 3 - 10 * q ** 2 + 1)
    # if d < 0: #paper seems wrong?
    #     tmp = -tmp
    return tmp

# for surface tension
@ti.func
def C(r, h):
    q = r / h
    tmp = 0.0
    if q <= 0.5:
        tmp = 2 * (1 - q) ** 3 * q ** 3 - 1.0 / 64
    elif q > 0.5 and q < 1:
        tmp = (1 - q) ** 3 * q ** 3
    tmp *= 32 / math.pi / h ** 3
    return tmp
import taichi as ti

# centered at (0,0,0), positive inside (hollow inside)
@ti.func
def sphereSDF(pos, sphere_radius):
    temp = 0.0
    temp =  -(pos.norm() - sphere_radius)
    return temp

# """
# On line 7 of file "E:\scm\Taichi_SPH\util\misc.py":
#     temp =  -(pos.norm() - sphere_radius)
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^
# RuntimeError: [D:/a/taichi/taichi/taichi/ir/frontend_ir.cpp:type_check@209] [@tmp371] was not type-checked
# """

#unit vector of sphereSDF gradient
@ti.func
def sphereSDFUnitGrad(pos):
    tmp = ti.Vector([0.0, 0.0, 0.0])
    if pos.norm() > 1e-7:
        tmp = -pos.normalized()
    return tmp

#interpolate value_field[index_float] from nearby integer indices
@ti.func
def trilinearInterpolation(index_float, value_field):
    V = ti.static(value_field)
    p = int(index_float)
    p_d = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for i in ti.static(range(3)):
        p_d[1, i] = index_float[i] - p[i]
        p_d[0, i] = 1.0 - p_d[1, i]
    c = value_field[p] * 0
    for i in ti.static(range(2)):
        for j in ti.static(range(2)):
            for k in ti.static(range(2)):
                c += V[p[0] + i, p[1] + j, p[2] + k] * p_d[i, 0] * p_d[j, 1] * p_d[k, 2]
    return c

import taichi as ti
from taichi import sin, cos

# x, y, z: angles(rad) to rotate counterclockwise around axis
@ti.func
def rotationMatrix(x, y, z):
    rx = ti.Matrix([
        [1, 0,       0,      0],
        [0, cos(x), -sin(x), 0],
        [0, sin(x),  cos(x), 0],
        [0, 0,       0,      1]
    ])
    ry = ti.Matrix([
        [ cos(y), 0, sin(y), 0],
        [ 0,      1, 0,      0],
        [-sin(y), 0, cos(y), 0],
        [ 0,      0, 0,      1]
    ])
    rz = ti.Matrix([
        [cos(z), -sin(z), 0, 0],
        [sin(z),  cos(z), 0, 0],
        [0,       0,      1, 0],
        [0,       0,      0, 1]
    ])
    return rz @ (ry @ rx)

@ti.func
def translationMatrix(x, y, z):
    return ti.Matrix([[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, y], [0.0, 0.0, 1.0, z], [0.0, 0.0, 0.0, 1.0]])

@ti.func
def scaleMatrix(x, y, z):
    return ti.Matrix([[x, 0.0, 0.0, 0.0], [0.0, y, 0.0, 0.0], [0.0, 0.0, z, 0.0], [0.0, 0.0, 0.0, 1.0]])

@ti.func
def applyTransform(pos, matrix):
    tmp = ti.Vector([pos[0], pos[1], pos[2], 1.0])
    res = matrix @ tmp
    return ti.Vector([res[0], res[1], res[2]])

import taichi as ti
from .. import config

@ti.data_oriented
class sphereContainer:
    def __init__(self, radius, surface_type = 0):
        self.radius = radius
        self.surface_type = surface_type
        self.friction = config.sph.sdf_bound_friction

    @ti.func
    def sdf(self, pos_i):
        r = ti.static(self.radius)
        return r - pos_i.norm()

    @ti.func
    def sdf_grad(self, pos_i):
        tmp = ti.Vector([0.0, 0.0, 0.0])
        if pos_i.norm() > 1e-7:
            tmp = -pos_i.normalized()
        return tmp

    @ti.func
    def velocity(self, pos_i, time_step):
        return ti.Vector([0.0, 0.0, 0.0])
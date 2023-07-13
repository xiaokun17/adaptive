import taichi as ti
from .. import config

#only supports horizontal plane currently

@ti.data_oriented
class plane:
    def __init__(self, y_pos, surface_type = 0):
        self.y_pos = y_pos
        self.surface_type = surface_type
        self.friction = config.sph.sdf_bound_friction

    @ti.func
    def sdf(self, pos_i):
        y = ti.static(self.y_pos)
        return pos_i[1] - y

    @ti.func
    def sdf_grad(self, pos_i):
        tmp = ti.Vector([0.0, 1.0, 0.0])
        return tmp

    @ti.func
    def velocity(self, pos_i, time_step):
        return ti.Vector([0.0, 0.0, 0.0])
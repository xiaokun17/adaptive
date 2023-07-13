import taichi as ti
from .. import config

@ti.data_oriented
class cylinderContainer:
    def __init__(self, radius, height, surface_type = 0):
        self.radius = radius
        self.height = height
        self.surface_type = surface_type
        self.friction = config.sph.sdf_bound_friction

    @ti.func
    def sdf(self, pos_i):
        r, h = ti.static(self.radius, self.height)
        dist_r = r - ti.sqrt(pos_i[0] ** 2 + pos_i[2] ** 2)
        dist_h_low = pos_i[1] + h / 2.0
        dist_h_high = h / 2.0 - pos_i[1]
        return min(dist_r, min(dist_h_low, dist_h_high))

    @ti.func
    def sdf_grad(self, pos_i):
        r, h = ti.static(self.radius, self.height)
        tmp = r - ti.sqrt(pos_i[0] ** 2 + pos_i[2] ** 2)
        grad = ti.Vector([-pos_i[0], 0.0, -pos_i[2]])
        dist_h_low = pos_i[1] + h / 2.0
        if dist_h_low < tmp:
            tmp = dist_h_low
            grad = ti.Vector([0.0, 1.0, 0.0])
        dist_h_high = h / 2.0 - pos_i[1]
        if dist_h_high < tmp:
            grad = ti.Vector([0.0, -1.0, 0.0])
        return grad

    @ti.func
    def velocity(self, pos_i, time_step):
        return ti.Vector([0.0, 0.0, 0.0])
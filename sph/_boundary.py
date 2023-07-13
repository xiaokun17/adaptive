import taichi as ti
from ..util.kernel import *
# from ..util.misc import *

#very simple boundary handling for a box centered at (0,0,0)
@ti.func
def handleBoundary(self):
    num, vel, pos, l = ti.static(self.particle_count, self.velocity, self.position, self.bound_box_sidelength)
    for i in range(num[None]):
        for j in ti.static(range(3)):
            if pos[i][j] < -l/2:
                vel[i][j] = max(0,vel[i][j])
            elif pos[i][j] > l/2:
                vel[i][j] = min(0,vel[i][j])

# SDF boundary handling (density) (penalty)
@ti.func
def densityFromSphereSDFBoundary(self):
    num, rho, rho_0, pos, h = ti.static(self.particle_count, self.density, self.rest_density, self.position, self.support_radius)
    num_b, bound = ti.static(self.boundary_count, self.boundary_list)
    for i in range(num[None]):
        for j in ti.static(range(num_b)):
            sdf = bound[j].sdf(pos[i])
            beta = 1.0 - sdf / h[i]
            rho[i] += rho_0 * beta * lamb(sdf, h[i])

@ti.func
def boundaryPressure(self, i, bound): # i: fluid particle, bound: boundary object
    num, rho, rho_0, pos, h, p, vel, dt, omega = ti.static(self.particle_count, self.density, self.rest_density, self.position, self.support_radius, self.pressure, self.velocity, self.time_step, self.adaptive_corrective_factor)
    num_b = ti.static(self.boundary_count)

    sdf = bound.sdf(pos[i])
    beta = 1.0 - sdf / h[i]
    d_beta = -1.0 / h[i]
    sdf_grad = bound.sdf_grad(pos[i])
    lamb_grad = sdf_grad * d_beta * lamb(sdf, h[i]) + beta * sdf_grad / h[i] * lamb_grad_norm(sdf, h[i])
    acc_p_bound = -rho_0 * (p[i] / rho[i] ** 2 / omega[i]) * lamb_grad
    return acc_p_bound

@ti.func
def boundaryFriction(self, i, bound, acc_p_bound): # i: fluid particle, j: boundary id
    num, pos, vel, dt, fric = ti.static(self.particle_count, self.position, self.velocity, self.time_step, bound.friction)
    num_b = ti.static(self.boundary_count)

    n = bound.sdf_grad(pos[i])
    v_rel = vel[i] - bound.velocity(pos[i], dt[None])
    v_tan = v_rel - v_rel.dot(n) * n #tangential velocity
    a_fric = ti.Vector([0.0,0.0,0.0])
    if v_tan.norm() > 1e-6:
        a_fric = -fric * acc_p_bound.norm() * v_tan.normalized()
    if a_fric.norm() * dt[None] > v_tan.norm():
        a_fric = - v_tan / dt[None]
    return a_fric

@ti.func
def updateVelocityFromBoundaryPressureAndFriction(self):
    num, vel, dt, pos = ti.static(self.particle_count, self.velocity, self.time_step, self.position)
    num_b, bound = ti.static(self.boundary_count, self.boundary_list)
    for i in range(num[None]):
        for j in ti.static(range(num_b)):
            vel_new = vel[i]
            acc_p_bound = boundaryPressure(self,i, bound[j])
            vel_new += dt[None] * acc_p_bound
            a_fric = boundaryFriction(self,i, bound[j], acc_p_bound)
            vel_new += dt[None] * a_fric
            # clamp velocity
            vel_b = bound[j].velocity(pos[i], dt[None])
            vel_rel = vel[i] - vel_b
            vel_rel_new = vel_new - vel_b
            if vel_rel.norm() < vel_rel_new.norm():
                vel_rel_new *= vel_rel.norm() / (vel_rel_new.norm() + 1e-6)
            vel[i] = vel_rel_new + vel_b
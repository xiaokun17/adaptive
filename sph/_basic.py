import taichi as ti
from ..util.kernel import *
# from ..util.misc import *

@ti.func
def CFLCondition(self):
    num, dt, r_min, cs, vel = ti.static(self.particle_count, self.time_step, self.min_diameter, self.sound_speed, self.velocity)
    dt[None] = r_min / cs
    for i in range(num[None]):
        v_norm = vel[i].norm()
        if v_norm > 1e-4:
            ti.atomic_min(dt[None], r_min / v_norm * self.cfl_factor)

@ti.func
def estimateDensity(self):
    num, rho, m, pos, h, neighbCount, nextNeighb = ti.static(self.particle_count, self.density, self.mass, self.position, self.support_radius, self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        rho[i] = 0.0
        if self.to_delete[i] == 1:
            rho[i] = self.rest_density
        for k in range(neighbCount(i)):
            j = nextNeighb(i)
            rho[i] += m[j] * W((pos[i]-pos[j]).norm(),(h[i] + h[j]) / 2)

@ti.func
def gravityAdvection(self):
    num, vel, dt = ti.static(self.particle_count, self.velocity, self.time_step)
    for i in range(num[None]):
        vel[i] += dt[None] * self.gravity

@ti.func
def clampDensity(self):
    num, rho, rho_0 = ti.static(self.particle_count, self.density, self.rest_density)
    for i in range(num[None]):
        rho[i] = min(rho[i], rho_0 * 1.1)

@ti.func
def calculatePressure(self):
    num, p, rho, rho_0, cs, gamma = ti.static(self.particle_count, self.pressure, self.density, self.rest_density, self.sound_speed, self.gamma)
    for i in range(num[None]):
        p[i] = rho_0 * cs ** 2 / gamma * max((rho[i] / rho_0)**gamma - 1, 0)

@ti.func
def updateVelocityFromPressure(self):
    num, rho, m, pos, h, p, vel, dt, neighbCount, nextNeighb, omega = ti.static(self.particle_count, self.density, self.mass, self.position, self.support_radius, self.pressure, self.velocity, self.time_step, self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor, self.adaptive_corrective_factor)
    for i in range(num[None]):
        for k in range(neighbCount(i)):
            j = nextNeighb(i)
            xij = pos[i]-pos[j]
            if xij.norm() > 1e-5:
                vel[i] += -dt[None] * m[j] * (p[i]/rho[i]**2/omega[i] +p[j]/rho[j]**2/omega[j]) * WP_grad(xij,(h[i] + h[j]) / 2)

@ti.func
def updatePosition(self):
    num, vel, pos, dt = ti.static(self.particle_count, self.velocity, self.position, self.time_step)
    for i in range(num[None]):
        pos[i] += vel[i] * dt[None]

# viscosity method from WCSPH paper
@ti.func
def viscosityAdvection(self):
    num, vel, pos, h, m, rho, miu, a_vis, dt, neighbCount, nextNeighb = ti.static(self.particle_count, self.velocity, self.position, self.support_radius, self.mass, self.density, self.dynamic_viscosity, self.viscosityAcceleration, self.time_step, self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        a_vis[i] = ti.Vector([0.0,0.0,0.0])
        for k in range(neighbCount(i)):
            j = nextNeighb(i)
            xij = pos[i] - pos[j]
            if xij.norm() > 1e-5:
                vij = vel[i] - vel[j]
                if vij.dot(xij) < 0:
                    a_vis[i] += W_lap(xij, (h[i] + h[j]) / 2, m[j] / rho[j], vij) * miu / rho[i]
    for i in range(num[None]):
        vel[i] += dt[None] * a_vis[i]

#from AAT13_Versatile surface tension and adhesion for SPH fluids
@ti.func
def estimateSurfaceNormal(self): #feels like boundary should be counted but doesn't seem important
    num, n, h, m, rho, pos = ti.static(self.particle_count, self.surface_normal, self.support_radius, self.mass, self.density, self.position)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        n[i] = ti.Vector([0.0, 0.0, 0.0])
        for k in range(neighbCount(i)):
            j = nextNeighb(i)
            xij = pos[i] - pos[j]
            if xij.norm() > 1e-5:
                hij = (h[i] + h[j]) / 2
                n[i] += hij * m[j] / rho[j] * W_grad(xij, hij)

#from AAT13_Versatile surface tension and adhesion for SPH fluids
@ti.func
def surfaceTensionAdvection(self): 
    num, n, h, m, rho, vel, rho_0, gamma, dt, pos = ti.static(self.particle_count, self.surface_normal, self.support_radius, self.mass, self.density, self.velocity, self.rest_density, self.surface_tension_coefficient, self.time_step, self.position)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        for k in range(neighbCount(i)):
            j = nextNeighb(i)
            xij = pos[i] - pos[j]
            if xij.norm() > 1e-5:
                kij = 2 * rho_0 / (rho[i] + rho[j]) # correction factor in eqn(4)
                a_coh = -gamma * m[j] * C(xij.norm(), (h[i] + h[j]) / 2) * xij / xij.norm() # cohesion
                a_cur = -gamma * (n[i] - n[j]) # curvature minimalization
                vel[i] += dt[None] * kij * (a_coh + a_cur)

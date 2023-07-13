import taichi as ti
from ..util.kernel import *
# from ..util.misc import *

#================================== coupling ==================================
@ti.func
def calculateCorrectiveFactor(self):
    num, omega, m, h, rho, pos, size_class = ti.static(self.particle_count, self.adaptive_corrective_factor, self.mass, self.support_radius, self.density, self.position, self.size_class)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        if size_class[i] == 1: # class l
            omega[i] = 1.0 + m[i] * W_derivative_to_h(0.0, h[i]) * h[i] / (3 * rho[i])
        else:
            omega[i] = 0.0
            for k in range(neighbCount(i)):
                j = nextNeighb(i)
                omega[i] += m[j] * W_derivative_to_h((pos[i] - pos[j]).norm(), (h[i] + h[j]) / 2)
            omega[i] = 1.0 + omega[i] * h[i] / (3 * rho[i])

#================================== local viscosity ==================================
@ti.func
def localViscosityAdvection(self):
    num, vel, pos, h, m, rho, miu, a_vis, dt, beta, factor_vis = ti.static(self.particle_count, self.velocity, self.position, self.support_radius, self.mass, self.density, self.dynamic_viscosity, self.viscosityAcceleration, self.time_step, self.temporal_blend_factor, self.local_viscosity_factor)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        a_vis[i] = ti.Vector([0.0,0.0,0.0])
        for k in range(neighbCount(i)):
            j = nextNeighb(i)
            xij = pos[i] - pos[j]
            if xij.norm() > 1e-5:
                vij = vel[i] - vel[j]
                if vij.dot(xij) < 0:
                    miu_local = miu + miu * (beta[i] + beta[j]) / 2 * factor_vis
                    a_vis[i] += W_lap(xij, (h[i] + h[j]) / 2, m[j] / rho[j], vij) * miu_local / rho[i]
    for i in range(num[None]):
        vel[i] += dt[None] * a_vis[i]

#================================== sizing ==================================
#compute optimal mass (m_opt)
@ti.func
def simpleSizingFunction(self):
    num, pos, phi_max, m, m_opt, m_base, alpha, size_class = ti.static(self.particle_count, self.position, self.max_surface_distance, self.mass, self.optimal_mass, self.max_mass, self.max_adaptive_ratio, self.size_class)
    for i in range(num[None]):
        phi = min(max(0,-pos[i][1]), phi_max) # y distance to (0,0,0)
        m_opt[i] = m_base * (phi / phi_max * (1 - alpha) + alpha) + 1e-6

@ti.func
def surfaceSizingFunction(self):
    num, pos, phi_min, m, m_opt, m_base, alpha, phi = ti.static(self.particle_count, self.position, self.min_surface_level_set, self.mass, self.optimal_mass, self.max_mass, self.max_adaptive_ratio, self.surface_level_set)
    phi_max = -phi_min
    for i in range(num[None]):
        # phi = min(max(0,-phi[i]), phi_max) # y distance to (0,0,0)
        m_opt[i] = m_base * (-phi[i] / phi_max * (1 - alpha) + alpha) + 1e-6

@ti.func
def computeSizeClass(self):
    num, m, m_opt, size_class = ti.static(self.particle_count, self.mass, self.optimal_mass, self.size_class)
    for i in range(num[None]):
        m_rel = m[i] / m_opt[i]
        if m_rel < 0.5:
            size_class[i] = -2
        elif m_rel <= 0.9:
            size_class[i] = -1
        elif m_rel < 1.1:
            size_class[i] = 0
        elif m_rel <= 2:
            size_class[i] = 1
        else:
            size_class[i] = 2

#================================== surface detection ==================================
@ti.func
def calculateSumWeight(self):
    num, pos, weight, h, m, rho = ti.static(self.particle_count, self.position, self.sum_weight, self.support_radius, self.mass, self.density)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        weight[i] = 0.0
        for k in range(neighbCount(i)):
            j = nextNeighb(i)  
            weight[i] += W((pos[i] - pos[j]).norm(), (h[i] + h[j]) / 2)
            # weight[i] += m[j] / rho[j] * W((pos[i] - pos[j]).norm(), (h[i] + h[j]) / 2)
        # TODO: from SDF boundary

@ti.func
def detectSurface(self):
    num, pos, weight, h, phi, diam, phi_min, sur_flg, m, rho = ti.static(self.particle_count, self.position, self.sum_weight, self.support_radius, self.surface_level_set, self.diameter, self.min_surface_level_set, self.surface_flag, self.mass, self.density)
    beta = ti.static(self.temporal_blend_factor)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    num_b, bound = ti.static(self.boundary_count, self.boundary_list)
    mark, mark_phi, dt = ti.static(self.mark_remain_time, self.marked_level_set, self.time_step)
    surface_adapt, mark_time = ti.static(self.enable_surface_adaptive, self.wake_preserve_time)

    for i in range(num[None]):
        x_avg = ti.Vector([0.0, 0.0, 0.0]) # weighted average position of neighbors
        r_avg = 0.0 # weighted average radius of neighbors
        nei_num = 0 # number of neighbors
        for k in range(neighbCount(i)):
            j = nextNeighb(i)
            rij = (pos[i] - pos[j]).norm()
            hij = (h[i] + h[j]) / 2
            if rij <= hij:
                wij = W(rij, hij) / weight[i]
                # wij = m[j] / rho[j] * W(rij, hij) / weight[i]
                x_avg += wij * pos[j]
                r_avg += wij * diam[j] / 2.0
                nei_num += 1

        bound_flg = 0
        for j in ti.static(range(num_b)):
            if bound[j].sdf(pos[i]) <= h[i]:
                if bound[j].surface_type == 0:
                    phi[i] = phi_min
                    sur_flg[i] = 0
                    bound_flg = 1
                elif bound[j].surface_type == 2:
                    sdf = min(-bound[j].sdf(pos[i]), 0.0)
                    sur_flg[i] = 1
                    bound_flg = 1
                    phi[i] = sdf
                    if mark[i] <= 1e-6:
                        mark_phi[i] = phi[i]
                    else:
                        mark_phi[i] = max(mark_phi[i], phi[i])
                    mark[i] = mark_time # adjust mark value

        if bound_flg == 0:
            if surface_adapt:
                r_i = diam[i] / 2.0
                # handle too much or too few neighbors
                if nei_num < 5: #temporarily ignore this
                    phi[i] = -0.8499 * r_i
                    sur_flg[i] = 1
                elif nei_num > 29: #sdf: force particles near boundary to use phi_min
                    phi[i] = phi_min
                    sur_flg[i] = 0
                else:
                    # calculate phi for particles that have medium amount of neighbors
                    phi_tmp = (pos[i] - x_avg).norm() - r_avg
                    max_change = 1.5 * r_i
                    phi_tmp = min(phi[i] + max_change, max(phi[i] - max_change, phi_tmp)) # clamp within max_change
                    phi[i] = min(0.0, max(phi_min, phi_tmp)) # clamp within [phi_min, 0]
                    if phi[i] >= -0.85 * r_i:
                        sur_flg[i] = 1
                    else:
                        phi[i] = phi_min
                        sur_flg[i] = 0
            else:
                phi[i] = phi_min
                sur_flg[i] = 0

        if mark[i] > 1e-6:
            if mark_phi[i] > phi[i]:
                phi[i] = mark_phi[i]
                sur_flg[i] = 1
            mark[i] -= dt[None]

@ti.func
def surfacePropagationStep(self):
    num, pos, phi, phi_min, sur_flg, prop_flg = ti.static(self.particle_count, self.position, self.surface_level_set, self.min_surface_level_set, self.surface_flag, self.propagation_flag)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        phi_next = phi[i]
        if sur_flg[i] == 0:
            for k in range(neighbCount(i)):
                j = nextNeighb(i)
                if sur_flg[j] != 0:
                    phi_tmp = phi[j] - (pos[i] - pos[j]).norm()
                    if phi_tmp > phi_next:
                        phi_next = phi_tmp
                        sur_flg[i] = 2
                        prop_flg[None] = 1
        phi[i] = phi_next

@ti.func
def smoothSurfaceLevelSet(self):
    num, pos, h, phi, phi_smooth, m, rho, phi_min = ti.static(self.particle_count, self.position, self.support_radius, self.surface_level_set, self.surface_level_set_smooth, self.mass, self.density, self.min_surface_level_set)
    num_b, bound = ti.static(self.boundary_count, self.boundary_list)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        phi_smooth[i] = 0.0
        for k in range(neighbCount(i)):
            j = nextNeighb(i)
            phi_smooth[i] += m[j] / rho[j] * phi[j] * W((pos[i] - pos[j]).norm(), (h[i] + h[j]) / 2)
        # from sphere SDF boundary
        for j in ti.static(range(num_b)):
            sdf = bound[j].sdf(pos[i])
            beta = 1.0 - sdf / h[i]
            phi_smooth[i] += beta * phi[i] * lamb(sdf, h[i])
    for i in range(num[None]):
        phi[i] = min(0.0, max(phi_min, phi_smooth[i]))

#================================== split & merge ==================================
#---------------- process particles ----------------
@ti.func
def initializeOriginalParticle(self, i):
    num = ti.static(self.original.particle_count)
    k = ti.atomic_add(num[None], 1)
    self.original.density[k] = self.density[i]
    self.original.position[k] = self.position[i]
    self.original.velocity[k] = self.velocity[i]
    self.original.mass[k] = self.mass[i]
    self.original.support_radius[k] = self.support_radius[i]
    self.original.children_count[k] = 0
    self.original.new_id[k] = k
    return k

@ti.func
def initializeSplitParticleWithRot(self, i, o, n, serial, o_index, rot): #i: new particle, o: original particle, n: number to split, serial:serial of added particle in all added particles, o_index: index of original in original fields
    self.diameter[i] = self.diameter[o] / n ** (1.0 / 3)
    self.mass[i] = self.mass[o] / n
    self.support_radius[i] = self.support_radius[o] / n ** (1.0 / 3)
    pattern = self.rotation_pattern[rot] @ self.pattern[n-1, serial]
    self.position[i] = self.position[o] + pattern * self.diameter[o] * 1.6#self.support_radius[o]#self.diameter[o]#self.support_radius[o]
    self.velocity[i] = self.velocity[o]
    self.temporal_blend_factor[i] = self.initial_temporal_blend_factor
    self.original_particle_id[i] = o_index
    self.merge_partner[i] = -1
    self.partner_count[i] = 0
    self.to_delete[i] = 0
    self.surface_level_set[i] = min(self.surface_level_set[o], -self.diameter[i] * 0.8499)
    self.mark_remain_time[i] = self.mark_remain_time[o]
    self.marked_level_set[i] = self.marked_level_set[o]

    self.size_class[i] = 0 #prevent marticipation in merging

@ti.func
def initializeSplitParticle(self, i, o, n, serial, o_index): #i: new particle, o: original particle, n: number to split, serial:serial of added particle in all added particles, o_index: index of original in original fields
    self.diameter[i] = self.diameter[o] / n ** (1.0 / 3)
    self.mass[i] = self.mass[o] / n
    self.support_radius[i] = self.support_radius[o] / n ** (1.0 / 3)
    self.position[i] = self.position[o] + self.pattern[n-1, serial] * self.diameter[o] * 1.6#self.support_radius[o]#self.diameter[o]#self.support_radius[o]
    self.velocity[i] = self.velocity[o]
    self.temporal_blend_factor[i] = self.initial_temporal_blend_factor
    self.original_particle_id[i] = o_index
    self.merge_partner[i] = -1
    self.partner_count[i] = 0
    self.to_delete[i] = 0
    self.surface_level_set[i] = min(self.surface_level_set[o], -self.diameter[i] * 0.8499)
    self.mark_remain_time[i] = self.mark_remain_time[o]
    self.marked_level_set[i] = self.marked_level_set[o]

    self.size_class[i] = 0 #prevent marticipation in merging

@ti.func
def updateSplitParticle(self, i): # delete should be called immediatly afterwards
    self.to_delete[i] = 1
    
@ti.func
def mergeToParticle(self, i, o, m_dist, o_index): #i: distribute to, o: distribute from, m_dist: mass to distribute, o_index: index of original in original fields
    m_pre = self.mass[i]
    self.mass[i] = self.mass[i] + m_dist
    self.diameter[i] = (self.mass[i] / self.rest_density) ** (1.0 / 3)
    self.support_radius[i] = self.diameter[i] * 2

    self.position[i] = (m_dist * self.position[o] + m_pre * self.position[i]) / self.mass[i]
    self.velocity[i] = (m_dist * self.velocity[o] + m_pre * self.velocity[i]) / self.mass[i]
    self.surface_level_set[i] = (m_dist * self.surface_level_set[o] + m_pre * self.surface_level_set[i]) / self.mass[i]
    self.temporal_blend_factor[i] = self.initial_temporal_blend_factor_merge
    self.original_particle_id[i] = o_index
    self.merge_partner[i] = -1
    self.partner_count[i] = 0
    self.to_delete[i] = 0
    self.size_class[i] = 0

@ti.func
def updateMergedParticle(self, i): #i: distribute to, o_index: index of original in original fields
    self.mass[i] = 0.0
    self.diameter[i] = 0.0
    self.support_radius[i] = 0.0

    self.position[i] = self.position[i]
    self.velocity[i] = self.velocity[i]
    self.temporal_blend_factor[i] = 0.0
    self.original_particle_id[i] = -1
    self.merge_partner[i] = -1
    self.partner_count[i] = 0
    self.to_delete[i] = 1
    self.size_class[i] = 0

@ti.func
def redistributeToParticle(self, i, o, m_dist, o_index): #i: distribute to, o: distribute from, m_dist: mass to distribute, o_index: index of original in original fields
    m_pre = self.mass[i]
    self.mass[i] = self.mass[i] + m_dist
    self.diameter[i] = (self.mass[i] / self.rest_density) ** (1.0 / 3)
    self.support_radius[i] = self.diameter[i] * 2

    self.position[i] = (m_dist * self.position[o] + m_pre * self.position[i]) / self.mass[i]
    self.velocity[i] = (m_dist * self.velocity[o] + m_pre * self.velocity[i]) / self.mass[i]
    self.surface_level_set[i] = (m_dist * self.surface_level_set[o] + m_pre * self.surface_level_set[i]) / self.mass[i]
    self.temporal_blend_factor[i] = self.initial_temporal_blend_factor_redistribute
    self.original_particle_id[i] = o_index
    self.merge_partner[i] = -1
    self.partner_count[i] = 0
    self.to_delete[i] = 0
    self.size_class[i] = 0

@ti.func
def updateRedistributedParticle(self, i, o_index): #i: distribute to, o_index: index of original in original fields
    self.mass[i] = self.optimal_mass[i] + (self.mass[i] - self.optimal_mass[i]) / (self.partner_count[i] + 1) #keep some mass
    self.diameter[i] = (self.mass[i] / self.rest_density) ** (1.0 / 3)
    self.support_radius[i] = self.diameter[i] * 2

    self.position[i] = self.position[i]
    self.velocity[i] = self.velocity[i]
    self.temporal_blend_factor[i] = self.initial_temporal_blend_factor_redistribute
    self.original_particle_id[i] = o_index
    self.merge_partner[i] = -1
    self.partner_count[i] = -1
    self.to_delete[i] = 0
    self.size_class[i] = 0

#---------------- split ----------------
@ti.func
def _computeSplitError(self, i, bg, s_bg, n_s): #i: split particle, bg: begin index of all inserted particles this step, s_bg: begin of split particle, n_s: number of inserted particles
    num, m, pos, h, rho_0 = ti.static(self.particle_count, self.mass, self.position, self.support_radius, self.rest_density)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    # neighbs of i and particles from bg to s may be neighbors of i
    # error from neighbs of i
    error = 0.0
    for k in range(neighbCount(i)): #neighb info of i is still the old one
        j = nextNeighb(i)
        error_j = -m[i] * W((pos[j] - pos[i]).norm(), (h[i] + h[j]) / 2)
        for s in range(s_bg, s_bg + n_s):
            error_j += m[s] * W((pos[j] - pos[s]).norm(), (h[s] + h[j]) / 2)
        error += max(0.0, error_j)
    for j in range(bg, s_bg):
        r_ji = (pos[j] - pos[i]).norm()
        if r_ji <= (h[i] + h[j]) / 2: #determine if j is neighb of i
            error_j = -m[i] * W(r_ji, (h[i] + h[j]) / 2)
            for s in range(s_bg, s_bg + n_s):
                error_j += m[s] * W((pos[j] - pos[s]).norm(), (h[s] + h[j]) / 2)
            error += max(0.0, error_j)
    # error from s
    for s in range(s_bg, s_bg + n_s):
        error_s = -rho_0
        for k in range(neighbCount(i)): #neighb info of i is still the old one
            j = nextNeighb(i)
            # if j != i: #shouldn't include i but seem stabler
            error_s += m[j] * W((pos[j] - pos[s]).norm(), (h[s] + h[j]) / 2)
        for j in range(bg, s_bg):
            r_ji = (pos[j] - pos[i]).norm()
            if r_ji <= (h[i] + h[j]) / 2: #determine if j is neighb of i
                error_s += m[j] * W((pos[j] - pos[s]).norm(), (h[s] + h[j]) / 2)
        for k in range(s_bg, s_bg + n_s):
            error_s += m[k] * W((pos[k] - pos[s]).norm(), (h[s] + h[k]) / 2)
        error += max(0.0, error_s)
    return error

@ti.func
def splitParticleWithOptimize(self, i, n, prev_num): #should be serial
    o_index = initializeOriginalParticle(self,i)
    num, rot_num = ti.static(self.particle_count, self.rotation_pattern_count)
    s_bg = ti.atomic_add(num[None], n)
    error = 1.0e10
    rot_index = 0
    #find best rot
    for rot in range(rot_num):
        for j in range(n):
            index = s_bg + j
            initializeSplitParticleWithRot(self,index, i, n, j, o_index, rot)
        error1 = _computeSplitError(self,i, prev_num, s_bg, n)
        if error1 < error:
            error = error1
            rot_index = rot
    # print(error)
    #execute rot
    for j in range(n):
        index = s_bg + j
        initializeSplitParticleWithRot(self,index, i, n, j, o_index, rot_index)

    updateSplitParticle(self,i)

@ti.func
def handleSplittingWithOptimize(self):
    num, size_class, m, m_opt, o = ti.static(self.particle_count, self.size_class, self.mass, self.optimal_mass, self.original_particle_id)
    ratio = ti.static(self.max_adaptive_ratio)
    prev_num = num[None]
    for serial in range(1):
        for i in range(prev_num):
            if size_class[i] == 2 and o[i] == -1:
                split_count = max(min(ti.ceil(m[i] / m_opt[i]), 1.0 / ratio), 1.0) # clamp split_count
                splitParticleWithOptimize(self,i, int(split_count + 1e-5), prev_num)

#---------------- split without optimize ----------------
@ti.func
def splitParticle(self, i, n):
    o_index = initializeOriginalParticle(self,i)
    num = ti.static(self.particle_count)
    for j in range(n):
        index = ti.atomic_add(num[None], 1)
        initializeSplitParticle(self,index, i, n, j, o_index)
    # initializeSplitParticle(self,i, i, n, n - 1, o_index)
    updateSplitParticle(self,i)

# unused
@ti.kernel
def splitParticles(self: ti.template(), n: int):
    num = ti.static(self.particle_count)
    prev_num = num[None]
    for i in range(prev_num):
        splitParticle(self,i, n)

@ti.func
def handleSplitting(self):
    num, size_class, m, m_opt, o = ti.static(self.particle_count, self.size_class, self.mass, self.optimal_mass, self.original_particle_id)
    ratio = ti.static(self.max_adaptive_ratio)
    prev_num = num[None]
    for i in range(prev_num):
        if size_class[i] == 2 and o[i] == -1:
            split_count = max(min(ti.ceil(m[i] / m_opt[i]), 1.0 / ratio), 1.0) # clamp split_count
            splitParticle(self,i, int(split_count + 1e-5))

#---------------- redistribute ----------------
@ti.func
def markPotentialRedistributePartners(self):
    num, pos, size_class, partner, o, h = ti.static(self.particle_count, self.position, self.size_class, self.merge_partner, self.original_particle_id, self.support_radius)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        partner[i] = -1
    for i in range(num[None]):
        if size_class[i] == 1 and o[i] == -1:
            for k in range(neighbCount(i)):
                j = nextNeighb(i)
                if (pos[i] - pos[j]).norm() <= h[i] / 2.0 and size_class[j] == -2 and o[j] == -1:
                    partner[j] = i

@ti.func
def finalizeRedistributePartners(self):
    num, num_partner, m, m_opt, m_base, partner = ti.static(self.particle_count, self.partner_count, self.mass, self.optimal_mass, self.max_mass, self.merge_partner)
    for i in range(num[None]):
        num_partner[i] = 0
    for j in range(num[None]):
        i = partner[j]
        if i != -1:
            if m[j] + (m[i] - m_opt[i]) / (num_partner[i] + 1) <= m_base:
                num_partner[i] += 1
            else:
                partner[j] = -1

@ti.func
def redistributeParticles(self):
    num, num_partner, o, m, m_opt, partner = ti.static(self.particle_count, self.partner_count, self.original_particle_id, self.mass, self.optimal_mass, self.merge_partner)
    m_o, h_o, rho_0, size_class = ti.static(self.original.mass, self.original.support_radius, self.rest_density, self.size_class)
    # push original particles
    for i in range(num[None]):
        if num_partner[i] > 3: # minimal redistribute partner
            o[i] = initializeOriginalParticle(self,i)
            #tamper original particle
            m_o[o[i]] = m[i] - m_opt[i]
            h_o[o[i]] = (m_o[o[i]] / rho_0) ** (1.0 / 3) * 2
        elif size_class[i] == 1: # set non-redistributed size_class to 0
            size_class[i] = 0
    # redistribute to particles
    for j in range(num[None]):
        i = partner[j]
        if i != -1:
            if num_partner[i] > 3:# minimal redistribute partner
                redistributeToParticle(self,j, i, (m[i] - m_opt[i]) / (num_partner[i] + 1), o[i])
            else:
                partner[j] = -1
    # update redistributed particles
    for i in range(num[None]):
        if num_partner[i] > 3:# minimal redistribute partner
            updateRedistributedParticle(self,i, o[i])

#---------------- merge ----------------
@ti.func
def markPotentialMergePartners(self): #the partners of the particle to be merged include itself
    num, pos, size_class, partner, o, h = ti.static(self.particle_count, self.position, self.size_class, self.merge_partner, self.original_particle_id, self.support_radius)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        partner[i] = -1
    for i in range(num[None]):
        if size_class[i] == -2 and o[i] == -1:
            for k in range(neighbCount(i)):
                j = nextNeighb(i)
                if (pos[i] - pos[j]).norm() <= h[i] / 2.0 and (size_class[j] == -2 or size_class[j] == -1) and o[j] == -1:
                    partner[j] = i

@ti.func
def finalizeMergePartners(self):
    num, num_partner, size_class, m, m_opt, m_base, partner = ti.static(self.particle_count, self.partner_count, self.size_class, self.mass, self.optimal_mass, self.max_mass, self.merge_partner)
    neighbCount, nextNeighb = ti.static(self.neighbor_list.getCountAndReset, self.neighbor_list.getNextNeighbor)
    for i in range(num[None]):
        num_partner[i] = 0
        if size_class[i] == -2 and partner[i] != i:
            partner[i] = -1
    #delete partner[i] != i
    for j in range(num[None]):
        i = partner[j]
        if i != -1:
            if partner[i] != i:
                partner[j] = -1
    for j in range(num[None]): #now i is discluded
        i = partner[j]
        if i != -1:
            if i != j and m[j] + m[i] / (num_partner[i] + 1) <= m_base:
                num_partner[i] += 1
            else:
                partner[j] = -1
    #runtime check
    for i in range(num[None]):
        if num_partner[i] != 0 and partner[i] != -1:
            print('ERROR: finalizeMergePartners, i:', i, 'num_partner', num_partner[i], 'partner:', partner[i])
        num_check = 0
        for k in range(neighbCount(i)):
            j = nextNeighb(i)
            if partner[j] == i:
                num_check += 1
        if num_check != num_partner[i]:
            print('ERROR: finalizeMergePartners, i:', i, 'num_partner', num_partner[i], 'num_check:', num_check)

@ti.func
def mergeParticles(self):
    num, num_partner, o, m, m_opt, partner = ti.static(self.particle_count, self.partner_count, self.original_particle_id, self.mass, self.optimal_mass, self.merge_partner)
    # push original particles
    for i in range(num[None]):
        if num_partner[i] > 0:
            o[i] = initializeOriginalParticle(self,i)
    # merge to particles
    for j in range(num[None]):
        i = partner[j]
        if i != -1 and j != i:
            mergeToParticle(self,j, i, m[i] / num_partner[i], o[i])
    # update merged particles
    for i in range(num[None]):
        if num_partner[i] > 0:
            updateMergedParticle(self,i)


@ti.func
def replaceParticle(self, i, j):
    self.diameter[i] = self.diameter[j]
    self.mass[i] = self.mass[j]
    self.support_radius[i] = self.support_radius[j]
    self.position[i] = self.position[j]
    self.velocity[i] = self.velocity[j]
    self.temporal_blend_factor[i] = self.temporal_blend_factor[j]
    self.original_particle_id[i] = self.original_particle_id[j]
    self.merge_partner[i] = self.merge_partner[j]
    self.partner_count[i] = self.partner_count[j]
    self.to_delete[i] = self.to_delete[j]
    self.to_delete[j] = 1
    self.size_class[i] = self.size_class[j]
    self.surface_level_set[i] = self.surface_level_set[j]
    self.mark_remain_time[i] = self.mark_remain_time[j]
    self.marked_level_set[i] = self.marked_level_set[j]

@ti.func
def removeToDeleteParticles(self):
    num, del_num, del_list, del_flg = ti.static(self.particle_count, self.delete_helper_count, self.delete_helper_list, self.to_delete)
    num_pre = num[None]
    num_cur = max(0, num[None] - 1)
    del_num[None] = 0
    for i in range(num_pre):
        if del_flg[i] == 1:
            d_n = ti.atomic_add(del_num[None], 1)
            del_list[d_n] = i
    for serial in range(1): #serial loop
        for k in range(del_num[None]):
            to_del = del_list[k]
            # find the first not-to-delete one fron right to left
            while del_flg[num_cur] == 1 and num_cur > 0:
                num_cur = num_cur - 1
            if to_del < num_cur:
                replaceParticle(self,to_del, num_cur)
        if del_flg[num_cur] == 0:
            num[None] = num_cur + 1
        else:
            num[None] = num_cur

    # runtime check for deletion, comment out for better speed
    if num[None] != num_pre - del_num[None]:
        print('ERROR: removeToDeleteParticles(): wrong number of particles deleted.','num:', num[None], 'num_pre:', num_pre, 'del_num', del_num[None], 'num_cur', num_cur)
        for i in range(del_num[None]):
            print(del_list[i])
    for i in range(num[None]):
        if del_flg[i] == 1:
            print('ERROR: removeToDeleteParticles(): wrong particle deleted')

#================================== temporal blending ==================================
@ti.func
def estimateDensity_ori(self):
    num_o, rho_o, m, pos, pos_o, h, h_o, o, m_o, rho_0 = ti.static(self.original.particle_count, self.original.density, self.mass, self.position, self.original.position, self.support_radius, self.original.support_radius, self.original_particle_id, self.original.mass, self.rest_density)
    neighbCount_o, nextNeighb_o = ti.static(self.original.neighbor_list.getCountAndReset, self.original.neighbor_list.getNextNeighbor)
    for i in range(num_o[None]):
        rho_o[i] = 0.0
        for k in range(neighbCount_o(i)):
            j = nextNeighb_o(i)
            if o[j] != i:
                rho_o[i] += m[j] * W((pos_o[i]-pos[j]).norm(),(h_o[i] + h[j]) / 2)
        rho_o[i] += m_o[i] * W(0, h_o[i])
        rho_o[i] = min(rho_o[i], rho_0) # clamping

# SDF boundary handling for original particle (density) (penalty)
@ti.func
def originalDensityFromSDFBoundary(self):
    num_o, rho_o, rho_0, pos_o, h_o = ti.static(self.original.particle_count, self.original.density, self.rest_density, self.original.position, self.original.support_radius)
    num_b, bound = ti.static(self.boundary_count, self.boundary_list)
    for i in range(num_o[None]):
        for j in ti.static(range(num_b)):
            sdf = bound[j].sdf(pos_o[i])
            beta = 1.0 - sdf / h_o[i]
            rho_o[i] += rho_0 * beta * lamb(sdf, h_o[i])

@ti.func
def blendDensity(self):
    num, rho, rho_o, o, beta = ti.static(self.particle_count, self.density, self.original.density, self.original_particle_id, self.temporal_blend_factor)
    for i in range(num[None]):
        if o[i] != -1:
            rho[i] = (1 - beta[i]) * rho[i] + beta[i] * rho_o[o[i]]

@ti.func
def calculateOriginalVelocity(self): #side effect: update num_child
    num, num_o, num_child, o, vel, vel_o = ti.static(self.particle_count, self.original.particle_count, self.original.children_count, self.original_particle_id, self.velocity, self.original.velocity)
    for i in range(num_o[None]):
        num_child[i] = 0
        vel_o[i] = ti.Vector([0.0,0.0,0.0])
    for i in range(num[None]):
        if o[i] != -1:
            ti.atomic_add(vel_o[o[i]], vel[i])
            ti.atomic_add(num_child[o[i]], 1)
    for i in range(num_o[None]):
        vel_o[i] = vel_o[i] / num_child[i]

@ti.func
def blendVelocity(self):
    num, o, vel, beta, vel_o = ti.static(self.particle_count, self.original_particle_id, self.velocity, self.temporal_blend_factor, self.original.velocity)
    for i in range(num[None]):
        if o[i] != -1:
            vel[i] = (1 - beta[i]) * vel[i] + beta[i] * vel_o[o[i]]

@ti.func
def updateOriginalPosition(self):
    num_o, pos_o, vel_o, dt = ti.static(self.original.particle_count, self.original.position, self.original.velocity, self.time_step)
    for i in range(num_o[None]):
        pos_o[i] += vel_o[i] * dt[None]

@ti.func
def decrementTemporalBlendFactor(self):
    num, beta, o = ti.static(self.particle_count, self.temporal_blend_factor, self.original_particle_id)
    for i in range(num[None]):
        beta[i] = max(beta[i] - self.temporal_blend_factor_decrease, 0.0)
        if beta[i] < 1e-6:
            o[i] = -1

@ti.func
def replaceOriginalParticle(self, i, j):
    self.original.density[i] = self.original.density[j]
    self.original.position[i] = self.original.position[j]
    self.original.velocity[i] = self.original.velocity[j]
    self.original.mass[i] = self.original.mass[j]
    self.original.support_radius[i] = self.original.support_radius[j]
    self.original.children_count[i] = self.original.children_count[j]
    self.original.children_count[j] = 0
    self.original.new_id[i] = -1
    self.original.new_id[j] = i

@ti.func
def removeOriginalParticles(self):
    num_o, del_list, del_num, num_child, num, o, new_id = ti.static(self.original.particle_count, self.original.delete_helper_list, self.original.delete_helper_count, self.original.children_count, self.particle_count, self.original_particle_id, self.original.new_id)
    num_pre = num_o[None]
    num_cur = max(0, num_o[None] - 1)
    del_num[None] = 0
    for i in range(num_pre):
        new_id[i] = i
        if num_child[i] <= 0:
            d_n = ti.atomic_add(del_num[None], 1)
            del_list[d_n] = i
    for serial in range(1): #serial loop
        for k in range(del_num[None]):
            to_del = del_list[k]
            # find the first non-empty one fron right to left
            while num_child[num_cur] <= 0 and num_cur > 0:
                num_cur = num_cur - 1
            if to_del < num_cur:
                replaceOriginalParticle(self,to_del, num_cur)
        if num_child[num_cur] > 0:
            num_o[None] = num_cur + 1
        else:
            num_o[None] = num_cur

        # # logging delete process
        # else:
        #     print('del_num:', del_num[None])
        #     for k in range(del_num[None]):
        #         to_del = del_list[k]
        #         print('to_del:', to_del)
        #         # find the first non-empty one fron right to left
        #         while num_child[num_cur] <= 0 and num_cur > 0:
        #             num_cur = num_cur - 1
        #         print('num_cur', num_cur)
        #         if to_del < num_cur:
        #             print('num_child',num_child[to_del])
        #             replaceOriginalParticle(self,to_del, num_cur)
        #             print('num_child',num_child[num_cur])
        #     if num_child[num_cur] > 0:
        #         num_o[None] = num_cur + 1
        #     else:
        #         num_o[None] = num_cur

    # runtime check for deletion, comment out for better speed
    if num_o[None] != num_pre - del_num[None]:
        print('ERROR: removeOriginalParticles(): wrong number of particles deleted')
        # for i in range(del_num[None]):
        #     print(del_list[i])
    for i in range(num_o[None]):
        if num_child[i] <= 0:
            print('ERROR: removeOriginalParticles(): wrong particle deleted')
    
    # print("INFO: num_o:", num_o[None], ', num_pre:', num_pre, ', del_num:', del_num[None])

    # update original_particle_id
    for i in range(num[None]):
        if o[i] != -1:
            o[i] = new_id[o[i]]
            if o[i] == -1:
                print('ERROR: update original_particle_id')





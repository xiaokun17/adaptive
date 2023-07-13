import taichi as ti
import math
from .. import config
@ti.data_oriented
class NeighborList:
    def __init__(self, max_particle_count = config.sph.max_particle_count, max_neighbor_count = config.sph.max_neighbor_count):
        self.max_particle_count = max_particle_count
        self.max_neighbor_count = max_neighbor_count
        self.use_sparse = config.sph.neighbor_use_sparse
        blocks = config.sph.neighbor_list_blocks

        self.current_neighbor_index = ti.field(int, self.max_particle_count)
        self.neighbor_count = ti.field(int, self.max_particle_count)

        self.neighbor_list = ti.field(int)
        self.neighbor_list_snode = None
        if self.use_sparse:
            self.neighbor_list_snode = ti.root.pointer(ti.ij, (blocks, blocks))
            self.neighbor_list_snode.dense(ti.ij, (math.ceil(self.max_particle_count/blocks), math.ceil(self.max_neighbor_count/blocks))).place(self.neighbor_list) #TODO: sparsity
            # ti.root.pointer(ti.i, blocks).dense(ti.i, math.ceil(self.max_particle_count / blocks - 1e-6)).pointer(ti.j,blocks).dense(ti.j, math.ceil(self.max_neighbor_count / blocks - 1e-6)).place(self.neighbor_list)
        else:
            self.neighbor_list_snode = ti.root.dense(ti.ij, (self.max_particle_count, self.max_neighbor_count))
            self.neighbor_list_snode.place(self.neighbor_list) #TODO: sparsity


    @ti.func
    def establishNeighborList(self, neighborSearch, position, particle_count, search_range):
        num, pos, nl, nc = ti.static(particle_count, position, self.neighbor_list, self.neighbor_count)
        # if self.use_sparse:
        #     self.neighbor_list_snode.deactivate_all() #need to be called in python scope (any substitute in taichi scope?)
        grid_range = int(ti.ceil(neighborSearch.grid_size / search_range - 1e-5))
        grid_range_vec = ti.Vector([grid_range]*3)
        grid_range_vec1 = ti.Vector([grid_range+1]*3)
        for i in range(num[None]):
            i_grid = neighborSearch.getGridIndex(pos[i]) #index of this grid
            nc[i] = 0
            for x in range(-grid_range, grid_range + 1):
                for y in range(-grid_range, grid_range + 1):
                    for z in range(-grid_range, grid_range + 1):
                        j_grid = i_grid \
                            + x * neighborSearch.grid_index_helper[0] \
                            + y * neighborSearch.grid_index_helper[1] \
                            + z * neighborSearch.grid_index_helper[2] # index of neighboring (including this) grid
                        if 0 < j_grid < neighborSearch.grid_count:
                            for l in range(neighborSearch.grid_begin_index[j_grid], neighborSearch.grid_end_index[j_grid]):
                                j = neighborSearch.particle_id[l]
                                if (pos[i] - pos[j]).norm() <= search_range:
                                    nl[i, nc[i]] = j
                                    nc[i] = nc[i] + 1

    @ti.func
    def establishNeighborList_ori(self, neighborSearch, position, neighbor_position, particle_count, search_range):
        num, pos, nl, nc, pos_nei = ti.static(particle_count, position, self.neighbor_list, self.neighbor_count, neighbor_position)
        # if self.use_sparse:
        #     self.neighbor_list_snode.deactivate_all() #need to be called in python scope (any substitute in taichi scope?)
        grid_range = int(ti.ceil(neighborSearch.grid_size / search_range - 1e-5))
        grid_range_vec = ti.Vector([grid_range]*3)
        grid_range_vec1 = ti.Vector([grid_range+1]*3)
        for i in range(num[None]):
            i_grid = neighborSearch.getGridIndex(pos[i]) #index of this grid
            nc[i] = 0
            for x in range(-grid_range, grid_range + 1):
                for y in range(-grid_range, grid_range + 1):
                    for z in range(-grid_range, grid_range + 1):
                        j_grid = i_grid \
                            + x * neighborSearch.grid_index_helper[0] \
                            + y * neighborSearch.grid_index_helper[1] \
                            + z * neighborSearch.grid_index_helper[2] # index of neighboring (including this) grid
                        if 0 < j_grid < neighborSearch.grid_count:
                            for l in range(neighborSearch.grid_begin_index[j_grid], neighborSearch.grid_end_index[j_grid]):
                                j = neighborSearch.particle_id[l]
                                if (pos[i] - pos_nei[j]).norm() <= search_range:
                                    nl[i, nc[i]] = j
                                    nc[i] = nc[i] + 1


    @ti.func
    def getCountAndReset(self, i):
        index, num = ti.static(self.current_neighbor_index, self.neighbor_count)
        index[i] = -1
        return num[i]

    @ti.func
    def getNextNeighbor(self, i):
        index, neighbs = ti.static(self.current_neighbor_index, self.neighbor_list)
        index[i] = index[i] + 1
        return neighbs[i, index[i]]
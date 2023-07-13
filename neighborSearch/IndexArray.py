# index sort without sorting, xyr's method
import taichi as ti
from .. import config
import math
@ti.data_oriented
class IndexArray:
    def __init__(self, max_particle_count = config.sph.max_particle_count):
        #constants
        max_corner = config.sph.simulation_space_max_corner
        min_corner = config.sph.simulation_space_min_corner
        self.grid_size = config.sph.diameter * 2
        grid_count_vec = [int(math.ceil((max_corner[i] - min_corner[i]) / self.grid_size)) for i in range(3)]
        self.grid_index_helper = ti.Vector([grid_count_vec[1]*grid_count_vec[2], grid_count_vec[2], 1])
        self.grid_count = grid_count_vec[0] * grid_count_vec[1] * grid_count_vec[2]
        self.max_particle_count = max_particle_count
        self.min_corner = ti.Vector(min_corner)

        self.grid_particle_count = ti.field(int, self.grid_count)
        self.grid_begin_index = ti.field(int, self.grid_count)
        self.grid_end_index = ti.field(int, self.grid_count)
        self.particle_id = ti.field(int, self.max_particle_count)

    @ti.func
    def getGridIndex(self, position):
        return int((position - self.min_corner) // self.grid_size).dot(self.grid_index_helper)

    @ti.func
    def countParticleInGrids(self, position, particle_count):
        num, pos = ti.static(particle_count, position)
        for i in range(self.grid_count):
            self.grid_particle_count[i] = 0
        for i in range(num[None]):
            index = self.getGridIndex(pos[i])
            if 0 < index < self.grid_count:
                self.grid_particle_count[index] += 1

    @ti.func
    def computeGridBeginIndex(self):
        sum = 0
        for i in range(self.grid_count):
            self.grid_begin_index[i] = ti.atomic_add(sum, self.grid_particle_count[i])
            self.grid_end_index[i] = self.grid_begin_index[i]

    @ti.func
    def fillGrids(self, position, particle_count):
        num, pos = ti.static(particle_count, position)
        for i in range(num[None]):
            index = self.getGridIndex(pos[i])
            if 0 < index < self.grid_count:
                j = ti.atomic_add(self.grid_end_index[index], 1)
                self.particle_id[j] = i

    @ti.func
    def establishNeighbors(self, position, particle_count):
        self.countParticleInGrids(position, particle_count)
        self.computeGridBeginIndex()
        self.fillGrids(position, particle_count)


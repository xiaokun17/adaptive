import numpy as np
import taichi as ti
from plyfile import *
from .. import config

# helper function for centering cubic matrice
def _helpCentering(min_corner, max_corner, spacing):
    max_corner = np.array(max_corner, dtype=np.float32)
    min_corner = np.array(min_corner, dtype=np.float32)
    matrix_shape = ((max_corner - min_corner + 1e-7) / spacing).astype(np.int32)
    padding = (max_corner - min_corner - matrix_shape * spacing) / 2
    return matrix_shape, padding

@ti.kernel
def _addPositionField(particles: ti.template(), pos_field: ti.template(), num_to_add: int, num_current: int):
    for i in range(num_to_add):
        particles.position[i + num_current] = pos_field[i]

@ti.kernel
def _initializeParticles(particles: ti.template(), diameter: float, num_to_add: int, num_current: int):
    print("_initializeParticles diameter:",diameter)
    for i in range(num_to_add):
        particles.initializeParticle(i + num_current, diameter)

def _addParticleArray(particles, pos_arr):
    num_to_add = len(pos_arr)
    print('Number of particles added:', num_to_add)
    num_current = particles.particle_count[None]
    pos_arr_ti = ti.Vector.field(3, float, num_to_add)
    pos_arr_ti.from_numpy(pos_arr)
    _addPositionField(particles, pos_arr_ti, num_to_add, num_current)
    particles.particle_count[None] = num_to_add + num_current

def addMatrix(particles, matrix, min_corner, spacing):
    if len(matrix.shape) != 3:
        raise Exception('addMatrix Error: wrong matrix dimension')
    index = np.where(matrix == True)
    pos_arr = np.stack(index, axis=1) * spacing + min_corner
    _addParticleArray(particles, pos_arr)

# min_corner:corner with smallest coordination in all axis, max_corner vise versa
def addCube(particles, min_corner, max_corner, spacing):
    matrix_shape, padding = _helpCentering(min_corner, max_corner, spacing)
    addMatrix(particles, np.ones(matrix_shape, dtype=np.bool_), min_corner + padding, spacing)

#centered on y-axis
def addCylinder(particles, radius, bottom, top, spacing):
    min_corner = [-radius, bottom, -radius]
    max_corner = [radius, top, radius]
    matrix_shape, padding = _helpCentering(min_corner, max_corner, spacing)
    arr = np.zeros(matrix_shape, dtype=np.bool_)
    for x in range(matrix_shape[0]):
        for z in range(matrix_shape[2]):
            if (x * spacing - radius + padding[0]) ** 2 + (z * spacing - radius + padding[2]) ** 2 <= radius ** 2:
                arr[x, :, z] = True
    addMatrix(particles, arr, min_corner + padding, spacing)

def _addParticleArray_adaptive(particles, pos_arr, spacing):
    num_to_add = len(pos_arr)
    print('Number of particles added:', num_to_add)
    num_current = particles.particle_count[None]
    pos_arr_ti = ti.Vector.field(3, float, num_to_add)
    pos_arr_ti.from_numpy(pos_arr)
    _initializeParticles(particles, spacing, num_to_add, num_current)
    _addPositionField(particles, pos_arr_ti, num_to_add, num_current)
    particles.particle_count[None] = num_to_add + num_current

def addMatrix_adaptive(particles, matrix, min_corner, spacing, relaxation = 1.0):
    if len(matrix.shape) != 3:
        raise Exception('addMatrix Error: wrong matrix dimension')
    index = np.where(matrix == True)
    pos_arr = np.stack(index, axis=1) * spacing * relaxation + min_corner
    _addParticleArray_adaptive(particles, pos_arr, spacing)

#centered on y-axis
def addCylinder_adaptive(particles, radius, bottom, top, spacing):
    min_corner = [-radius, bottom, -radius]
    max_corner = [radius, top, radius]
    matrix_shape, padding = _helpCentering(min_corner, max_corner, spacing)
    arr = np.zeros(matrix_shape, dtype=np.bool_)
    for x in range(matrix_shape[0]):
        for z in range(matrix_shape[2]):
            if (x * spacing - radius + padding[0]) ** 2 + (z * spacing - radius + padding[2]) ** 2 <= radius ** 2:
                arr[x, :, z] = True
    addMatrix_adaptive(particles, arr, min_corner + padding, spacing, 1.0)

# min_corner:corner with smallest coordination in all axis, max_corner vise versa
def addCube_adaptive(particles, min_corner, max_corner, spacing, relaxation = 1.0):
    matrix_shape, padding = _helpCentering(min_corner, max_corner, spacing * relaxation)
    addMatrix_adaptive(particles, np.ones(matrix_shape, dtype=np.bool_), min_corner + padding, spacing, relaxation)


def read_ply_adaptive(filename):
    obj_ply = PlyData.read(filename)
    obj_verts = obj_ply['vertex'].data
    verts_array = np.array([[x, y, z, vx, vy, vz, d] for x, y, z, vx, vy, vz, d in obj_verts])
    return verts_array

@ti.kernel
def _initializeParticles_adaptive(particles: ti.template(), diameter: ti.template(), num_to_add: int, num_current: int):
    for i in range(num_to_add):
        particles.initializeParticle(i + num_current, diameter[i])

def add_from_ply_adaptive(filename, particles):
    verts_array = read_ply_adaptive(filename)
    pos_arr = verts_array[:, 0:3]
    diam_arr = verts_array[:, 3]

    num_to_add = len(pos_arr)
    print('Number of particles added:', num_to_add)
    num_current = particles.particle_count[None]
    pos_arr_ti = ti.Vector.field(3, float, num_to_add)
    pos_arr_ti.from_numpy(pos_arr)
    diam_arr_ti = ti.field(float, num_to_add)
    diam_arr_ti.from_numpy(diam_arr)    
    _initializeParticles_adaptive(particles, diam_arr_ti, num_to_add, num_current)
    _addPositionField(particles, pos_arr_ti, num_to_add, num_current)
    particles.particle_count[None] = num_to_add + num_current

# writes position and diameter
def write_ply_adaptive(filename, particles):
    num = particles.particle_count[None]
    pos = particles.position.to_numpy()
    diam = particles.diameter.to_numpy()
    vel = particles.velocity.to_numpy()

    list_pos = []
    for i in range(num):
        # position
        pos_tmp = [pos[i, 0], pos[i, 1], pos[i, 2], vel[i, 0], vel[i, 1], vel[i, 2]]
        # diameter
        pos_tmp.append(diam[i])

        list_pos.append(tuple(pos_tmp))

    data_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4')]
    data_type.append(('d','f4'))

    np_pos = np.array(list_pos, dtype=data_type)
    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData([el_pos], config.io.ply_write_text).write(filename)
    print('ply wrote to:', filename)

# PLY I/O for non-adaptive sph

def read_ply(filename):
    obj_ply = PlyData.read(filename)
    obj_verts = obj_ply['vertex'].data
    verts_array = np.array([[x, y, z, vx, vy, vz] for x, y, z, vx, vy, vz in obj_verts])
    return verts_array


def add_from_ply(filename, particles):
    verts_array = read_ply(filename)
    pos_arr = verts_array[:, 0:3]

    num_to_add = len(pos_arr)
    print('Number of particles added:', num_to_add)
    num_current = particles.particle_count[None]
    pos_arr_ti = ti.Vector.field(3, float, num_to_add)
    pos_arr_ti.from_numpy(pos_arr)

    _addPositionField(particles, pos_arr_ti, num_to_add, num_current)
    particles.particle_count[None] = num_to_add + num_current

# writes position and diameter
def write_ply(filename, particles):
    num = particles.particle_count[None]
    pos = particles.position.to_numpy()
    vel = particles.velocity.to_numpy()

    list_pos = []
    for i in range(num):
        pos_tmp = [pos[i, 0], pos[i, 1], pos[i, 2], vel[i, 0], vel[i, 1], vel[i, 2]] # data to write

        list_pos.append(tuple(pos_tmp))

    data_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4')] # name of data to write

    np_pos = np.array(list_pos, dtype=data_type)
    el_pos = PlyElement.describe(np_pos, 'vertex')
    PlyData([el_pos]).write(filename)
    print('ply wrote to:', filename)
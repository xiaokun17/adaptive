import taichi
from ..rigid.sphereContainer import sphereContainer
from ..rigid.DiscreteSDFRigid import DiscreteSDFRigid
from ..rigid.plane import plane
from ..rigid.boxContainer import boxContainer
from ..rigid.cylinderContainer import cylinderContainer
import math
from ..util.transformationMatrix import *
from ..util import particleOperation
import json
import os

class sph:
    adaptive = True # enable adaptivity
    enable_surface_adaptive = True # enable surface-based adaptivity (for adaptive)
    wake_preserve_time = 1.0 # wake flow preservation duration (for adaptive)
    pause_frame = -1 # frame to pause simulation, -1 for no pausing
    max_particle_count = int(1e5) # max number of particles
    max_original_particle_count = int(1e4) # max number of original particles (for adaptive)
    diameter = 0.2 # particle diameter (for non-adaptive)
    max_diameter = diameter # max particle diameter (for adaptive)
    max_adaptive_ratio = 32.0 # max adaptive ratio (for adaptive)
    min_diameter = max_diameter / (max_adaptive_ratio ** (1.0 / 3)) # min particle diameter (for adaptive)
    cfl_factor = 0.5 # factor for CFL condition
    gravity = [0.0,-9.8,0.0] # gravity
    sound_speed = 100.0 # speed of sound (for WCSPH)
    wc_gamma = 7 # WCSPH gamma (for WCSPH)
    rest_density = 1000.0 # rest density of the fluid
    bound_box_sidelength = 2.0 # default boundary box sidelength (for simple box boundary)
    bound_sphere_radius = 1.5 # default boundary sphere radius (for sphere SDF boundary)
    sdf_bound_friction = 0.1 # default boundary friction (for SDF boundary)
    dynamic_viscosity = 20 # dynamic viscosity
    simulation_space_min_corner = [-1.8, -1.8, -1.8] # min coordinates for simulation space
    simulation_space_max_corner = [1.8, 1.8, 1.8] # max coordinates for simulation space
    max_neighbor_count = 2000 # max neighbor count for any particle
    neighbor_list_blocks = 16 # not supported for now
    neighbor_use_sparse = False # not supported for now
    split_pattern_count = 32 # number of split patterns (for adaptive)
    initial_temporal_blend_factor = 0.9 # initial temporal blend factor after splitting (for adaptive)
    initial_temporal_blend_factor_merge = 0.2 # initial temporal blend factor after merging (for adaptive)
    initial_temporal_blend_factor_redistribute = 0.5 # initial temporal blend factor after redistribution (for adaptive)
    temporal_blend_factor_decrease = 0.01 # amount to decrease temporal blend factor per time step (for adaptive)
    local_viscosity_factor = 1.0 # local viscosity coefficient (for adaptive)
    surface_tension_coefficient = 0.1 # surface tension coefficient
    min_surface_level_set = -9.0 * max_diameter # min value for surface level set (for adaptive)
    optimize_splitting_error = False # enable simple optimization for splitting, will cause initialization to be very slow (~15min) (for adaptive)

class gui:
    enabled = True # enable gui
    res = (1080, 1080) # gui resolution
    camera_pos = (4.0, 0.25, 0.0) # initial camara position
    camera_lookat = (0.0,0.0,0.0) # initial camara lookat
    camera_fov = 55 # camera fov
    background_color = (0.2, 0.2, 0.6) # background color
    particle_min_color = ti.Vector([0.9,0.9,0.2]) # particle color for min display value
    particle_max_color = ti.Vector([0.3,0.1,0.3]) # particle color for max display value
    radius = sph.min_diameter / 2.0 if sph.adaptive else sph.diameter / 2 # particle radius
    color_config = json.load(open(os.path.join(os.getcwd(), "Taichi_SPH", "util", "colorConfig.json"))) # color config file
    display_fields = ["mass", "pressure", "density", "temporal_blend_factor", "velocity", "surface_level_set"] # fields to display
    show_cross_section = True # show cross section of the fluid instead of all the fluid

class ti:
    arch = taichi.gpu # taichi architecture to use
    device_mem = 2 # amount of device memory to allocate
    fp = taichi.f32 # float-point precision
    ip = taichi.i32 # integer precision

class io:
    ply_write_text = False # write ply in plain text instead of binary
    write_snap_file = False # write snapshots for recent timesteps
    snap_file_limit = 100 # max amount of snapshots to keep

# define the boundary objects to be used
# surface type:
#   0: no boundary refinement near this object
#   2: enable boundary refinement near this object
def makeBoundaryList():
    boundary_list = [sphereContainer(sph.bound_sphere_radius, surface_type=0)]
    return boundary_list

# initialize boundary after sph initialization, befor any sph step
@taichi.kernel
def initializeBoundary(sph: taichi.template()):
    bound = taichi.static(sph.boundary_list)

# initialize fluid after sph initialization, befor any sph step
def initializeFluid(sph_data):
    particleOperation.addCube_adaptive(sph_data,[-0.7,-0.7,-0.7],[0.7,0.7,0.7],sph.max_diameter)
    print(sph_data.particle_count)


# performed every timestep
@taichi.kernel
def manageBoundaryTransform(sph_data: taichi.template()):
    bound, dt = taichi.static(sph_data.boundary_list, sph_data.time_step)

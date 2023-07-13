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
    enable_surface_adaptive = False # enable surface-based adaptivity (for adaptive)
    wake_preserve_time = 1.0 # wake flow preservation duration (for adaptive)
    pause_frame = -1 # frame to pause simulation, -1 for no pausing
    max_particle_count = int(1e6) # max number of particles
    max_original_particle_count = int(3e5) # max number of original particles (for adaptive)
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
    bound_sphere_radius = 3.0 # default boundary sphere radius (for sphere SDF boundary)
    sdf_bound_friction = 0.1 # default boundary friction (for SDF boundary)
    dynamic_viscosity = 20 # dynamic viscosity
    simulation_space_min_corner = [-11.0, -2.0, -6.0] # min coordinates for simulation space
    simulation_space_max_corner = [11.0, 4.0, 6.0] # max coordinates for simulation space
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
    optimize_splitting_error = True # enable simple optimization for splitting, will cause initialization to be very slow (~15min) (for adaptive)

class gui:
    enabled = True # enable gui
    res = (1080, 1080) # gui resolution
    camera_pos = (0.0, 10.0, 30.0) # initial camara position
    camera_lookat = (0.0,0.0,0.0) # initial camara lookat
    camera_fov = 55 # camera fov
    background_color = (0.2, 0.2, 0.6) # background color
    particle_min_color = ti.Vector([0.9,0.9,0.2]) # particle color for min display value
    particle_max_color = ti.Vector([0.3,0.1,0.3]) # particle color for max display value
    radius = sph.min_diameter / 2.0 if sph.adaptive else sph.diameter / 2 # particle radius
    color_config = json.load(open(os.path.join(os.getcwd(), "Taichi_SPH", "util", "colorConfig.json"))) # color config file
    display_fields = ["mass", "pressure", "density", "temporal_blend_factor", "velocity", "surface_level_set"] # fields to display
    show_cross_section = False # show cross section of the fluid instead of all the fluid

class ti:
    arch = taichi.gpu # taichi architecture to use
    device_mem = 6 # amount of device memory to allocate
    fp = taichi.f32 # float-point precision
    ip = taichi.i32 # integer precision

class io:
    ply_write_text = False # write ply in plain text instead of binary
    write_snap_file = False # write snapshots for recent timesteps
    snap_file_limit = 100 # max amount of snapshots to keep

# parameters for scene initialization
ship_l = 3.2
ship_w = 1.6
ship_h = 1.4
ship_water = 0.8
ship_l_padding = 0.6

particle_r = sph.diameter / 2

fluid_depth = 1.8
river_w = 5.0
river_l = 20.0 

box_min_x = -river_l / 2
box_max_x = river_l / 2
box_min_y = -fluid_depth
box_max_y = ship_h * 2
box_min_z = -river_w / 2
box_max_z = river_w / 2

fluid_init_min_x = box_min_x + particle_r
fluid_init_max_x = box_max_x - particle_r
fluid_init_min_y = box_min_y + particle_r
fluid_init_max_y = fluid_init_min_y + fluid_depth
fluid_init_min_z = box_min_z + particle_r
fluid_init_max_z = box_max_z - particle_r

ship_init_y = fluid_init_max_y + ship_water
ship_init_x = box_min_x + ship_l / 2 + ship_l_padding

ship_final_y = -ship_water

v_down = 1.0
v_x_max = 10.0

time_acc = 0.25
a_x = v_x_max + time_acc

time_wait = 0.0
time_down = time_wait + (ship_init_y - ship_final_y) / 1.0 # adjust translation speed
time_rest = time_down + 1.0


# make a boundary list used in sph initialization
def makeBoundaryList():
    boundary_list = [
        boxContainer(
            taichi.Vector([box_min_x, box_min_y, box_min_z]), taichi.Vector([box_max_x, box_max_y, box_max_z]), 
            surface_type=0
        ),
        DiscreteSDFRigid(
            os.path.join(os.getcwd(), "Taichi_SPH", "rigid_ply", "ship_0_025.ply"), 
            0.025,
            surface_type=2
        )
    ]
    boundary_list[0].friction = 0.0
    return boundary_list

# define the boundary objects to be used
# surface type:
#   0: no boundary refinement near this object
#   2: enable boundary refinement near this object
@taichi.kernel
def initializeBoundary(sph: taichi.template()):
    bound = taichi.static(sph.boundary_list)
    bound[1].translate(translationMatrix(ship_init_x, ship_init_y, 0.0))
    bound[1].updatePreviousTransformation()
    print("time down: ", time_down, ", time rest: ", time_rest)

# initialize fluid after sph initialization, befor any sph step
def initializeFluid(sph_data):
    particleOperation.addCube_adaptive(sph_data, [fluid_init_min_x, fluid_init_min_y, fluid_init_min_z], [fluid_init_max_x, fluid_init_max_y, fluid_init_max_z], sph.max_diameter)

# performed every timestep
@taichi.kernel
def manageBoundaryTransform(sph_data: taichi.template()):
    bound, dt, t = taichi.static(sph_data.boundary_list, sph_data.time_step, sph_data.time_passed)
    t[None] += dt[None]
    if t[None] < time_wait:
        bound[1].updatePreviousTransformation()
    elif t[None] < time_down:
        bound[1].updatePreviousTransformation()
        bound[1].translate(translationMatrix(0.0, -v_down * dt[None], 0.0)) # adjust translation speed
    elif t[None] < time_rest:
        bound[1].updatePreviousTransformation()
    else:
        bound[1].updatePreviousTransformation()
        bound[1].translate(translationMatrix(taichi.max(v_x_max, (t[None] - time_rest) * a_x) * dt[None], 0.0, 0.0)) # adjust translation speed


    


# Setting up scene

The scene is set up through a configuration file in the ```config``` folder, such as ```simple_adaptive.py```.

## Choosing a file

To choose a configuration file to use for the simulation, open ```config\__init.py__```, change the following line of code:\
```from .simple_adaptive import *```\
to\
```from .your_config_file_name import *```

## Initializing fluid and boundary

### Fluid
Fluid is initialized in the ```initializeFluid``` function in the configuration file. Use ```addCube_adaptive``` to initialize fluid cubes in the scene.

```
def initializeFluid(sph_data):
    particleOperation.addCube_adaptive(sph_data,[-0.7,-0.7,-0.7],[0.7,0.7,0.7],sph.max_diameter)
    print(sph_data.particle_count)
```

### Boundary

#### Initialize boundary objects

Boundary objects are defined in the ```makeBoundaryList``` function, and added to ```boundary_list```.

```
def makeBoundaryList():
    boundary_list = [sphereContainer(sph.bound_sphere_radius, surface_type=0)]
    return boundary_list
```

#### Configure boundary movement

Boundary movement is configured in ```manageBoundaryTransform```. This function is called each time step by the simulator to set the transform of boundary objects.

```
@taichi.kernel
def manageBoundaryTransform(sph_data: taichi.template()):
    bound, dt, t = taichi.static(sph_data.boundary_list, sph_data.time_step, sph_data.time_passed)
    t[None] += dt[None]
    bound[1].updatePreviousTransformation()
    bound[1].translate(translationMatrix(0.0, -v_down * dt[None], 0.0)) # move the boundary object
```

## Config parameters

### class sph

|variable|description|
|-|-|
|adaptive |enable adaptivity|
|enable_surface_adaptive |enable surface-based adaptivity (for adaptive)|
|wake_preserve_time |wake flow preservation duration (for adaptive)|
|pause_frame |frame to pause simulation, -1 for no pausing|
|max_particle_count |max number of particles|
|max_original_particle_count |max number of original particles (for adaptive)|
|diameter |particle diameter (for non-adaptive)|
|max_diameter |max particle diameter (for adaptive)|
|max_adaptive_ratio |max adaptive ratio (for adaptive)|
|min_diameter |min particle diameter (for adaptive)|
|cfl_factor |factor for CFL condition|
|gravity |gravity|
|sound_speed |speed of sound (for WCSPH)|
|wc_gamma |WCSPH gamma (for WCSPH)|
|rest_density |rest density of the fluid|
|bound_box_sidelength |default boundary box sidelength (for simple box boundary)|
|bound_sphere_radius |default boundary sphere radius (for sphere SDF boundary)|
|sdf_bound_friction |default boundary friction (for SDF boundary)|
|dynamic_viscosity |dynamic viscosity|
|simulation_space_min_corner |min coordinates for simulation space|
|simulation_space_max_corner |max coordinates for simulation space|
|max_neighbor_count |max neighbor count for any particle|
|neighbor_list_blocks |not supported for now|
|neighbor_use_sparse |not supported for now|
|split_pattern_count |number of split patterns (for adaptive)|
|initial_temporal_blend_factor |initial temporal blend factor after splitting (for adaptive)|
|initial_temporal_blend_factor_merge |initial temporal blend factor after merging (for adaptive)|
|initial_temporal_blend_factor_redistribute |initial temporal blend factor after redistribution (for adaptive)|
|temporal_blend_factor_decrease |amount to decrease temporal blend factor per time step (for adaptive)|
|local_viscosity_factor |local viscosity coefficient (for adaptive)|
|surface_tension_coefficient |surface tension coefficient|
|min_surface_level_set |min value for surface level set (for adaptive)|
|optimize_splitting_error |enable simple optimization for splitting (for adaptive), will cause initialization to be slow (~15min) |

### class gui

|variable|description|
|-|-|
|enabled |enable gui|
|res |gui resolution|
|camera_pos |initial camara position|
|camera_lookat |initial camara lookat|
|camera_fov |camera fov|
|background_color |background color|
|particle_min_color |particle color for min display value|
|particle_max_color |particle color for max display value|
|radius |particle radius|
|color_config |color config file|
|display_fields |fields to display|
|show_cross_section |show cross section of the fluid instead of all the fluid|

### class ti

|variable|description|
|-|-|
|arch |taichi architecture to use|
|device_mem |amount of device memory to allocate|
|fp |float-point precision|
|ip |integer precision|

### class io

|variable|description|
|-|-|
|ply_write_text |write ply in plain text instead of binary|
|write_snap_file |write snapshots for recent timesteps|
|snap_file_limit |max amount of snapshots to keep|
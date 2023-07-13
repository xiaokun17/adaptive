import taichi as ti
from .. import config
from ..util.kernel import *
# from ...util.misc import *
from .. import global_var
from ..neighborSearch.IndexArray import IndexArray
from ..neighborSearch.NeighborListAdaptive import NeighborListAdaptive
from ..patterns import pattern
from ..rigid.sphereContainer import sphereContainer
from ..rigid.DiscreteSDFRigid import DiscreteSDFRigid
import numpy as np

# from ._adaptive import initializeSplitParticle,initializeOriginalParticle,splitParticle,estimateDensity_ori,blendDensity,calculateOriginalVelocity,blendVelocity,updateOriginalPosition,decrementTemporalBlendFactor,calculateCorrectiveFactor,replaceOriginalParticle,removeOriginalParticles,simpleSizingFunction,handleSplitting,originalDensityFromSDFBoundary,redistributeToParticle,mergeToParticle,computeSizeClass,markPotentialRedistributePartners,finalizeRedistributePartners,redistributeParticles,updateRedistributedParticle,markPotentialMergePartners,finalizeMergePartners,mergeParticles,replaceParticle,removeToDeleteParticles,updateMergedParticle,updateSplitParticle,_computeSplitError,splitParticleWithOptimize,handleSplittingWithOptimize,initializeSplitParticleWithRot,localViscosityAdvection,calculateSumWeight,detectSurface,surfacePropagationStep,smoothSurfaceLevelSet,surfaceSizingFunction
# from ._basic import CFLCondition,estimateDensity,gravityAdvection,calculatePressure,updateVelocityFromPressure,updatePosition,viscosityAdvection,clampDensity,estimateSurfaceNormal,surfaceTensionAdvection
# from ._boundary import handleBoundary,densityFromSphereSDFBoundary,boundaryPressure,boundaryFriction,updateVelocityFromBoundaryPressureAndFriction
from ._adaptive import *
from ._basic import *
from ._boundary import *


@ti.data_oriented
class WCSPH:
    def __init__(self, boundary_list = [sphereContainer(config.sph.bound_sphere_radius)]):
        max_n = config.sph.max_particle_count
        max_o = config.sph.max_original_particle_count
        pattern_n = config.sph.split_pattern_count
        # ===================== constants =====================
        # constants
        self.max_particle_count = max_n
        self.rest_density = config.sph.rest_density
        self.max_support_radius = config.sph.max_diameter * 2
        self.min_diameter = config.sph.min_diameter
        self.max_mass = config.sph.max_diameter ** 3 * self.rest_density
        self.min_mass = config.sph.min_diameter ** 3 * self.rest_density
        self.max_surface_distance = config.sph.bound_sphere_radius
        self.max_adaptive_ratio = 1.0 / config.sph.max_adaptive_ratio
        self.gravity = ti.Vector(config.sph.gravity)
        self.sound_speed = config.sph.sound_speed
        self.gamma = config.sph.wc_gamma
        self.bound_box_sidelength = config.sph.bound_box_sidelength
        self.enable_surface_adaptive = config.sph.enable_surface_adaptive
        self.wake_preserve_time = config.sph.wake_preserve_time
        self.optimize_splitting_error = config.sph.optimize_splitting_error

        # self.bound_sphere_radius = config.sph.bound_sphere_radius
        self.dynamic_viscosity = config.sph.dynamic_viscosity
        self.sdf_bound_friction = config.sph.sdf_bound_friction
        self.cfl_factor = config.sph.cfl_factor
        self.split_pattern_count = pattern_n
        self.initial_temporal_blend_factor = config.sph.initial_temporal_blend_factor
        self.initial_temporal_blend_factor_merge = config.sph.initial_temporal_blend_factor_merge
        self.initial_temporal_blend_factor_redistribute = config.sph.initial_temporal_blend_factor_redistribute
        self.temporal_blend_factor_decrease = config.sph.temporal_blend_factor_decrease
        self.local_viscosity_factor = config.sph.local_viscosity_factor
        self.surface_tension_coefficient = config.sph.surface_tension_coefficient
        self.min_surface_level_set = config.sph.min_surface_level_set

        # constant fields
        self.pattern = ti.Vector.field(3, float, (pattern_n,pattern_n))
        initializePattern(self, pattern_n)
        rot_pattern_arr = np.load(global_var.root_path + r'\patterns\rotationPattern.npy')
        self.rotation_pattern_count = len(rot_pattern_arr)
        self.rotation_pattern = ti.Matrix.field(3, 3, float, self.rotation_pattern_count)
        self.rotation_pattern.from_numpy(rot_pattern_arr)

        # ===================== variables =====================
        # shared variables
        self.particle_count = ti.field(int,())
        self.time_step = ti.field(float,())
        self.particle_count[None] = 0
        self.time_step[None] = config.sph.min_diameter / config.sph.sound_speed
        self.exit_flag = ti.field(int, ())
        self.exit_flag[None] = 0
        self.merged_count = ti.field(int,())
        self.split_count = ti.field(int,())
        self.redistributed_count = ti.field(int,())
        self.merged_count [None] = 0
        self.split_count [None] = 0
        self.redistributed_count [None] = 0
        self.propagation_flag = ti.field(int,())
        self.propagation_flag[None] = 0

        # per-particle variables
        # need initialization
        self.diameter = ti.field(float, max_n)
        self.mass = ti.field(float, max_n)
        self.support_radius = ti.field(float, max_n)
        self.position = ti.Vector.field(3, float, max_n)
        self.velocity = ti.Vector.field(3, float, max_n)
        self.temporal_blend_factor = ti.field(float, max_n)
        self.original_particle_id = ti.field(int, max_n) # initial value: -1
        self.merge_partner = ti.field(int, max_n) # initial value: -1
        self.partner_count = ti.field(int, max_n) # initial value: 0
        self.to_delete = ti.field(int, max_n) # initial value: 0
        self.surface_level_set = ti.field(float, max_n)

        # don't need initialization
        self.density = ti.field(float, max_n)
        self.pressure = ti.field(float, max_n)
        self.viscosityAcceleration = ti.Vector.field(3, float, max_n)
        self.boundary_pressure_acceleration = ti.Vector.field(3, float, max_n) 
        self.adaptive_corrective_factor = ti.field(float, max_n)
        self.optimal_mass = ti.field(float, max_n)
        self.size_class = ti.field(int, max_n) #need to be updated in split/merge
        self.surface_normal = ti.Vector.field(3, float, max_n)
        self.sum_weight = ti.field(float, max_n)
        self.surface_level_set_smooth = ti.field(float, max_n) # used in smoothing
        self.surface_flag = ti.field(int, max_n) # 0: interior, 1: surface or nearSurface

        # classes
        self.neighbor_search = IndexArray()
        self.neighbor_list = NeighborListAdaptive()
        self.original = self.OriginalParticleData()
        self.boundary_list = boundary_list
        self.boundary_count = len(self.boundary_list)
        # helper for deletion
        self.delete_helper_list = ti.field(int, max_n)
        self.delete_helper_count = ti.field(int, ())
        # new surface method's mark
        self.mark_remain_time = ti.field(float, max_n)
        self.marked_level_set = ti.field(float, max_n)
        # scene helper
        self.time_passed = ti.field(float, ())
        self.time_passed[None] = 0.0


    # data structure for original particles used in temporal blending
    @ti.data_oriented
    class OriginalParticleData:
        def __init__(self, max_original_particle_count = config.sph.max_original_particle_count):
            max_o = config.sph.max_original_particle_count

            # shared variable
            self.particle_count = ti.field(int,())
            self.particle_count[None] = 0

            # per original particle variable
            self.density = ti.field(float, max_o)
            self.position = ti.Vector.field(3, float, max_o)
            self.velocity = ti.Vector.field(3, float, max_o)
            self.mass = ti.field(float, max_o)
            self.support_radius = ti.field(float, max_o)
            self.children_count = ti.field(int, max_o) # updated when computing original velocity
            self.new_id = ti.field(int, max_o) # helper for deleting original particles

            # class
            self.neighbor_list = NeighborListAdaptive(max_particle_count = max_o)
            # helper for deletion
            self.delete_helper_list = ti.field(int, max_o)
            self.delete_helper_count = ti.field(int, ())

    @ti.func
    def initializeParticle(self, i, diameter):
        self.diameter[i] = diameter
        self.mass[i] = diameter ** 3 * self.rest_density
        self.support_radius[i] = diameter * 2
        self.position[i] = ti.Vector([0.0, 0.0, 0.0])
        self.velocity[i] = ti.Vector([0.0, 0.0, 0.0])
        self.temporal_blend_factor[i] = 0.0
        self.original_particle_id[i] = -1
        self.merge_partner[i] = -1
        self.partner_count[i] = 0
        self.to_delete[i] = 0
        self.surface_level_set[i] = 0.0
        self.adaptive_corrective_factor[i] = 1.0
        self.mark_remain_time[i] = 0.0

def initializePattern(self, n):
    pattern_arr = np.zeros((n,n,3), np.float32)
    for i in range(n):
        for j in range(i + 1):
            for k in range(3):
                pattern_arr[i, j, k] = pattern.pattern_list[i][j][k]
    self.pattern.from_numpy(pattern_arr)

def printParticle(self, i):
    print('id:', i, 'diameter:', self.diameter[i], 'mass:', self.mass[i], 'support_radius:', self.support_radius[i], 'position:', self.position[i], 'velocity:', self.velocity[i], 'temporal_blend_factor:', self.temporal_blend_factor[i], 'original_particle_id:', self.original_particle_id[i], 'merge_partner:', self.merge_partner[i], 'partner_count:', self.partner_count[i], 'to_delete:', self.to_delete[i])

def printOriginalParticle(self, i):
    print('id:', i, 'density:', self.original.density[i], 'position:', self.original.position[i], 'velocity:', self.original.velocity[i], 'mass:', self.original.mass[i], 'support_radius:', self.original.support_radius[i], 'children_count:', self.original.children_count[i], 'new_id:', self.original.new_id[i])

# stuff that need asserting
@ti.func
def runtimeCheck(self):
    if self.original.particle_count[None] >= config.sph.max_original_particle_count:
        print('ERROR: original.particle_count exceeded')
    if self.particle_count[None] >= self.max_particle_count:
        print('ERROR: particle_count exceeded')
    if self.time_step[None] < 1e-5:
        print('WARNING: time step too small')
    # for i in range(self.particle_count[None]):
    #     for j in ti.static(range(self.boundary_count)):
    #         sdf = self.boundary_list[j].sdf(self.position[i])
    #         if sdf < -0.5 * self.diameter[i]:
    #             print('WARNING: solid penetration, solid:', j, ', sdf:', sdf)

@ti.func
def sphFistOfNextStep(self):
    prop_flg = ti.static(self.propagation_flag)
    calculateSumWeight(self)
    detectSurface(self)
    prop_flg[None] = 1

@ti.kernel
def adaptivePrepare(self:ti.template()):
    self.neighbor_search.establishNeighbors(self.position, self.particle_count)
    self.neighbor_list.establishNeighborList(self.neighbor_search,self.position, self.particle_count, self.max_support_radius, self.support_radius)
    estimateDensity(self)

    sphFistOfNextStep(self)

@ti.kernel
def surfacePropagationKernel(self:ti.template()):
    prop_flg = ti.static(self.propagation_flag)
    prop_flg[None] = 0
    surfacePropagationStep(self)

def surfacePropagation(self):
    while self.propagation_flag[None] == 1:
        surfacePropagationKernel(self)

@ti.kernel
def adaptiveStep(self:ti.template()):
    '''compute sizing function and size class'''  
    # simpleSizingFunction(self)
    smoothSurfaceLevelSet(self)
    surfaceSizingFunction(self)
    computeSizeClass(self)

    ''' perform splitting, merging and mass redistribution '''
    if self.optimize_splitting_error:
        handleSplittingWithOptimize(self)
    else:
        handleSplitting(self)
    removeToDeleteParticles(self)

    markPotentialRedistributePartners(self)
    finalizeRedistributePartners(self)
    redistributeParticles(self)

    markPotentialMergePartners(self)
    finalizeMergePartners(self)
    mergeParticles(self)
    removeToDeleteParticles(self)

    ''' checking '''
    runtimeCheck(self)

    ''' neighbor search '''
    self.neighbor_search.establishNeighbors(self.position, self.particle_count)
    self.neighbor_list.establishNeighborList(self.neighbor_search,self.position, self.particle_count, self.max_support_radius, self.support_radius)
    self.original.neighbor_list.establishNeighborList_ori(self.neighbor_search,self.original.position, self.position, self.original.particle_count, self.max_support_radius, self.original.support_radius, self.support_radius)
    
    ''' CFL condition '''
    CFLCondition(self)

    ''' density estimation'''
    estimateDensity(self)
    densityFromSphereSDFBoundary(self)

    ''' density blending'''
    estimateDensity_ori(self)
    originalDensityFromSDFBoundary(self)
    blendDensity(self)

    # clampDensity(self)

    ''' advection '''
    gravityAdvection(self)
    estimateSurfaceNormal(self)
    surfaceTensionAdvection(self)
    # viscosityAdvection(self)
    localViscosityAdvection(self)

    ''' pressure '''
    calculateCorrectiveFactor(self)
    calculatePressure(self)
    updateVelocityFromPressure(self)
    updateVelocityFromBoundaryPressureAndFriction(self) # pressure and friction

    ''' additional boundary stuff '''
    # handleBoundary(self)

    ''' velocity blending '''
    calculateOriginalVelocity(self)
    blendVelocity(self)

    ''' update temporal blending '''
    updateOriginalPosition(self)
    removeOriginalParticles(self)
    decrementTemporalBlendFactor(self)

    ''' update position '''
    updatePosition(self)

    ''' first part of next step '''
    sphFistOfNextStep(self)

@ti.kernel
def noAdaptiveStep(self:ti.template()):
    self.neighbor_search.establishNeighbors(self.position, self.particle_count)
    self.neighbor_list.establishNeighborList(self.neighbor_search,self.position, self.particle_count, self.max_support_radius, self.support_radius)
    CFLCondition(self)
    estimateDensity(self)
    densityFromSphereSDFBoundary(self)
    gravityAdvection(self)
    estimateSurfaceNormal(self)
    surfaceTensionAdvection(self)
    viscosityAdvection(self)
    calculatePressure(self)
    updateVelocityFromPressure(self)
    updateVelocityFromBoundaryPressureAndFriction(self)
    # handleBoundary(self)
    updatePosition(self)
    
    # runtimeCheck(self)

def sphPrepare(self):
    if config.sph.adaptive:
        adaptivePrepare(self)
    else:
        pass

def sphStep(self):
    if config.sph.adaptive:
        surfacePropagation(self)
        adaptiveStep(self)
    else:
        noAdaptiveStep(self)
    
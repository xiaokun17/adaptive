import taichi as ti
from . import global_var
from .Gui import Gui
from .sph import WCSPH, sphPrepare, sphStep
from .util import particleOperation
from .util import colorScheme
from .util.snap import SnapRecorder
import numpy as np
import math
from . import config
from .config import makeBoundaryList, initializeBoundary, manageBoundaryTransform, initializeFluid

ti.init(
    arch = config.ti.arch, 
    device_memory_GB=config.ti.device_mem, 
    default_fp=config.ti.fp, 
    default_ip=config.ti.ip
)

gui = None
if config.gui.enabled:
    gui = Gui()
snapRecorder = SnapRecorder()
snapRecorder.clearRecord()
sph = WCSPH(makeBoundaryList())

initializeBoundary(sph)
initializeFluid(sph)

print('min_surface_level_set:', sph.min_surface_level_set)
sphPrepare(sph)

print('prepare done')

def frame_update():
    print('particle count:', sph.particle_count[None], 'original particle count:', sph.original.particle_count[None])
    global_var.frame += 1
    if global_var.frame == config.sph.pause_frame:
        print('simulation paused at: ', global_var.frame)
        global_var.run_sph = False
    if global_var.write_file:
        particleOperation.write_ply_adaptive(global_var.root_path + r'\ply\frame_' + str(global_var.frame) +'.ply', sph)
    
    handle_gui_output()

def step_update():
    global_var.simulation_time += sph.time_step[None]
    global_var.step_id += 1
    manageBoundaryTransform(sph)
    sphStep(sph)
    if global_var.write_snap_file:
        snapRecorder.addFieldRecord(sph)

def handle_gui_output():
    if config.gui.enabled and gui.refresh_window:
        if gui.show_cross_section:
            gui.addParticlesCrossSection(sph,colorScheme.scalar)
        else:
            gui.addParticles(sph,colorScheme.scalar)
    
    if config.gui.enabled:
        gui.show()

def handle_snap():
    if global_var.write_snap_file:
        if gui.load_prev_snap:
            snapRecorder.loadPrevFieldRecord(sph)
        elif gui.load_next_snap:
            snapRecorder.loadNextFieldRecord(sph)

global_var.frame = 0
while global_var.frame < 1000:
    if config.gui.enabled:
        gui.handleInput()
    if global_var.run_sph:
        if global_var.snap_read_flag:
            snapRecorder.loadNewestFieldRecord(sph)
        step_update()
        if global_var.simulation_time > global_var.frame / 24.0:
            frame_update()
    else:
        handle_snap()
        handle_gui_output()
    if config.gui.enabled:
        gui.clearTriggerInput()

print('Done simulation')
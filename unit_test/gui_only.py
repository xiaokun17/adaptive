import taichi as ti
from .. import global_var
from ..Gui import Gui
from ..sph import WCSPH, sphPrepare, sphStep
from ..util import particleOperation
from ..util import colorScheme
from ..util.snap import SnapRecorder
import numpy as np
import math
from .. import config
from ..config import makeBoundaryList, initializeBoundary, manageBoundaryTransform, initializeFluid

ti.init(
    arch = config.ti.arch, 
    device_memory_GB=config.ti.device_mem, 
    default_fp=config.ti.fp, 
    default_ip=config.ti.ip
)

gui = None
if config.gui.enabled:
    gui = Gui()

def frame_update():
    handle_gui_output()

def handle_gui_output():
    if config.gui.enabled:
        gui.show()


global_var.frame = 0
# while not sph.exit_flag[None] == 1:
while True:
    frame_update()

print('Done simulation')


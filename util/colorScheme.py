from .. import config
from ..Gui import Gui
import taichi as ti

@ti.func
def scalar(field, i, min_v, max_v):
    val = (field[i] - min_v) / (max_v - min_v)
    val = max(0.0, min(1.0, val))
    return config.gui.particle_min_color * (1 - val) + config.gui.particle_max_color * val
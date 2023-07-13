from . import config
import taichi as ti
import os

run_sph = True
write_file = True
write_snap_file = config.io.write_snap_file
snap_read_flag = False
simulation_time = 0.0
step_id = 0
frame = 0
root_path = os.getcwd() + r"\Taichi_SPH"
print("Path is: ", root_path)
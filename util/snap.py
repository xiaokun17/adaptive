import taichi as ti
import numpy as np
import os
from .. import config
from .. import global_var


# Define a function to save all Taichi fields to NumPy arrays
def save_fields(filename, my_fields):
    folder_name = os.path.join(os.getcwd(), "Taichi_SPH", "snap")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    fields_dict = {}
    for k, v in my_fields.__dict__.items():
        if isinstance(v, ti.Field):
            fields_dict[k] = v.to_numpy()
    fields_dict["STEP_ID"] = np.array([global_var.step_id])
    fields_dict["SIM_TIME"] = np.array([global_var.simulation_time])
    np.savez(os.path.join(folder_name, filename), **fields_dict)


# Define a function to read all Taichi fields from NumPy arrays
def load_fields(filename, my_fields):
    npzfile = np.load(os.path.join(os.getcwd(), "Taichi_SPH", "snap", filename + '.npz'))
    for k, v in npzfile.items():
        if k == "STEP_ID":
            print("step id:", v[0])
        elif k == "SIM_TIME":
            print("simulation time:", v[0])
        elif hasattr(my_fields, k) and isinstance(getattr(my_fields, k), ti.Field):
            getattr(my_fields, k).from_numpy(v)

class SnapRecorder():
    file_limit = config.io.snap_file_limit
    file_count = 0
    newest_file_id = -1
    loaded_file_id = -1

    def addFieldRecord(self, sph):
        self.file_count = min(self.file_count + 1, self.file_limit)
        self.newest_file_id = (self.newest_file_id + 1) % self.file_count
        self.loaded_file_id = self.newest_file_id
        save_fields(str(self.newest_file_id), sph)
    
    def loadPrevFieldRecord(self, sph):
        global_var.snap_read_flag = False
        if self.newest_file_id == -1: return
        self.loaded_file_id = (self.loaded_file_id - 1) % self.file_count
        if (self.loaded_file_id == self.newest_file_id):
            self.loaded_file_id += 1
        else:
            global_var.snap_read_flag = True
            load_fields(str(self.loaded_file_id), sph)
    
    def loadNextFieldRecord(self, sph):
        global_var.snap_read_flag = False
        if self.newest_file_id == -1: return
        if self.loaded_file_id == self.newest_file_id:
            return
        global_var.snap_read_flag = True
        self.loaded_file_id = (self.loaded_file_id + 1) % self.file_count
        load_fields(str(self.loaded_file_id), sph)

    def loadNewestFieldRecord(self, sph):
        global_var.snap_read_flag = False
        if self.newest_file_id == -1: return
        load_fields(str(self.newest_file_id), sph)
        self.loaded_file_id = self.newest_file_id

    def clearRecord(self):
        folder_path = os.path.join(os.getcwd(), "Taichi_SPH", "snap")
        # get all the files in the folder
        files = os.listdir(folder_path)
        # loop through the files and delete them
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

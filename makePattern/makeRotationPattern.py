from math import pi, cos, sin
import numpy as np
from .. import global_var
# example: rotation_matrix(config, angle_x, angle_y, angle_z)
def rotation_matrix_3(*args):
    m = None
    if len(args) == 3: #3d
        x, y, z = args
        rx = np.array([
            [1, 0,       0      ],
            [0, cos(x), -sin(x) ],
            [0, sin(x),  cos(x) ]
        ])
        ry = np.array([
            [ cos(y), 0, sin(y) ],
            [ 0,      1, 0      ],
            [-sin(y), 0, cos(y) ]
        ])
        rz = np.array([
            [cos(z), -sin(z), 0 ],
            [sin(z),  cos(z), 0 ],
            [0,       0,      1 ]
        ])
        m = np.matmul(rz,np.matmul(ry,rx))
    else:
        raise Exception('transformation ERROR: dimension mismatch')
    return m

def initializeAngles():
    angles = []
    # for i in range(-1, 2):
    #     x = pi / 4 * i
    #     for j in range(8):
    #         y = pi / 4 * i
    #         angles.append((x, y, 0))
    for j in range(4):
        y = pi / 2 * j
        angles.append((0.0, y, 0.0))
    for i in range(-1,2,2):
        x = pi / 2 * i
        angles.append((x, 0.0, 0.0))
    return angles

angles = initializeAngles()
rotation_patterns = np.zeros((len(angles), 3, 3), np.float32)
for i in range(len(angles)):
    rotation_patterns[i] = rotation_matrix_3(*angles[i]) 
    print(rotation_patterns[i])
print(rotation_patterns)
np.save(global_var.root_path + r'\rotationPattern', rotation_patterns)
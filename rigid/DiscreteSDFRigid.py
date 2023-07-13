import taichi as ti
import numpy as np
from plyfile import *
from ..util.misc import trilinearInterpolation
from ..util.transformationMatrix import *
import math
from .. import config

@ti.data_oriented
class DiscreteSDFRigid:
    def __init__(self, filename, spacing, surface_type = 0):
        verts_arr = self.readPly(filename)

        ''' sort verts_arr by position '''
        sort_indices = np.lexsort([verts_arr[:, i] for i in range(6, -1, -1)])
        verts_arr = np.array([verts_arr[sort_indices[i]] for i in range(len(verts_arr))])

        ''' get attributes from verts_arr ''' 
        pos_arr = verts_arr[:, 0:3]
        sdf_arr = verts_arr[:, 3]
        grad_arr = verts_arr[:, 4:7]

        ''' get xyz dimensions '''
        x = self._getAxisCount(pos_arr[:, 0], spacing)
        y = self._getAxisCount(pos_arr[:, 1], spacing)
        z = self._getAxisCount(pos_arr[:, 2], spacing)

        print('x:', x, 'y:', y, 'z:', z)

        ''' reshape arrays according to xyz dimensions '''
        sdf_arr = sdf_arr.reshape((x, y, z))
        print('sdf_arr_min:', np.min(sdf_arr), 'sdf_arr_max:', np.max(sdf_arr))
        grad_arr = grad_arr.reshape((x, y, z, 3))

        ''' init fields and values '''
        self.sdf_field = ti.field(float, (x, y, z))
        self.grad = ti.Vector.field(3, float, (x, y, z))
        self.sdf_field.from_numpy(sdf_arr)
        self.grad.from_numpy(grad_arr)

        self.count = ti.Vector([x, y, z])
        self.min_corner = ti.Vector([pos_arr[0, 0], pos_arr[0, 1], pos_arr[0, 2]])
        self.spacing = spacing
        self.max_corner = self.min_corner + spacing * (self.count - ti.Vector([1, 1, 1]))


        self.rotation = ti.Matrix.field(4, 4, float, ())
        self.translation = ti.Matrix.field(4, 4, float, ())
        self.transformation = ti.Matrix.field(4, 4, float, ())# transformation matrix (scale matrices UNSUPPORTED), updates together with rotation and translation
        self.inverse_transformation = ti.Matrix.field(4, 4, float, ())# inverse of transformation matrix
        self.previous_transformation = ti.Matrix.field(4, 4, float, ())

        self.transformation[None] = ti.Matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]) # identity matrix
        self.inverse_transformation[None] = self.transformation[None]
        self.rotation[None] = self.transformation[None]
        self.translation[None] = self.transformation[None]
        self.previous_transformation[None] = self.transformation[None]
        self.surface_type = surface_type
        self.friction = config.sph.sdf_bound_friction


    def _getAxisCount(self, arr, spacing):
        return int((np.max(arr) - np.min(arr)) / spacing + 1e-4) + 1

    def readPly(self, filename):
        ply = PlyData.read(filename)
        obj_verts = ply['vertex'].data
        verts_array = np.array([[x, y, z, sdf, nx, ny, nz] for x, y, z, sdf, nx, ny, nz in obj_verts])
        return verts_array

    @ti.func
    def _isPosInRange(self, pos_i):
        flg = True
        for j in ti.static(range(3)):
            if pos_i[j] < self.min_corner[j] + 1e-7 or pos_i[j] > self.max_corner[j] - 1e-7:
                flg = False
        return flg

    @ti.func
    def sdf(self, pos_i):
        inv = ti.static(self.inverse_transformation)
        pos_i = applyTransform(pos_i, inv[None])
        tmp = 0.0
        if self._isPosInRange(pos_i):
            index_f = (pos_i - self.min_corner) / self.spacing
            tmp = trilinearInterpolation(index_f, self.sdf_field)
            # print('pos_i', pos_i, 'index_f', index_f, 'sdf', tmp)
        else:
            tmp = 999.0 #arbitrary large number that's larger than support radius of i
        return tmp

    @ti.func
    def sdf_grad(self, pos_i):
        inv, rot = ti.static(self.inverse_transformation, self.rotation)
        pos_i = applyTransform(pos_i, inv[None])
        tmp = ti.Vector([0.0, 0.0, 0.0])
        if self._isPosInRange(pos_i):
            index_f = (pos_i - self.min_corner) / self.spacing
            tmp = trilinearInterpolation(index_f, self.grad)
        if tmp.norm() > 1e-7:
            tmp = applyTransform(tmp.normalized(), rot[None])
        return tmp

    @ti.func
    def velocity(self, pos_i, time_step):
        inv, pre = ti.static(self.inverse_transformation, self.previous_transformation)
        sdf = self.sdf(pos_i)
        grad = self.sdf_grad(pos_i)
        pos_b_cur = pos_i - sdf * grad
        pos_b_pre = applyTransform(applyTransform(pos_b_cur, inv[None]), pre[None])
        return (pos_b_cur - pos_b_pre) / time_step

    # update rotation, transformation, inverse_transformation
    @ti.func
    def rotate(self, matrix):
        rot, translate, transform, inv = ti.static(self.rotation, self.translation, self.transformation, self.inverse_transformation)
        rot[None] = matrix @ rot[None]
        transform[None] = translate[None] @ rot[None]
        inv[None] = transform[None].inverse()
    
    # update rotation, transformation, inverse_transformation
    @ti.func
    def translate(self, matrix):
        rot, translate, transform, inv = ti.static(self.rotation, self.translation, self.transformation, self.inverse_transformation)
        translate[None] = matrix @ translate[None]
        transform[None] = translate[None] @ rot[None]
        inv[None] = transform[None].inverse()

    # should be performed at the start of each time step before transforming the rigid
    @ti.func
    def updatePreviousTransformation(self):
        self.previous_transformation[None] = self.transformation[None]

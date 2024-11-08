# -*- coding: utf-8 -*-
"""
@author: Alexandre JANIN
@aim:    Main objects of pypax
"""

# External dependencies:
import numpy as np
from scipy.interpolate import griddata
from copy import deepcopy


# ================== FUNCTIONS ==================


class Volume():
    def __new__(cls,points,data):
        """
        Force to have more than just 'duck typing' in Python: 'dynamical typing'
        <i> : geometry = str, geometry of the grid. Must be in ('cart2D',
                         'cart3D','yy','annulus') for cartesian 2D, 3D,
                         Yin-Yang or annulus geometry, respectively. By
                         default, geometry = 'cart3D'
        """
        if len(data.shape) > 1:
            dtype = 'vectorial'
        else:
            dtype = 'scalar'
        #
        if dtype == 'scalar':
            return VolumeScalar(points,data)
        else:
            return VolumeVectorial(points,data)


class VolumeMain:
    def __init__(self,points,data) -> None:
        self.points = points
        self.nod = len(self.points)
        self.x = points[:,0]
        self.y = points[:,1]
        self.z = points[:,2]
        self.data = data
    
    @property
    def bounds(self):
        """spatial bounds"""
        return [np.amin(self.x),np.amax(self.x),\
                np.amin(self.y),np.amax(self.y),\
                np.amin(self.z),np.amax(self.z)]

    @property
    def dbounds(self):
        """data bounds"""
        if isinstance(self,VolumeScalar):
            return (np.amin(self.data),np.amax(self.data))
        elif isinstance(self,VolumeVectorial):
            return [
                    (np.amin(self.data[:,i]),np.amax(self.data[i,:])) for i in range(self.ndims)
                   ]

    def copy(self):
        return deepcopy(self)
    
    def set_mask(self,mask):
        if np.shape(mask) != np.shape(self.x):
            raise IndexError
        else:
            ids = np.arange(self.x.shape[0])[mask]
            self.nod = len(ids)
            self.points = self.points[ids,:]
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.z = self.z[mask]
            if isinstance(self,VolumeScalar):
                self.data = self.data[mask]
            elif isinstance(self,VolumeVectorial):
                self.data = self.data[ids,:]

class VolumeScalar(VolumeMain):
    def __init__(self,points,data) -> None:
        super().__init__(points,data)
        self.dtype = 'scalar'

class VolumeVectorial(VolumeMain):
    def __init__(self,points,data) -> None:
        super().__init__(points,data)
        self.dtype = 'vectorial'
        self.ndim = self.data.shape[1] # number of dimensions





class Slice2D:
    def __init__(self,plan,value,thickness=0.001) -> None:
        # Test
        if plan not in ['x','y','z']:
            raise ValueError()
        # Def
        self.plan = plan
        self.thickness = thickness
        self.value = value
        # Interpolated slice
        self.x = np.array([])
        self.y = np.array([])
        self.z = np.array([])
        self.data = np.array([])
    
    @property
    def ax0(self):
        if self.plan == 'x':
            return self.y
        else:
            return self.x
    
    @property
    def ax1(self):
        if self.plan == 'z':
            return self.y
        else:
            return self.z
    
    @property
    def ay0(self):
        if self.plan == 'x':
            return self.data[:,:,1]
        else:
            return self.data[:,:,0]
    
    @property
    def ay1(self):
        if self.plan == 'z':
            return self.data[:,:,1]
        else:
            return self.data[:,:,2]
        
    
    def slice(self,volume,nodx=100,nody=100,nodz=100,method_interp='linear'):
        # --- Grid creation: unit
        if self.plan != 'x':
            self.x = np.linspace(volume.bounds[0], volume.bounds[1], nodx)
        if self.plan != 'y':
            self.y = np.linspace(volume.bounds[2], volume.bounds[3], nody)
        if self.plan != 'z':
            self.z = np.linspace(volume.bounds[4], volume.bounds[5], nodz)
        # --- Grid creation: meshing
        if self.plan == 'x':
            self.y, self.z = np.meshgrid(self.y, self.z)
            self.x = np.ones(self.y.shape) * self.value
        elif self.plan == 'y':
            self.x, self.z = np.meshgrid(self.x, self.z)
            self.y = np.ones(self.x.shape) * self.value
        elif self.plan == 'z':
            self.x, self.y = np.meshgrid(self.x, self.y)
            self.z = np.ones(self.x.shape) * self.value
        # --- Interpolation
        if volume.dtype == 'scalar':
            self.data = griddata((volume.x, volume.y, volume.z), volume.data, (self.x, self.y, self.z), method=method_interp)
        else:
            self.data = np.zeros((self.x.shape[0],self.x.shape[1],volume.ndim))
            for i in range(volume.ndim):
                self.data[:,:,i] = griddata((volume.x, volume.y, volume.z), volume.data[:,i], (self.x, self.y, self.z), method=method_interp)
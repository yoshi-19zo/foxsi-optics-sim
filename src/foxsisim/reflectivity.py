'''
Created on 2014

@author: StevenChriste
'''


from __future__ import absolute_import

__all__ = ["Reflectivity"]

import numpy as np
import glob
import re
from scipy.interpolate import RegularGridInterpolator
import os
import foxsisim
import h5py


class Reflectivity:
    """Provides reflectivities as a function of angle (in degrees) and energy
        (in keV)"""
    def __init__(self, material='Ir'):
        path = os.path.dirname(foxsisim.__file__) + '/data/'
        h = h5py.File(os.path.join(path, "reflectivity_data.hdf5"), 'r')
        self.data = h['reflectivity/' + material.lower()][:]
        self.energy_ax = h['energy'][:]
        self.angle_ax = h['angle'][:]
        self.material = material
        # RegularGridInterpolator requires the axes to be in ascending order
        # and the data to be in the shape (len(x), len(y))
        self._interpolator = RegularGridInterpolator(
            (self.angle_ax, self.energy_ax), 
            self.data.T, 
            bounds_error=False, 
            fill_value=None,
            method='cubic'
        )

    def func(self, angle, energy):
        """
        Wrapper to maintain compatibility with the old interp2d-like call signature.
        """
        return self._interpolator((angle, energy))

    def energy_range(self):
        """Return the valid range of energies"""
        return np.array([self.energy_ax.min(), self.energy_ax.max()])

    def angle_range(self):
        """Return the valud range of angles"""
        return np.array([self.angle_ax.min(), self.angle_ax.max()])

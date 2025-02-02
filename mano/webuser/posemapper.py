'''
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de

About this file:
================
This file defines a wrapper for the loading functions of the MANO model.
'''

import numpy as np
import jax.numpy as jnp
import cv2


class Rodrigues:
    def __init__(self, rt):
        self.rt = np.asarray(rt)

    def compute_r(self):
        return cv2.Rodrigues(self.rt)[0]

    def compute_dr_wrt(self):
        return cv2.Rodrigues(self.rt)[1].T


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]  # Skip the first 3 elements
        return np.concatenate(
            [(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel()
             for pp in p.reshape((-1, 3))]).ravel()

    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))

    p = p[1:]  # Skip the first rotation
    return np.concatenate([(Rodrigues(pp).compute_r() - np.eye(3)).ravel()
                           for pp in p]).ravel()


def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception(f'Unknown posemapping: {s}')

'''
Copyright 2017 Javier Romero, Dimitrios Tzionas, Michael J Black and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the MANO/SMPL+H Model license here http://mano.is.tue.mpg.de/license

More information about MANO/SMPL+H is available at http://mano.is.tue.mpg.de.
For comments or questions, please email us at: mano@tue.mpg.de

About this file:
================
This file defines a wrapper for the loading functions of the MANO model.

Modules included:
- load_model:
  loads the MANO model from a given file location (i.e. a .pkl file location),
  or a dictionary object.
'''

__all__ = ['load_model', 'save_model']

import numpy as np
import jax.numpy as jnp
import pickle
from mano.webuser.posemapper import posemap
from mano.webuser.verts import verts_core

def ready_arguments(fname_or_dict):

    if not isinstance(fname_or_dict, dict):
        with open(fname_or_dict, 'rb') as f:
            dd = pickle.load(f, encoding='latin1')
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    dd.setdefault('trans', np.zeros(3))
    dd.setdefault('pose', np.zeros(nposeparms))
    
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if s in dd:
            dd[s] = jnp.array(dd[s])  # Replacing chumpy.array with jax.numpy.array

    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'] @ dd['betas'] + dd['v_template']
        v_shaped = dd['v_shaped']
        
        # Replacing MatVecMult with standard matrix multiplication
        J_tmpx = dd['J_regressor'] @ v_shaped[:, 0]
        J_tmpy = dd['J_regressor'] @ v_shaped[:, 1]
        J_tmpz = dd['J_regressor'] @ v_shaped[:, 2]
        
        dd['J'] = jnp.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        dd['v_posed'] = v_shaped + dd['posedirs'] @ posemap(dd['bs_type'])(dd['pose'])
    else:
        dd['v_posed'] = dd['v_template'] + dd['posedirs'] @ posemap(dd['bs_type'])(dd['pose'])

    return dd


def load_model(fname_or_dict):
    dd = ready_arguments(fname_or_dict)

    args = {
        'pose': dd['pose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': jnp,  # Replacing chumpy with jax.numpy
        'want_Jtr': True,
        'bs_style': dd['bs_style']
    }

    result, Jtr = verts_core(**args)
    result = result + dd['trans'].reshape((1, 3))
    result.J_transformed = Jtr + dd['trans'].reshape((1, 3))

    for k, v in dd.items():
        setattr(result, k, v)

    return result

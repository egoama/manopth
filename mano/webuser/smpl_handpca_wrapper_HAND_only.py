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
import pickle
import scipy.sparse as sp
from mano.webuser.posemapper import posemap
from mano.webuser.verts import verts_core


def ready_arguments(fname_or_dict, posekey4vposed='pose'):
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
            dd[s] = jnp.array(dd[s])  # Replace chumpy with jax.numpy

    assert posekey4vposed in dd

    if want_shapemodel:
        dd['v_shaped'] = dd['shapedirs'] @ dd['betas'] + dd['v_template']
        v_shaped = dd['v_shaped']

        # Replace MatVecMult with numpy matrix multiplication
        J_tmpx = dd['J_regressor'] @ v_shaped[:, 0]
        J_tmpy = dd['J_regressor'] @ v_shaped[:, 1]
        J_tmpz = dd['J_regressor'] @ v_shaped[:, 2]
        dd['J'] = jnp.vstack((J_tmpx, J_tmpy, J_tmpz)).T

        pose_map_res = posemap(dd['bs_type'])(dd[posekey4vposed])
        dd['v_posed'] = v_shaped + dd['posedirs'] @ pose_map_res
    else:
        pose_map_res = posemap(dd['bs_type'])(dd[posekey4vposed])
        dd['v_posed'] = dd['v_template'] + dd['posedirs'] @ pose_map_res

    return dd


def load_model(fname_or_dict, ncomps=6, flat_hand_mean=False, v_template=None):
    ''' Loads the fully articulable HAND SMPL model and replaces chumpy with numpy/jax. '''

    np.random.seed(1)

    if not isinstance(fname_or_dict, dict):
        with open(fname_or_dict, 'rb') as f:
            smpl_data = pickle.load(f, encoding='latin1')
    else:
        smpl_data = fname_or_dict

    rot = 3  # for global orientation

    hands_components = smpl_data['hands_components']
    hands_mean = np.zeros(hands_components.shape[1]) if flat_hand_mean else smpl_data['hands_mean']
    hands_coeffs = smpl_data['hands_coeffs'][:, :ncomps]

    selected_components = np.vstack((hands_components[:ncomps]))
    hands_mean = hands_mean.copy()

    pose_coeffs = jnp.zeros(rot + selected_components.shape[0])  # Replace chumpy.zeros
    full_hand_pose = pose_coeffs[rot:(rot + ncomps)] @ selected_components

    smpl_data['fullpose'] = jnp.concatenate((pose_coeffs[:rot], hands_mean + full_hand_pose))
    smpl_data['pose'] = pose_coeffs

    Jreg = smpl_data['J_regressor']
    if not sp.issparse(Jreg):
        smpl_data['J_regressor'] = sp.csc_matrix((Jreg.data, (Jreg.row, Jreg.col)), shape=Jreg.shape)

    # Modify ready_arguments to use fullpose instead of pose
    dd = ready_arguments(smpl_data, posekey4vposed='fullpose')

    args = {
        'pose': dd['fullpose'],
        'v': dd['v_posed'],
        'J': dd['J'],
        'weights': dd['weights'],
        'kintree_table': dd['kintree_table'],
        'xp': jnp,  # Replacing chumpy with jax.numpy
        'want_Jtr': True,
        'bs_style': dd['bs_style'],
    }

    result_previous, meta = verts_core(**args)

    result = result_previous + dd['trans'].reshape((1, 3))
    result.no_translation = result_previous

    if meta is not None:
        for field in ['Jtr', 'A', 'A_global', 'A_weighted']:
            if hasattr(meta, field):
                setattr(result, field, getattr(meta, field))

    setattr(result, 'Jtr', meta)
    if hasattr(result, 'Jtr'):
        result.J_transformed = result.Jtr + dd['trans'].reshape((1, 3))

    for k, v in dd.items():
        setattr(result, k, v)

    if v_template is not None:
        result.v_template[:] = v_template

    return result


if __name__ == '__main__':
    load_model("path/to/your/model.pkl")

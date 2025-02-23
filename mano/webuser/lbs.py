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

from mano.webuser.posemapper import posemap
import numpy as np
import jax.numpy as jnp
import cv2


def global_rigid_transformation(pose, J, kintree_table, xp=np):
    results = {}
    pose = pose.reshape((-1, 3))
    id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
    parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}

    # Use OpenCV's Rodrigues function instead of chumpy
    rodrigues = lambda x: cv2.Rodrigues(x)[0]

    with_zeros = lambda x: xp.vstack((x, xp.array([[0.0, 0.0, 0.0, 1.0]])))

    results[0] = with_zeros(xp.hstack((rodrigues(pose[0, :]), J[0, :].reshape((3, 1)))))

    for i in range(1, kintree_table.shape[1]):
        results[i] = results[parent[i]].dot(
            with_zeros(
                xp.hstack((rodrigues(pose[i, :]), ((J[i, :] - J[parent[i], :]).reshape((3, 1)))))))

    pack = lambda x: xp.hstack([xp.zeros((4, 3)), x.reshape((4, 1))])

    results = [results[i] for i in sorted(results.keys())]
    results_global = results

    if True:
        results2 = [
            results[i] - pack(results[i].dot(xp.concatenate((J[i, :], np.array([0])))))
            for i in range(len(results))
        ]
        results = results2

    result = xp.dstack(results)
    return result, results_global


def verts_core(pose, v, J, weights, kintree_table, want_Jtr=False, xp=np):
    A, A_global = global_rigid_transformation(pose, J, kintree_table, xp)
    T = A.dot(weights.T)

    rest_shape_h = xp.vstack((v.T, xp.ones((1, v.shape[0]))))

    v = (T[:, 0, :] * rest_shape_h[0, :].reshape((1, -1)) +
         T[:, 1, :] * rest_shape_h[1, :].reshape((1, -1)) +
         T[:, 2, :] * rest_shape_h[2, :].reshape((1, -1)) +
         T[:, 3, :] * rest_shape_h[3, :].reshape((1, -1))).T

    v = v[:, :3]

    if not want_Jtr:
        return v

    Jtr = xp.vstack([g[:3, 3] for g in A_global])
    return v, Jtr

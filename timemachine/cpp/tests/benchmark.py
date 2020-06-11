import functools
import unittest
import scipy.linalg
from jax.config import config; config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp

import functools

from common import GradientTest
from common import prepare_nonbonded_system

from timemachine.lib import custom_ops
from timemachine.potentials import alchemy
from timemachine.lib import ops, custom_ops


np.set_printoptions(linewidth=500)

class TestNonbonded(GradientTest):

    def get_water_system(self,
        D,
        P_charges,
        P_lj,
        sort=False):

        x = np.load("water.npy").astype(np.float64)
        if sort:
            perm = hilbert_sort(x, D)
            x = x[perm, :]

        N = x.shape[0]

        params = np.random.rand(P_charges).astype(np.float64)
        params = np.zeros_like(params)
        param_idxs = np.random.randint(low=0, high=P_charges, size=(N), dtype=np.int32)

        lj_params = np.random.rand(P_lj)/10 # we want these to be pretty small for numerical stability reasons
        lj_param_idxs = np.random.randint(low=0, high=P_lj, size=(N,2), dtype=np.int32)
        lj_param_idxs = lj_param_idxs + len(params) # offset 

        return x, np.concatenate([params, lj_params]), param_idxs, lj_param_idxs

    def get_ref_mp(self, x, params, param_idxs, cutoff):
        ref_mp = mixed_fn(x, params, param_idxs, cutoff)[0][0]
        ref_mp = np.transpose(ref_mp, (2,0,1))
        return ref_mp


    def test_water_box(self):
        
        np.random.seed(123)

        P_lj = 50
        P_exc = 7
        x = self.get_water_coords(3)

        P_charges = x.shape[0]

        N = x.shape[0]

        lambda_plane_idxs = np.random.randint(
            low=0,
            high=4,
            size=(N),
            dtype=np.int32
        )

        lambda_offset_idxs = np.random.randint(
            low=0,
            high=1,
            size=(N),
            dtype=np.int32
        )

        # this needs to match the  config.update("jax_enable_x64", True) import
        # for precision, rtol in [(np.float32, 5e-5)]:
        for precision, rtol in [(np.float64, 5e-10)]:

            print("Testing precision", precision)
            E = 100
            cutoff = 10000.0

            params, ref_forces, test_forces = prepare_nonbonded_system(
                x,
                E,
                P_charges,
                P_lj,
                P_exc,
                lambda_plane_idxs,
                lambda_offset_idxs,
                p_scale=10.0,
                e_scale=1.0, # double the charges
                cutoff=cutoff,
                precision=precision
            )

            lamb = 0.5

            for r, t in zip(ref_forces, test_forces):
                self.compare_forces(    
                    x,
                    params,
                    lamb,
                    r,
                    t,
                    precision,
                    rtol)

if __name__ == "__main__":
    unittest.main()

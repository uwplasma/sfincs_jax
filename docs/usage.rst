Usage
=====

Parsing an input file
---------------------

.. code-block:: python

   from sfincs_jax.namelist import read_sfincs_input

   nml = read_sfincs_input("input.namelist")
   print(nml.group("geometryParameters")["GEOMETRYSCHEME"])

Building v3 grids and geometry
------------------------------

.. code-block:: python

   from sfincs_jax.v3 import grids_from_namelist, geometry_from_namelist

   grids = grids_from_namelist(nml)
   geom = geometry_from_namelist(nml=nml, grids=grids)

Applying operator building blocks
---------------------------------

Collisionless v3 operator slice (streaming + mirror):

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np

   from sfincs_jax.collisionless import CollisionlessV3Operator, apply_collisionless_v3

   species = nml.group("speciesParameters")
   t_hats = jnp.asarray(np.atleast_1d(np.asarray(species["THATS"], dtype=float)))
   m_hats = jnp.asarray(np.atleast_1d(np.asarray(species["MHATS"], dtype=float)))

   op = CollisionlessV3Operator(
       x=grids.x,
       ddtheta=grids.ddtheta,
       ddzeta=grids.ddzeta,
       b_hat=geom.b_hat,
       b_hat_sup_theta=geom.b_hat_sup_theta,
       b_hat_sup_zeta=geom.b_hat_sup_zeta,
       db_hat_dtheta=geom.db_hat_dtheta,
       db_hat_dzeta=geom.db_hat_dzeta,
       t_hats=t_hats,
       m_hats=m_hats,
       n_xi_for_x=grids.n_xi_for_x,
   )

   f = jnp.zeros((t_hats.size, grids.x.size, grids.n_xi, grids.theta.size, grids.zeta.size))
   y = apply_collisionless_v3(op, f)

Pitch-angle scattering collisions (``collisionOperator = 1`` without Phi1):

.. code-block:: python

   from sfincs_jax.collisions import make_pitch_angle_scattering_v3_operator, apply_pitch_angle_scattering_v3

   z_s = jnp.asarray(np.atleast_1d(np.asarray(species["ZS"], dtype=float)))
   n_hats = jnp.asarray(np.atleast_1d(np.asarray(species["NHATS"], dtype=float)))

   phys = nml.group("physicsParameters")
   nu_n = float(phys["NU_N"])

   cop = make_pitch_angle_scattering_v3_operator(
       x=grids.x,
       z_s=z_s,
       m_hats=m_hats,
       n_hats=n_hats,
       t_hats=t_hats,
       nu_n=nu_n,
       n_xi_for_x=grids.n_xi_for_x,
   )

   y_col = apply_pitch_angle_scattering_v3(cop, f)

Running the Fortran v3 executable
--------------------------------

.. code-block:: bash

   export SFINCS_FORTRAN_EXE=/path/to/sfincs/fortran/version3/sfincs
   sfincs_jax run-fortran --input /path/to/input.namelist

Writing `sfincsOutput.h5` with `sfincs_jax`
------------------------------------------

.. code-block:: bash

   sfincs_jax write-output --input /path/to/input.namelist --out sfincsOutput.h5

.. code-block:: python

   from pathlib import Path
   from sfincs_jax.io import write_sfincs_jax_output_h5

   write_sfincs_jax_output_h5(
       input_namelist=Path("input.namelist"),
       output_path=Path("sfincsOutput.h5"),
   )

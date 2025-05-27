from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class DEGDIAG:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : DEGDIAG
#    *********
# 
#    A degenerate bound constrained convex quadratic program
#    with a tri-diagonal Hessian
# 
#    SIF input: Nick Gould, August 2011
# 
#    classification = "C-CQBR2-AN-V-0"
# 
#    The number of variables - 1
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER
# IE N                   50             $-PARAMETER
# IE N                   100            $-PARAMETER
# IE N                   1000           $-PARAMETER
# IE N                   10000          $-PARAMETER
# IE N                   100000         $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'DEGDIAG'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(10);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
        v_['0'] = 0
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['0']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        v_['RN'] = float(v_['N'])
        v_['RN'] = 1.0+v_['RN']
        for I in range(int(v_['0']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['LOWER'] = v_['RI']/v_['RN']
            self.xlower = jtu.np_like_set(self.xlower, ix_['X'+str(I)], v_['LOWER'])
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(2.0))
        #%%%%%%%%%%%%%%%%%%%% QUADRATIC %%%%%%%%%%%%%%%%%%%
        irH  = jnp.array([],dtype=int)
        icH  = jnp.array([],dtype=int)
        valH = jnp.array([],dtype=float)
        for I in range(int(v_['0']),int(v_['N'])+1):
            irH  = jtu.append(irH,[ix_['X'+str(I)]])
            icH  = jtu.append(icH,[ix_['X'+str(I)]])
            valH = jtu.append(valH,float(1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.H = BCSR.from_bcoo(BCOO((valH, jnp.array((irH,icH)).T), shape=(self.n,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CQBR2-AN-V-0"
        self.objderlvl = 2


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


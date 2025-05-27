from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class DIAGPQT:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : DIAGPQT
#    *********
#    A variable dimension convex quadratic problem
#    with eigenvalues clusetered towards the top of the spectrum
# 
#    lambda_i = - i^2/n + n + 1/n, i = 1, ... , n
# 
#    Source: simple test for GALAHAD gltr/glrt
# 
#    SIF input: Nick Gould, Feb 2019
# 
#    classification = "C-CQBR2-AN-V-0"
# 
#    Number of variables (variable)
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'DIAGPQT'

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
# IE N                   50             $-PARAMETER
# IE N                   100            $-PARAMETER
# IE N                   500            $-PARAMETER     original value
# IE N                   1000           $-PARAMETER
# IE N                   5000           $-PARAMETER
# IE N                   10000          $-PARAMETER
# IE N                   100000         $-PARAMETER
# IE N                   1000000        $-PARAMETER
        v_['1'] = 1
        v_['ONE'] = 1.0
        v_['RN'] = float(v_['N'])
        v_['1/RN'] = v_['ONE']/v_['RN']
        v_['SHIFT'] = v_['RN']+v_['1/RN']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(1.0))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-100000.0)
        self.xupper = jnp.full((self.n,1),1000000.0)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(1.0))
        #%%%%%%%%%%%%%%%%%%%% QUADRATIC %%%%%%%%%%%%%%%%%%%
        irH  = jnp.array([],dtype=int)
        icH  = jnp.array([],dtype=int)
        valH = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['RI2'] = v_['RI']*v_['RI']
            v_['RI2/RN'] = v_['RI2']/v_['RN']
            v_['-RI2/RN'] = -1.0*v_['RI2/RN']
            v_['H'] = v_['-RI2/RN']+v_['SHIFT']
            irH  = jtu.append(irH,[ix_['X'+str(I)]])
            icH  = jtu.append(icH,[ix_['X'+str(I)]])
            valH = jtu.append(valH,float(v_['H']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               0.0
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        self.H = BCSR.from_bcoo(BCOO((valH, jnp.array((irH,icH)).T), shape=(self.n,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CQBR2-AN-V-0"
        self.objderlvl = 2


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


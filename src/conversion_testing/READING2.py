from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class READING2:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : READING2
#    *********
# 
#    A linear optimal control problem from Nancy Nichols
#    with a given initial condition.
#    This problem arises in tide modelling.
# 
#    Source:
#    S. Lyle and N.K. Nichols,
#    "Numerical Methods for Optimal Control Problems with State Constraints",
#    Numerical Analysis Report 8/91, Dept of Mathematics, 
#    University of Reading, UK.
# 
#    SIF input: Nick Gould, July 1991.
# 
#    classification = "C-CLLR2-MN-V-V"
# 
#    Number of discretized points in [0,1]
# 
#           Alternative values for the SIF file parameters:
# IE N                   2              $-PARAMETER
# IE N                   50             $-PARAMETER
# IE N                   100            $-PARAMETER     original value
# IE N                   500            $-PARAMETER
# IE N                   1000           $-PARAMETER
# IE N                   2000           $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'READING2'

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
# IE N                   5000           $-PARAMETER
        v_['PI'] = 3.1415926535
        v_['2PI'] = 2.0*v_['PI']
        v_['PI**2'] = v_['PI']*v_['PI']
        v_['8PI**2'] = 8.0*v_['PI**2']
        v_['1/8PI**2'] = 1.0/v_['8PI**2']
        v_['A'] = 0.07716
        v_['1/A'] = 1.0/v_['A']
        v_['1/2A'] = 2.0*v_['1/A']
        v_['2A'] = 2.0*v_['A']
        v_['-2A'] = -1.0*v_['2A']
        v_['-1/2A'] = 1.0/v_['-2A']
        v_['N-1'] = -1+v_['N']
        v_['RN'] = float(v_['N'])
        v_['H'] = 1.0/v_['RN']
        v_['2/H'] = 2.0*v_['RN']
        v_['H/2'] = 0.5*v_['H']
        v_['-H/2'] = -1.0*v_['H/2']
        v_['1/H'] = 1.0*v_['RN']
        v_['-1/H'] = -1.0*v_['RN']
        v_['H/8PI**2'] = v_['1/8PI**2']*v_['H']
        v_['0'] = 0
        v_['1'] = 1
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['0']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X1u'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X1u'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('X2u'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X2u'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('U'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'U'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['TI'] = v_['RI']*v_['H']
            v_['2PITI'] = v_['2PI']*v_['TI']
            v_['CTI'] = jnp.cos(v_['2PITI'])
            v_['-CCTI'] = v_['CTI']*v_['-H/2']
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            v_['TI-1'] = v_['RI-1']*v_['H']
            v_['2PITI-1'] = v_['2PI']*v_['TI-1']
            v_['CTI-1'] = jnp.cos(v_['2PITI-1'])
            v_['-CCTI-1'] = v_['CTI-1']*v_['-H/2']
            [ig,ig_,_] = jtu.s2mpj_ii('COST',ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X1u'+str(I)]])
            valA = jtu.append(valA,float(v_['-CCTI']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X1u'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['-CCTI-1']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
            valA = jtu.append(valA,float(v_['H/8PI**2']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['H/8PI**2']))
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            [ig,ig_,_] = jtu.s2mpj_ii('C1u'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'C1u'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X1u'+str(I)]])
            valA = jtu.append(valA,float(v_['1/H']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X1u'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['-1/H']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X2u'+str(I)]])
            valA = jtu.append(valA,float(-0.5))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X2u'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(-0.5))
            [ig,ig_,_] = jtu.s2mpj_ii('C2u'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'C2u'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X2u'+str(I)]])
            valA = jtu.append(valA,float(v_['1/H']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X2u'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['-1/H']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
            valA = jtu.append(valA,float(-0.5))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(-0.5))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        legrps = jnp.where(gtype=='<=')[0]
        eqgrps = jnp.where(gtype=='==')[0]
        gegrps = jnp.where(gtype=='>=')[0]
        self.nle = len(legrps)
        self.neq = len(eqgrps)
        self.nge = len(gegrps)
        self.m   = self.nle+self.neq+self.nge
        self.congrps = jnp.concatenate((legrps,eqgrps,gegrps))
        self.cnames = cnames[self.congrps]
        self.nob = ngrp-self.m
        self.objgrps = jnp.where(gtype=='<>')[0]
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['X1u'+str(int(v_['0']))], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X1u'+str(int(v_['0']))], 0.0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['X2u'+str(int(v_['0']))], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X2u'+str(int(v_['0']))], 0.0)
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['X1u'+str(I)], -float('Inf'))
            self.xupper = jtu.np_like_set(self.xupper, ix_['X1u'+str(I)], +float('Inf'))
            self.xlower = jtu.np_like_set(self.xlower, ix_['X2u'+str(I)], -0.125)
            self.xupper = jtu.np_like_set(self.xupper, ix_['X2u'+str(I)], 0.125)
        for I in range(int(v_['0']),int(v_['N'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['U'+str(I)], -1.0)
            self.xupper = jtu.np_like_set(self.xupper, ix_['U'+str(I)], 1.0)
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons   = jnp.arange(len(self.congrps))
        self.pbclass   = "C-CLLR2-MN-V-V"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


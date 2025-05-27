from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class ARGLALE:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : ARGLALE
#    *********
#    Variable dimension full rank linear problem, a linear equation version.
# 
#    Source: Problem 32 in
#    J.J. More', B.S. Garbow and K.E. Hillstrom,
#    "Testing Unconstrained Optimization Software",
#    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.
# 
#    See also Buckley#80 (with different N and M)
#    SIF input: Ph. Toint, Dec 1989.
# 
#    classification = "C-CNLR2-AN-V-V"
# 
#    This is a(n infeasible) linear feasibility problem 
# 
#    N is the number of free variables
#    M is the number of equations ( M.ge.N)
# 
#           Alternative values for the SIF file parameters:
# IE N                   4              $-PARAMETER
# IE N                   10             $-PARAMETER
# IE N                   50             $-PARAMETER 
# IE N                   100            $-PARAMETER
# IE N                   200            $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'ARGLALE'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(4);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE M                   6              $-PARAMETER .ge. N
# IE M                   20             $-PARAMETER .ge. N
# IE M                   100            $-PARAMETER .ge. N
# IE M                   200            $-PARAMETER .ge. N
# IE M                   400            $-PARAMETER .ge. N
        if nargin<2:
            v_['M'] = int(6);  #  SIF file default value
        else:
            v_['M'] = int(args[1])
        v_['1'] = 1
        v_['-2.0'] = -2.0
        v_['N+1'] = 1+v_['N']
        v_['RM'] = float(v_['M'])
        v_['-2/M'] = v_['-2.0']/v_['RM']
        v_['1-2/M'] = 1.0+v_['-2/M']
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
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            for J in range(int(v_['1']),int(v_['I-1'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'G'+str(I))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(v_['-2/M']))
            [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'G'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(v_['1-2/M']))
            for J in range(int(v_['I+1']),int(v_['N'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'G'+str(I))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(v_['-2/M']))
        for I in range(int(v_['N+1']),int(v_['M'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'G'+str(I))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(v_['-2/M']))
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
        #%%%%%%%%%%%%%%%%%%  CONSTANTS %%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.full((ngrp,1),1.0)
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(1.0))
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
        self.pbclass   = "C-CNLR2-AN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LSQFIT:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : LSQFIT
#    *********
#    An elementary constrained linear least-squares fit
# 
#    Source:
#    A.R. Conn, N. Gould and Ph.L. Toint,
#    "The LANCELOT User's Manual",
#    Dept of Maths, FUNDP, 1991.
# 
#    SIF input: Ph. Toint, Jan 1991.
# 
#    classification = "C-CSLR2-AN-2-1"
# 
#    Data points
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LSQFIT'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['X1'] = 0.1
        v_['X2'] = 0.3
        v_['X3'] = 0.5
        v_['X4'] = 0.7
        v_['X5'] = 0.9
        v_['Y1'] = 0.25
        v_['Y2'] = 0.3
        v_['Y3'] = 0.625
        v_['Y4'] = 0.701
        v_['Y5'] = 1.0
        v_['C'] = 0.85
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('a',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'a')
        [iv,ix_,_] = jtu.s2mpj_ii('b',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'b')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('Obj1',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['a']])
        valA = jtu.append(valA,float(v_['X1']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['b']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(2.0))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj2',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['a']])
        valA = jtu.append(valA,float(v_['X2']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['b']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(2.0))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj3',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['a']])
        valA = jtu.append(valA,float(v_['X3']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['b']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(2.0))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj4',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['a']])
        valA = jtu.append(valA,float(v_['X4']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['b']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(2.0))
        [ig,ig_,_] = jtu.s2mpj_ii('Obj5',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['a']])
        valA = jtu.append(valA,float(v_['X5']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['b']])
        valA = jtu.append(valA,float(1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(2.0))
        [ig,ig_,_] = jtu.s2mpj_ii('Cons',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'Cons')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['a']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['b']])
        valA = jtu.append(valA,float(1.0))
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
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj1'],float(v_['Y1']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj2'],float(v_['Y2']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj3'],float(v_['Y3']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj4'],float(v_['Y4']))
        self.gconst = jtu.arrset(self.gconst,ig_['Obj5'],float(v_['Y5']))
        self.gconst = jtu.arrset(self.gconst,ig_['Cons'],float(v_['C']))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['b'], -float('Inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['b'], +float('Inf'))
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gSQUARE',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['Obj1']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj2']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj3']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj4']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        ig = ig_['Obj5']
        self.grftype = jtu.arrset(self.grftype,ig,'gSQUARE')
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons   = jnp.arange(len(self.congrps))
        self.pbclass   = "C-CSLR2-AN-2-1"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gSQUARE(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= GVAR_*GVAR_
        if nargout>1:
            g_ = GVAR_+GVAR_
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = 2.0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


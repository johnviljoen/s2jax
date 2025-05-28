import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class YAO:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
#    A linear least-sqaure problem with k-convex constraints
#       min (1/2) || f(t) - x ||^2
#    subject to the constraints
#       _ 
#       \/_k  x  >=  0,
#    where  f(t) and  x  are vectors in (n+k)-dimensional space.
# 
#    We choose f(t) = sin(t), x(1) >= 0.08 and fix x(n+i) = 0
# 
#    SIF input: Aixiang Yao, Virginia Tech., May 1995
#               modifications by Nick Gould
# 
#    classification = "C-CQLR2-AN-V-V"
# 
#   Number of discretization points
# 
#           Alternative values for the SIF file parameters:
# IE P                   20             $-PARAMETER
# IE P                   200            $-PARAMETER
# IE P                   2000           $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'YAO'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['P'] = int(20);  #  SIF file default value
        else:
            v_['P'] = int(args[0])
# IE k                   2              $-PARAMETER
        if nargin<2:
            v_['k'] = int(2);  #  SIF file default value
        else:
            v_['k'] = int(args[1])
# IE k                   3              $-PARAMETER
# IE k                   4              $-PARAMETER
        v_['1'] = 1
        v_['2'] = 2
        v_['P+1'] = v_['P']+v_['1']
        v_['P+k'] = v_['P']+v_['k']
        v_['RP'] = float(v_['P+k'])
        v_['OVP'] = 1.0/v_['RP']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for i in range(int(v_['1']),int(v_['P+k'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(i),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(i))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for i in range(int(v_['1']),int(v_['P+k'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('S'+str(i),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(i)]])
            valA = jtu.append(valA,float(1.0))
            self.gscale = jtu.arrset(self.gscale,ig,float(2.0))
        for i in range(int(v_['1']),int(v_['P'])+1):
            v_['i+1'] = 1+i
            [ig,ig_,_] = jtu.s2mpj_ii('B'+str(i),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'B'+str(i))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(i)]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['i+1']))]])
            valA = jtu.append(valA,float(-2.0))
            v_['i+2'] = 2+i
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['i+2']))]])
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
        for i in range(int(v_['1']),int(v_['P+k'])+1):
            v_['Ri'] = float(i)
            v_['iOVP'] = v_['Ri']*v_['OVP']
            v_['SINI'] = jnp.sin(v_['iOVP'])
            self.gconst = jtu.arrset(self.gconst,ig_['S'+str(i)],float(v_['SINI']))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['X'+str(int(v_['1']))], 0.08)
        for i in range(int(v_['P+1']),int(v_['P+k'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['X'+str(i)], 0.0)
            self.xupper = jtu.np_like_set(self.xupper, ix_['X'+str(i)], 0.0)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gSQ',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for i in range(int(v_['1']),int(v_['P+k'])+1):
            ig = ig_['S'+str(i)]
            self.grftype = jtu.arrset(self.grftype,ig,'gSQ')
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# XL SOLUTION             2.39883D+00   $ (p=20)
# XL SOLUTION             2.01517D+01   $ (p=200)
# XL SOLUTION             1.97705D+02   $ (p=2000)
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons   = jnp.arange(len(self.congrps))
        self.pbclass   = "C-CQLR2-AN-V-V"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]

# ********************
#  SET UP THE GROUPS *
#  ROUTINE           *
# ********************

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gSQ(self,nargout,*args):

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


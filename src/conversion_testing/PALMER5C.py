import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class PALMER5C:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : PALMER5C
#    *********
# 
#    A linear least squares problem
#    arising from chemical kinetics.
# 
#     model: H-N=C=Se TZVP + MP2
#    fitting Y to A0 T_0 + A2 T_2 + A4 T_4 + A6 T_6 + A8 T_8 +
#                 A10 T_10 + A12 T_12 + A14 T_14
#    where T_i is the i-th (shifted) Chebyshev polynomial
# 
#    Source:
#    M. Palmer, Edinburgh, private communication.
# 
#    SIF input: Nick Gould, 1992.
# 
#    classification = "C-CQUR2-RN-6-0"
# 
#    Number of data points
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'PALMER5C'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['M'] = 23
        v_['1'] = 1
        v_['2'] = 2
        v_['12'] = 12
        v_['14'] = 14
        v_['X12'] = 0.000000
        v_['X13'] = 1.570796
        v_['X14'] = 1.396263
        v_['X15'] = 1.308997
        v_['X16'] = 1.221730
        v_['X17'] = 1.125835
        v_['X18'] = 1.047198
        v_['X19'] = 0.872665
        v_['X20'] = 0.698132
        v_['X21'] = 0.523599
        v_['X22'] = 0.349066
        v_['X23'] = 0.174533
        v_['B'] = v_['X13']
        v_['A'] = -1.0e+0*v_['B']
        v_['DIFF'] = 2.0e+0*v_['B']
        v_['Y12'] = 83.57418
        v_['Y13'] = 81.007654
        v_['Y14'] = 18.983286
        v_['Y15'] = 8.051067
        v_['Y16'] = 2.044762
        v_['Y17'] = 0.000000
        v_['Y18'] = 1.170451
        v_['Y19'] = 10.479881
        v_['Y20'] = 25.785001
        v_['Y21'] = 44.126844
        v_['Y22'] = 62.822177
        v_['Y23'] = 77.719674
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('A0',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'A0')
        [iv,ix_,_] = jtu.s2mpj_ii('A2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'A2')
        [iv,ix_,_] = jtu.s2mpj_ii('A4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'A4')
        [iv,ix_,_] = jtu.s2mpj_ii('A6',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'A6')
        [iv,ix_,_] = jtu.s2mpj_ii('A8',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'A8')
        [iv,ix_,_] = jtu.s2mpj_ii('A10',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'A10')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['12']),int(v_['M'])+1):
            v_['T0'] = 1.0e+0
            v_['Y'] = 2.0e+0*v_['X'+str(I)]
            v_['Y'] = v_['Y']-v_['A']
            v_['Y'] = v_['Y']-v_['B']
            v_['Y'] = v_['Y']/v_['DIFF']
            v_['T1'] = v_['Y']
            v_['2Y'] = 2.0e+0*v_['Y']
            for J in range(int(v_['2']),int(v_['14'])+1):
                v_['J-1'] = -1+J
                v_['J-2'] = -2+J
                v_['T'+str(J)] = v_['2Y']*v_['T'+str(int(v_['J-1']))]
                v_['T'+str(J)] = v_['T'+str(J)]-v_['T'+str(int(v_['J-2']))]
            [ig,ig_,_] = jtu.s2mpj_ii('O'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A0']])
            valA = jtu.append(valA,float(v_['T0']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A2']])
            valA = jtu.append(valA,float(v_['T2']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A4']])
            valA = jtu.append(valA,float(v_['T4']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A6']])
            valA = jtu.append(valA,float(v_['T6']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A8']])
            valA = jtu.append(valA,float(v_['T8']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A10']])
            valA = jtu.append(valA,float(v_['T10']))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for I in range(int(v_['12']),int(v_['M'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['O'+str(I)],float(v_['Y'+str(I)]))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['A0'], -float('Inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['A0'], +float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['A2'], -float('Inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['A2'], +float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['A4'], -float('Inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['A4'], +float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['A6'], -float('Inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['A6'], +float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['A8'], -float('Inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['A8'], +float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['A10'], -float('Inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['A10'], +float('Inf'))
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(1.0))
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gL2',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['12']),int(v_['M'])+1):
            ig = ig_['O'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Least square problems are bounded below by zero
        self.objlower = 0.0
#    Solution
# LO SOLTN              5.0310687D-02
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CQUR2-RN-6-0"
        self.objderlvl = 2

# ********************
#  SET UP THE GROUPS *
#  ROUTINE           *
# ********************

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gL2(self,nargout,*args):

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


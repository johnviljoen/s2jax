from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class SCURLY30:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : SCURLY30
#    --------
# 
#    A banded function with semi-bandwidth 10 and
#    negative curvature near the starting point.
#    NB: scaled version of CURLY30
# 
#    Source: Nick Gould
# 
#    SIF input: Nick Gould, September 1997.
# 
#    classification = "C-COUR2-AN-V-0"
# 
#    Number of variables
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER
# IE N                   100            $-PARAMETER
# IE N                   1000           $-PARAMETER     original value
# IE N                   10000          $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'SCURLY30'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(35);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE N                   100000         $-PARAMETER
# IE N                   1000000        $-PARAMETER
        v_['K'] = 30
        v_['SCAL'] = 12.0
        v_['1'] = 1
        v_['N-K'] = v_['N']-v_['K']
        v_['N-K+1'] = 1+v_['N-K']
        v_['RN'] = float(v_['N'])
        v_['RN-1'] = -1+v_['RN']
        v_['RN+1'] = 1+v_['RN']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            v_['RAT'] = v_['RI-1']/v_['RN-1']
            v_['ARG'] = v_['RAT']*v_['SCAL']
            v_['S'+str(I)] = jnp.exp(v_['ARG'])
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['N-K'])+1):
            v_['I+K'] = I+v_['K']
            for J in range(int(I),int(v_['I+K'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('Q'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(v_['S'+str(J)]))
        for I in range(int(v_['N-K+1']),int(v_['N'])+1):
            for J in range(int(I),int(v_['N'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('Q'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(v_['S'+str(J)]))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['T'] = v_['RI']/v_['RN+1']
            v_['T'] = 0.0001*v_['T']
            v_['T'] = v_['T']*v_['S'+str(I)]
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(v_['T']))
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gP4',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for ig in range(0,ngrp):
            self.grftype = jtu.arrset(self.grftype,ig,'gP4')
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# ZL SOLTN               -1.003163D+5   $ (n=1000)
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-COUR2-AN-V-0"
        self.objderlvl = 2

# ********************
#  SET UP THE GROUPS *
#  ROUTINE           *
# ********************

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gP4(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        APB = 2.0e+1
        f_= GVAR_*(GVAR_*(GVAR_**2-APB)-1.0e-1)
        if nargout>1:
            g_ = 2.0e+0*GVAR_*(2.0e+0*GVAR_**2-APB)-1.0e-1
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = 1.2e+1*GVAR_**2-2.0e+0*APB
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


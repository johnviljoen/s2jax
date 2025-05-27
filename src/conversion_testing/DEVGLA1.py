from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class DEVGLA1:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : DEVGLA1
#    *********
# 
#    SCIPY global optimization benchmark example DeVilliersGlasser01
# 
#    Fit: y  = x_1 x_2^t sin( t x_3 + x_4 )  +  e
# 
#    Source:  Problem from the SCIPY benchmark set
#      https://github.com/scipy/scipy/tree/master/benchmarks/ ...
#              benchmarks/go_benchmark_functions
# 
#    SIF input: Nick Gould, Jan 2020
# 
#    classification = "C-CSUR2-MN-4-0"
# 
#    Number of data values
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'DEVGLA1'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['M'] = 24
        v_['N'] = 4
        v_['1'] = 1
        v_['A'] = 1.371
        v_['LNA'] = jnp.log(v_['A'])
        for I in range(int(v_['1']),int(v_['M'])+1):
            v_['RI'] = float(I)
            v_['RIM1'] = -1.0+v_['RI']
            v_['T'] = 0.1*v_['RIM1']
            v_['T'+str(I)] = v_['T']
            v_['TLNA'] = v_['T']*v_['LNA']
            v_['AT'] = jnp.exp(v_['TLNA'])
            v_['TP'] = 3.112*v_['T']
            v_['TPA'] = 1.761+v_['TP']
            v_['STPA'] = jnp.sin(v_['TPA'])
            v_['P'] = v_['AT']*v_['STPA']
            v_['PP'] = 60.137*v_['P']
            v_['Y'+str(I)] = v_['PP']
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
        for I in range(int(v_['1']),int(v_['M'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('F'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for I in range(int(v_['1']),int(v_['M'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['F'+str(I)],float(v_['Y'+str(I)]))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(2.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X2'], float(2.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X3'], float(2.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X4'], float(2.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eDG1', iet_)
        elftv = jtu.loaset(elftv,it,0,'X1')
        elftv = jtu.loaset(elftv,it,1,'X2')
        elftv = jtu.loaset(elftv,it,2,'X3')
        elftv = jtu.loaset(elftv,it,3,'X4')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'T')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['M'])+1):
            ename = 'E'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eDG1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eDG1"])
            vname = 'X1'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X2'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X3'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X3')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X4'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X4')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='T')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['T'+str(I)]))
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
        for ig in range(0,ngrp):
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
        for I in range(int(v_['1']),int(v_['M'])+1):
            ig = ig_['F'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Least square problems are bounded below by zero
        self.objlower = 0.0
#    Solution
# LO SOLUTION            0.0
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSUR2-MN-4-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eDG1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        A = self.elpar[iel_][0]*EV_[2]+EV_[3]
        SINA = jnp.sin(A)
        COSA = jnp.cos(A)
        X2T = EV_[1]**self.elpar[iel_][0]
        DX2T = self.elpar[iel_][0]*EV_[1]**(self.elpar[iel_][0]-1.0e0)
        D2X2T  = (
              self.elpar[iel_][0]*(self.elpar[iel_][0]-1.0e0)*EV_[1]**(self.elpar[iel_][0]-2.0e0))
        X1X2T = EV_[0]*X2T
        f_   = X1X2T*SINA
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, X2T*SINA)
            g_ = jtu.np_like_set(g_, 1, EV_[0]*DX2T*SINA)
            g_ = jtu.np_like_set(g_, 2, X1X2T*COSA*self.elpar[iel_][0])
            g_ = jtu.np_like_set(g_, 3, X1X2T*COSA)
            if nargout>2:
                H_ = jnp.zeros((4,4))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), DX2T*SINA)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), X2T*COSA*self.elpar[iel_][0])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), X2T*COSA)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), EV_[0]*D2X2T*SINA)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), EV_[0]*DX2T*COSA*self.elpar[iel_][0])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), EV_[0]*DX2T*COSA)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), -X1X2T*SINA*self.elpar[iel_][0]*self.elpar[iel_][0])
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), -X1X2T*SINA*self.elpar[iel_][0])
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), -X1X2T*SINA)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

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


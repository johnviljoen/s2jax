import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class DEVGLA2B:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : DEVGLA2B
#    *********
# 
#    SCIPY global optimization benchmark example DeVilliersGlasser02
# 
#    Fit: y  = x_1 x_2^t tanh ( t x_3 + sin( t x_4 ) ) cos( t e^x_5 )  +  e
# 
#    version with box-constrained feasible region
# 
#    Source:  Problem from the SCIPY benchmark set
#      https://github.com/scipy/scipy/tree/master/benchmarks/ ...
#              benchmarks/go_benchmark_functions
# 
#    SIF input: Nick Gould, Jan 2020
# 
#    classification = "C-CSBR2-MN-5-0"
# 
#    Number of data values
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'DEVGLA2B'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['M'] = 16
        v_['N'] = 5
        v_['1'] = 1
        v_['A'] = 1.27
        v_['LNA'] = jnp.log(v_['A'])
        for I in range(int(v_['1']),int(v_['M'])+1):
            v_['RI'] = float(I)
            v_['RIM1'] = -1.0+v_['RI']
            v_['T'] = 0.1*v_['RIM1']
            v_['T'+str(I)] = v_['T']
            v_['TLNA'] = v_['T']*v_['LNA']
            v_['AT'] = jnp.exp(v_['TLNA'])
            v_['TP'] = 3.012*v_['T']
            v_['TP2'] = 2.13*v_['T']
            v_['STP2'] = jnp.sin(v_['TP2'])
            v_['TPA'] = v_['TP']+v_['STP2']
            v_['HTPA'] = jnp.tanh(v_['TPA'])
            v_['EC'] = jnp.exp(0.507)
            v_['ECT'] = v_['EC']*v_['T']
            v_['CECT'] = jnp.cos(v_['ECT'])
            v_['P'] = v_['AT']*v_['HTPA']
            v_['PP'] = v_['P']*v_['CECT']
            v_['PPP'] = 53.81*v_['PP']
            v_['Y'+str(I)] = v_['PPP']
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
        self.xlower = jnp.full((self.n,1),1.0)
        self.xupper = jnp.full((self.n,1),60.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(20.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X2'], float(2.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X3'], float(2.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X4'], float(2.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X5'], float(0.2))
        pass
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eDG2', iet_)
        elftv = jtu.loaset(elftv,it,0,'X1')
        elftv = jtu.loaset(elftv,it,1,'X2')
        elftv = jtu.loaset(elftv,it,2,'X3')
        elftv = jtu.loaset(elftv,it,3,'X4')
        elftv = jtu.loaset(elftv,it,4,'X5')
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
            self.elftype = jtu.arrset(self.elftype,ie,'eDG2')
            ielftype = jtu.arrset(ielftype,ie,iet_["eDG2"])
            vname = 'X1'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(60.0),None)
            posev = jnp.where(elftv[ielftype[ie]]=='X1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X2'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(60.0),None)
            posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X3'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(60.0),None)
            posev = jnp.where(elftv[ielftype[ie]]=='X3')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X4'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(60.0),None)
            posev = jnp.where(elftv[ielftype[ie]]=='X4')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X5'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0),float(60.0),None)
            posev = jnp.where(elftv[ielftype[ie]]=='X5')[0]
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
        self.pbclass   = "C-CSBR2-MN-5-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eDG2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        X2T = EV_[1]**self.elpar[iel_][0]
        F2 = X2T
        F2X2 = self.elpar[iel_][0]*EV_[1]**(self.elpar[iel_][0]-1.0e0)
        F2X2X2  = (               self.elpar[iel_][0]*(self.elpar[iel_][0]-1.0e0)*EV_[1]**(self.elpar[iel_][0]-2.0e0))
        X3T = EV_[2]*self.elpar[iel_][0]
        X4T = EV_[3]*self.elpar[iel_][0]
        SINX4T = jnp.sin(X4T)
        COSX4T = jnp.cos(X4T)
        A = X3T+SINX4T
        AX3 = self.elpar[iel_][0]
        AX4 = self.elpar[iel_][0]*COSX4T
        AX4X4 = -self.elpar[iel_][0]*self.elpar[iel_][0]*SINX4T
        F = jnp.tanh(A)
        FA = 1.0/(jnp.cosh(A))**2
        FAA = -2.0*FA*F
        F3 = F
        F3X3 = FA*AX3
        F3X4 = FA*AX4
        F3X3X3 = FAA*AX3*AX3
        F3X3X4 = FAA*AX3*AX4
        F3X4X4 = FA*AX4X4+FAA*AX4*AX4
        EX5 = jnp.exp(EV_[4])
        TEX5 = self.elpar[iel_][0]*EX5
        STEX5 = jnp.sin(TEX5)
        CTEX5 = jnp.cos(TEX5)
        F4 = CTEX5
        F4X5 = -STEX5*TEX5
        F4X5X5 = -STEX5*TEX5-CTEX5*TEX5*TEX5
        f_   = EV_[0]*F2*F3*F4
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, F2*F3*F4)
            g_ = jtu.np_like_set(g_, 1, EV_[0]*F2X2*F3*F4)
            g_ = jtu.np_like_set(g_, 2, EV_[0]*F2*F3X3*F4)
            g_ = jtu.np_like_set(g_, 3, EV_[0]*F2*F3X4*F4)
            g_ = jtu.np_like_set(g_, 4, EV_[0]*F2*F3*F4X5)
            if nargout>2:
                H_ = jnp.zeros((5,5))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), F2X2*F3*F4)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), F2*F3X3*F4)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), F2*F3X4*F4)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([0,4]), F2*F3*F4X5)
                H_ = jtu.np_like_set(H_, jnp.array([4,0]), H_[0,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), EV_[0]*F2X2X2*F3*F4)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), EV_[0]*F2X2*F3X3*F4)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), EV_[0]*F2X2*F3X4*F4)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,4]), EV_[0]*F2X2*F3*F4X5)
                H_ = jtu.np_like_set(H_, jnp.array([4,1]), H_[1,4])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), EV_[0]*F2*F3X3X3*F4)
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), EV_[0]*F2*F3X3X4*F4)
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,4]), EV_[0]*F2*F3X3*F4X5)
                H_ = jtu.np_like_set(H_, jnp.array([4,2]), H_[2,4])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), EV_[0]*F2*F3X4X4*F4)
                H_ = jtu.np_like_set(H_, jnp.array([3,4]), EV_[0]*F2*F3X4*F4X5)
                H_ = jtu.np_like_set(H_, jnp.array([4,3]), H_[3,4])
                H_ = jtu.np_like_set(H_, jnp.array([4,4]), EV_[0]*F2*F3*F4X5X5)
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


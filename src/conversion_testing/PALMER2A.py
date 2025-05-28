import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class PALMER2A:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : PALMER2A
#    *********
# 
#    A nonlinear least squares problem with bounds
#    arising from chemical kinetics.
# 
#    model: H-N=C=O TZVP + MP2
#    fitting Y to A0 + A2 X**2 + A4 X**4 + A6 X**6
#                 + B / ( C + X**2 ), B, C nonnegative.
# 
#    Source:
#    M. Palmer, Edinburgh, private communication.
# 
#    SIF input: Nick Gould, 1990.
# 
#    classification = "C-CSBR2-RN-6-0"
# 
#    Number of data points
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'PALMER2A'

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
        v_['X1'] = -1.745329
        v_['X2'] = -1.570796
        v_['X3'] = -1.396263
        v_['X4'] = -1.221730
        v_['X5'] = -1.047198
        v_['X6'] = -0.937187
        v_['X7'] = -0.872665
        v_['X8'] = -0.698132
        v_['X9'] = -0.523599
        v_['X10'] = -0.349066
        v_['X11'] = -0.174533
        v_['X12'] = 0.0
        v_['X13'] = 0.174533
        v_['X14'] = 0.349066
        v_['X15'] = 0.523599
        v_['X16'] = 0.698132
        v_['X17'] = 0.872665
        v_['X18'] = 0.937187
        v_['X19'] = 1.047198
        v_['X20'] = 1.221730
        v_['X21'] = 1.396263
        v_['X22'] = 1.570796
        v_['X23'] = 1.745329
        v_['Y1'] = 72.676767
        v_['Y2'] = 40.149455
        v_['Y3'] = 18.8548
        v_['Y4'] = 6.4762
        v_['Y5'] = 0.8596
        v_['Y6'] = 0.00000
        v_['Y7'] = 0.2730
        v_['Y8'] = 3.2043
        v_['Y9'] = 8.1080
        v_['Y10'] = 13.4291
        v_['Y11'] = 17.7149
        v_['Y12'] = 19.4529
        v_['Y13'] = 17.7149
        v_['Y14'] = 13.4291
        v_['Y15'] = 8.1080
        v_['Y16'] = 3.2053
        v_['Y17'] = 0.2730
        v_['Y18'] = 0.00000
        v_['Y19'] = 0.8596
        v_['Y20'] = 6.4762
        v_['Y21'] = 18.8548
        v_['Y22'] = 40.149455
        v_['Y23'] = 72.676767
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
        [iv,ix_,_] = jtu.s2mpj_ii('B',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'B')
        [iv,ix_,_] = jtu.s2mpj_ii('C',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'C')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['M'])+1):
            v_['XSQR'] = v_['X'+str(I)]*v_['X'+str(I)]
            v_['XQUART'] = v_['XSQR']*v_['XSQR']
            v_['XSEXT'] = v_['XQUART']*v_['XSQR']
            [ig,ig_,_] = jtu.s2mpj_ii('O'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A0']])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A2']])
            valA = jtu.append(valA,float(v_['XSQR']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A4']])
            valA = jtu.append(valA,float(v_['XQUART']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A6']])
            valA = jtu.append(valA,float(v_['XSEXT']))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for I in range(int(v_['1']),int(v_['M'])+1):
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
        self.xlower = jtu.np_like_set(self.xlower, ix_['B'], 0.00001)
        self.xlower = jtu.np_like_set(self.xlower, ix_['C'], 0.00001)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(1.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eQUOT', iet_)
        elftv = jtu.loaset(elftv,it,0,'B')
        elftv = jtu.loaset(elftv,it,1,'C')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'XSQR')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['M'])+1):
            v_['XSQR'] = v_['X'+str(I)]*v_['X'+str(I)]
            ename = 'E'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eQUOT')
            ielftype = jtu.arrset(ielftype,ie,iet_["eQUOT"])
            vname = 'B'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='B')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'C'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='C')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='XSQR')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['XSQR']))
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
        for I in range(int(v_['1']),int(v_['M'])+1):
            ig = ig_['O'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Least square problems are bounded below by zero
        self.objlower = 0.0
#    Solution
# LO SOLTN               1.7109717D-02
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSBR2-RN-6-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eQUOT(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        DENOM = 1.0/(EV_[1]+self.elpar[iel_][0])
        f_   = EV_[0]*DENOM
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, DENOM)
            g_ = jtu.np_like_set(g_, 1, -EV_[0]*DENOM*DENOM)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), -DENOM*DENOM)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), 2.0*EV_[0]*DENOM**3)
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


from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class POWELLBC:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : POWELLBC
#    --------
# 
#    A bound-constrained optimization problem to 
#    separate points within a square in the plane
# 
#    Given points p_j = ( x_2j-1 , x_2j ), i = 1, ..., p
# 
#    minimize sum_k=2^p sum_j=1^k-1 1 / || p_j - p_k ||_2
# 
#    subject to 0 <= x_i <= 1, i = 1, ..., 2p = n
# 
#    Source: 
#    M. J. D. Powell
#    Private communication (Optbridge, 2006)
# 
#    SIF input: Nick Gould, Aug 2006.
# 
#    classification = "C-COBR2-AN-V-0"
# 
#    Number of points
# 
#           Alternative values for the SIF file parameters:
# IE P                   2              $-PARAMETER
# IE P                   5              $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'POWELLBC'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['P'] = int(12);  #  SIF file default value
        else:
            v_['P'] = int(args[0])
# IE P                   10             $-PARAMETER
# IE P                   100            $-PARAMETER
# IE P                   500            $-PARAMETER
        v_['1'] = 1
        v_['2'] = 2
        v_['N'] = 2*v_['P']
        v_['RN'] = float(v_['N'])
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
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),0.0)
        self.xupper = jnp.full((self.n,1),1.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['T'] = v_['RI']/v_['RN']
            v_['T'] = v_['T']*v_['T']
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(v_['T']))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eINVNRM', iet_)
        elftv = jtu.loaset(elftv,it,0,'XJ')
        elftv = jtu.loaset(elftv,it,1,'YJ')
        elftv = jtu.loaset(elftv,it,2,'XK')
        elftv = jtu.loaset(elftv,it,3,'YK')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for K in range(int(v_['2']),int(v_['P'])+1):
            v_['K-1'] = K-v_['1']
            v_['2K'] = v_['2']*K
            v_['2K-1'] = v_['2K']-v_['1']
            for J in range(int(v_['1']),int(v_['K-1'])+1):
                v_['2J'] = v_['2']*J
                v_['2J-1'] = v_['2J']-v_['1']
                ename = 'E'+str(K)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'eINVNRM')
                    ielftype = jtu.arrset(ielftype,ie,iet_['eINVNRM'])
                vname = 'X'+str(int(v_['2J-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(0.0),float(1.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='XJ')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(int(v_['2K-1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(0.0),float(1.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='XK')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(int(v_['2J']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(0.0),float(1.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='YJ')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(int(v_['2K']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(0.0),float(1.0),None)
                posev = jnp.where(elftv[ielftype[ie]]=='YK')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for K in range(int(v_['2']),int(v_['P'])+1):
            v_['K-1'] = K-v_['1']
            for J in range(int(v_['1']),int(v_['K-1'])+1):
                ig = ig_['OBJ']
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(K)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               ??
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-COBR2-AN-V-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eINVNRM(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,4))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,2]), U_[0,2]-1)
        U_ = jtu.np_like_set(U_, jnp.array([1,1]), U_[1,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,3]), U_[1,3]-1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        NORM = 1.0/jnp.sqrt(IV_[0]*IV_[0]+IV_[1]*IV_[1])
        f_   = NORM
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -IV_[0]*NORM**3)
            g_ = jtu.np_like_set(g_, 1, -IV_[1]*NORM**3)
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), (3.0*IV_[0]*IV_[0]*NORM**2-1.0)*NORM**3)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 3.0*IV_[0]*IV_[1]*NORM**5)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), (3.0*IV_[1]*IV_[1]*NORM**2-1.0)*NORM**3)
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


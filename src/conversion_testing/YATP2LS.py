from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class YATP2LS:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : YATP2LS
#    *********
# 
#    Another test problem involving double pseudo-stochastic constraints
#    on a square matrix. If the matrix dimension is N, the number of
#    variables is equal to  N**2 + 2 * N. The equations are
#    x_{ij} - ( y_i + z_i ) ( 1 + cos( x_{ij} ) ) = A   (i,j = 1, ..., N )
#    \sum_i^N ( x_{ij} + sin( x_{ij}) ) = 1             (j = 1,..., N)
#    \sum_j^N ( x_{ij} + sin( x_{ij}) ) = 1             (i = 1,..., N)
#    The problem is non convex.
# 
#    Source:
#    a late evening idea by Ph. Toint
# 
#    SIF input: Ph. Toint, June 2003.
# 
#    classification = "C-CSUR2-AN-V-V"
# 
#   least-squares version, October 2014
# 
#    The dimension of the matrix
# 
#           Alternative values for the SIF file parameters:
# IE N                   2              $-PARAMETER n = 8
# IE N                   10             $-PARAMETER n = 120
# IE N                   50             $-PARAMETER n = 2600
# IE N                   100            $-PARAMETER n = 10200
# IE N                   200            $-PARAMETER n = 40400
# IE N                   350            $-PARAMETER n = 123200
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'YATP2LS'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(5);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE N                   2              $-PARAMETER n = 8
        v_['A'] = 1.0
        v_['1'] = 1
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I)+','+str(J))
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('Y'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'Y'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('Z'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'Z'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('E'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(1.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(I)]])
                valA = jtu.append(valA,float(-1.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Z'+str(J)]])
                valA = jtu.append(valA,float(-1.0))
                [ig,ig_,_] = jtu.s2mpj_ii('ER'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(1.0))
                [ig,ig_,_] = jtu.s2mpj_ii('EC'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)+','+str(J)]])
                valA = jtu.append(valA,float(1.0))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['ER'+str(I)],float(1.0))
            self.gconst = jtu.arrset(self.gconst,ig_['EC'+str(I)],float(1.0))
            for J in range(int(v_['1']),int(v_['N'])+1):
                self.gconst = jtu.arrset(self.gconst,ig_['E'+str(I)+','+str(J)],float(v_['A']))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(I)+','+str(J)]]), float(10.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eATP2', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        elftv = jtu.loaset(elftv,it,2,'Z')
        [it,iet_,_] = jtu.s2mpj_ii( 'eSINX', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                ename = 'DC'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eATP2')
                ielftype = jtu.arrset(ielftype,ie,iet_["eATP2"])
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(I)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Z'+str(I)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'SX'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eSINX')
                ielftype = jtu.arrset(ielftype,ie,iet_["eSINX"])
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                ig = ig_['E'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['DC'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                ig = ig_['ER'+str(I)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['SX'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
                ig = ig_['EC'+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['SX'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSUR2-AN-V-V"
        self.objderlvl = 2


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eATP2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,3))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,1]), U_[1,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,2]), U_[1,2]+1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        C = jnp.cos(IV_[0])
        S = jnp.sin(IV_[0])
        f_   = IV_[1]*C
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 1, C)
            g_ = jtu.np_like_set(g_, 0, -IV_[1]*S)
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), -S)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -IV_[1]*C)
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eSINX(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        S = jnp.sin(EV_[0])
        C = jnp.cos(EV_[0])
        f_   = S
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, C)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -S)
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


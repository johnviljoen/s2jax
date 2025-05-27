from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class YATP1CNE:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : YATP1CNE
#    *********
# 
#    Yet another test problem involving double pseudo-stochastic constraints
#    on a square matrix. If the matrix dimension is N, the number of
#    variables is equal to  N**2 + 2* N. The equations are
#    x_{ij}^3 - A x_{ij}^2 - ( y_i + z_j ) ( x_{ij}cos(x_{ij} - sin(x_{ij}) ) = 0
#                                                     (i,j = 1, ..., N )
#    \sum_i^N sin(x_{ij}) / x_{ij} = 1                  (j = 1,..., N)
#    \sum_j^N sin(x_{ij}) / x_{ij} = 1                  (i = 1,..., N)
#    The problem is non convex.
# 
#    Source:
#    a late evening idea by Ph. Toint
# 
#    SIF input: Ph. Toint, June 2003.
#               corrected Nick Gould, March 2019
# 
#    classification = "C-CNOR2-AN-V-V"
# 
#    The dimension of the matrix
# 
#    nonlinear equation version
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER n = 120
# IE N                   50             $-PARAMETER n = 2600
# IE N                   100            $-PARAMETER n = 10200
# IE N                   200            $-PARAMETER n = 40400
# IE N                   350            $-PARAMETER n = 123200
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'YATP1CNE'

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
        v_['A'] = 10.0
        v_['1'] = 1
        v_['-A'] = -1.0*v_['A']
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
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'E'+str(I)+','+str(J))
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('ER'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'ER'+str(I))
            [ig,ig_,_] = jtu.s2mpj_ii('EC'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'EC'+str(I))
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['ER'+str(I)],float(1.0))
            self.gconst = jtu.arrset(self.gconst,ig_['EC'+str(I)],float(1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                if('X'+str(I)+','+str(J) in ix_):
                    self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(I)+','+str(J)]]), float(6.0))
                else:
                    self.y0  = (
                          jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X'+str(I)+','+str(J)]),float(6.0)))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        [it,iet_,_] = jtu.s2mpj_ii( 'eCB', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        [it,iet_,_] = jtu.s2mpj_ii( 'eLXC', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        elftv = jtu.loaset(elftv,it,2,'Z')
        [it,iet_,_] = jtu.s2mpj_ii( 'eLS', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        elftv = jtu.loaset(elftv,it,2,'Z')
        [it,iet_,_] = jtu.s2mpj_ii( 'eRAT', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                ename = 'CB'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eCB')
                ielftype = jtu.arrset(ielftype,ie,iet_["eCB"])
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'SQ'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'DC'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eLXC')
                ielftype = jtu.arrset(ielftype,ie,iet_["eLXC"])
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(I)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Z'+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'DS'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eLS')
                ielftype = jtu.arrset(ielftype,ie,iet_["eLS"])
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Y'+str(I)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'Z'+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                ename = 'SX'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eRAT')
                ielftype = jtu.arrset(ielftype,ie,iet_["eRAT"])
                vname = 'X'+str(I)+','+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                ig = ig_['E'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['CB'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['SQ'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-A']))
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['DC'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
                posel = posel+1
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['DS'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['N'])+1):
                ig = ig_['ER'+str(I)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['SX'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
                ig = ig_['EC'+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['SX'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CNOR2-AN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eSQ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[0]+EV_[0])
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eCB(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[0]*EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 3.0*EV_[0]*EV_[0])
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 6.0*EV_[0])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eLXC(self, nargout,*args):

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
        f_   = IV_[1]*IV_[0]*C
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 1, IV_[0]*C)
            g_ = jtu.np_like_set(g_, 0, IV_[1]*(C-IV_[0]*S))
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), C-IV_[0]*S)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -IV_[1]*(S+S+IV_[0]*C))
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eLS(self, nargout,*args):

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
        f_   = IV_[1]*S
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 1, S)
            g_ = jtu.np_like_set(g_, 0, IV_[1]*C)
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), C)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -IV_[1]*S)
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eRAT(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        C = jnp.cos(EV_[0])
        S = jnp.sin(EV_[0])
        f_   = S/EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, (C-S/EV_[0])/EV_[0])
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -S/EV_[0]-(C+C)/EV_[0]**2+(S+S)/EV_[0]**3)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


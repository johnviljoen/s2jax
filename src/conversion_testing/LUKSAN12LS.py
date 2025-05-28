import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LUKSAN12LS:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : LUKSAN12LS
#    *********
# 
#    Problem 12 (chained and modified HS47) in the paper
# 
#      L. Luksan
#      Hybrid methods in large sparse nonlinear least squares
#      J. Optimization Theory & Applications 89(3) 575-595 (1996)
# 
#    SIF input: Nick Gould, June 2017.
# 
#    least-squares version
# 
#    classification = "C-CSUR2-AN-V-0"
# 
#   seed for dimensions
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LUKSAN12LS'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['S'] = 32
        v_['N'] = 3*v_['S']
        v_['N'] = 2+v_['N']
        v_['M'] = 6*v_['S']
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
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        v_['I'] = 1
        v_['K'] = 1
        for J in range(int(v_['1']),int(v_['S'])+1):
            v_['K+1'] = 1+v_['K']
            v_['K+2'] = 2+v_['K']
            v_['K+3'] = 3+v_['K']
            v_['K+4'] = 4+v_['K']
            v_['K+5'] = 5+v_['K']
            v_['I+1'] = 1+v_['I']
            v_['I+2'] = 2+v_['I']
            [ig,ig_,_] = jtu.s2mpj_ii('E'+str(int(v_['K'])),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['I+1']))]])
            valA = jtu.append(valA,float(-10.0e0))
            [ig,ig_,_] = jtu.s2mpj_ii('E'+str(int(v_['K+1'])),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['I+2']))]])
            valA = jtu.append(valA,float(1.0e0))
            [ig,ig_,_] = jtu.s2mpj_ii('E'+str(int(v_['K+2'])),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            [ig,ig_,_] = jtu.s2mpj_ii('E'+str(int(v_['K+3'])),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            [ig,ig_,_] = jtu.s2mpj_ii('E'+str(int(v_['K+4'])),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            [ig,ig_,_] = jtu.s2mpj_ii('E'+str(int(v_['K+5'])),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['I+1']))]])
            valA = jtu.append(valA,float(1.0e0))
            v_['I'] = 3+v_['I']
            v_['K'] = 6+v_['K']
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        v_['K'] = 1
        for J in range(int(v_['1']),int(v_['S'])+1):
            v_['K+1'] = 1+v_['K']
            v_['K+4'] = 4+v_['K']
            v_['K+5'] = 5+v_['K']
            self.gconst = jtu.arrset(self.gconst,ig_['E'+str(int(v_['K+1']))],float(1.0))
            self.gconst = jtu.arrset(self.gconst,ig_['E'+str(int(v_['K+4']))],float(10.0))
            self.gconst = jtu.arrset(self.gconst,ig_['E'+str(int(v_['K+5']))],float(20.0))
            v_['K'] = 6+v_['K']
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(-1.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eE1', iet_)
        elftv = jtu.loaset(elftv,it,0,'X0')
        [it,iet_,_] = jtu.s2mpj_ii( 'eE3', iet_)
        elftv = jtu.loaset(elftv,it,0,'X3')
        [it,iet_,_] = jtu.s2mpj_ii( 'eE4', iet_)
        elftv = jtu.loaset(elftv,it,0,'X4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eE5', iet_)
        elftv = jtu.loaset(elftv,it,0,'X0')
        elftv = jtu.loaset(elftv,it,1,'X3')
        [it,iet_,_] = jtu.s2mpj_ii( 'eF5', iet_)
        elftv = jtu.loaset(elftv,it,0,'X3')
        elftv = jtu.loaset(elftv,it,1,'X4')
        [it,iet_,_] = jtu.s2mpj_ii( 'eE6', iet_)
        elftv = jtu.loaset(elftv,it,0,'X2')
        elftv = jtu.loaset(elftv,it,1,'X3')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        v_['I'] = 1
        v_['K'] = 1
        for J in range(int(v_['1']),int(v_['S'])+1):
            v_['K+2'] = 2+v_['K']
            v_['K+3'] = 3+v_['K']
            v_['K+4'] = 4+v_['K']
            v_['K+5'] = 5+v_['K']
            v_['I+2'] = 2+v_['I']
            v_['I+3'] = 3+v_['I']
            v_['I+4'] = 4+v_['I']
            ename = 'E'+str(int(v_['K']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eE1')
            ielftype = jtu.arrset(ielftype,ie,iet_["eE1"])
            ename = 'E'+str(int(v_['K']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X0')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'E'+str(int(v_['K+2']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eE3')
            ielftype = jtu.arrset(ielftype,ie,iet_["eE3"])
            ename = 'E'+str(int(v_['K+2']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+3']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X3')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'E'+str(int(v_['K+3']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eE4')
            ielftype = jtu.arrset(ielftype,ie,iet_["eE4"])
            ename = 'E'+str(int(v_['K+3']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+4']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X4')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'E'+str(int(v_['K+4']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eE5')
            ielftype = jtu.arrset(ielftype,ie,iet_["eE5"])
            ename = 'E'+str(int(v_['K+4']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X0')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'E'+str(int(v_['K+4']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+3']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X3')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'F'+str(int(v_['K+4']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eF5')
            ielftype = jtu.arrset(ielftype,ie,iet_["eF5"])
            ename = 'F'+str(int(v_['K+4']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+3']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X3')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'F'+str(int(v_['K+4']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+4']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X4')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'E'+str(int(v_['K+5']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eE6')
            ielftype = jtu.arrset(ielftype,ie,iet_["eE6"])
            ename = 'E'+str(int(v_['K+5']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+2']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'E'+str(int(v_['K+5']))
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            vname = 'X'+str(int(v_['I+3']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X3')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['I'] = 3+v_['I']
            v_['K'] = 6+v_['K']
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
        v_['K'] = 1
        for J in range(int(v_['1']),int(v_['S'])+1):
            v_['K+2'] = 2+v_['K']
            v_['K+3'] = 3+v_['K']
            v_['K+4'] = 4+v_['K']
            v_['K+5'] = 5+v_['K']
            ig = ig_['E'+str(int(v_['K']))]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(int(v_['K']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            ig = ig_['E'+str(int(v_['K+2']))]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(int(v_['K+2']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            ig = ig_['E'+str(int(v_['K+3']))]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(int(v_['K+3']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            ig = ig_['E'+str(int(v_['K+4']))]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(int(v_['K+4']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['F'+str(int(v_['K+4']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
            ig = ig_['E'+str(int(v_['K+5']))]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(int(v_['K+5']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            v_['K'] = 6+v_['K']
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN                0.0
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSUR2-AN-V-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eE1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = 10.0e0*EV_[0]**2
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 20.0e0*EV_[0])
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 20.0e0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eE3(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = (EV_[0]-1.0e0)**2
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 2.0e0*(EV_[0]-1.0e0))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0e0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eE4(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = (EV_[0]-1.0e0)**3
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 3.0e0*(EV_[0]-1.0e0)**2)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 6.0e0*(EV_[0]-1.0e0))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eE5(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[1]*EV_[0]*EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 2.0e0*EV_[1]*EV_[0])
            g_ = jtu.np_like_set(g_, 1, EV_[0]*EV_[0])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0e0*EV_[1])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 2.0e0*EV_[0])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eF5(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((1,2))
        IV_ = jnp.zeros(1)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]-1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        f_   = jnp.sin(IV_[0])
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, jnp.cos(IV_[0]))
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -jnp.sin(IV_[0]))
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eE6(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = (EV_[0]**4)*(EV_[1]**2)
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 4.0e0*(EV_[0]**3)*(EV_[1]**2))
            g_ = jtu.np_like_set(g_, 1, 2.0e0*(EV_[0]**4)*EV_[1])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 12.0e0*(EV_[0]**2)*(EV_[1]**2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 8.0e0*(EV_[0]**3)*EV_[1])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), 2.0e0*(EV_[0]**4))
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


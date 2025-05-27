from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class BROWNAL:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : BROWNAL
#    *********
#    Brown almost linear least squares problem.
#    This problem is a sum of n least-squares groups, the last one of
#    which has a nonlinear element.
#    It Hessian matrix is dense.
# 
#    Source: Problem 27 in
#    J.J. More', B.S. Garbow and K.E. Hillstrom,
#    "Testing Unconstrained Optimization Software",
#    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.
# 
#    See also Buckley#79
#    SIF input: Ph. Toint, Dec 1989.
# 
#    classification = "C-CSUR2-AN-V-0"
# 
#    N is the number of free variables (variable).
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER     original value
# IE N                   100            $-PARAMETER
# IE N                   200            $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'BROWNAL'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(10);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE N                   1000           $-PARAMETER
        v_['1'] = 1
        v_['N-1'] = -1+v_['N']
        v_['N+1'] = 1+v_['N']
        v_['RN+1'] = float(v_['N+1'])
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
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            for J in range(int(v_['1']),int(v_['I-1'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(2.0))
            for J in range(int(v_['I+1']),int(v_['N'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'<>')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(J)]])
                valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('G'+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['G'+str(I)],float(v_['RN+1']))
        self.gconst = jtu.arrset(self.gconst,ig_['G'+str(int(v_['N']))],float(1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(0.5))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        elftv = jtu.loaset(elftv,it,1,'V2')
        elftv = jtu.loaset(elftv,it,2,'V3')
        elftv = jtu.loaset(elftv,it,3,'V4')
        elftv = jtu.loaset(elftv,it,4,'V5')
        elftv = jtu.loaset(elftv,it,5,'V6')
        elftv = jtu.loaset(elftv,it,6,'V7')
        elftv = jtu.loaset(elftv,it,7,'V8')
        elftv = jtu.loaset(elftv,it,8,'V9')
        elftv = jtu.loaset(elftv,it,9,'V10')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        ename = 'E'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V4')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V5')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V6')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V7')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V8')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X9'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V9')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X10'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='V10')[0]
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
        ig = ig_['G'+str(int(v_['N']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               0.0
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
    def ePROD(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        V12 = EV_[0]*EV_[1]
        V34 = EV_[2]*EV_[3]
        V56 = EV_[4]*EV_[5]
        V78 = EV_[6]*EV_[7]
        V910 = EV_[8]*EV_[9]
        f_   = V12*V34*V56*V78*V910
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[1]*V34*V56*V78*V910)
            g_ = jtu.np_like_set(g_, 1, EV_[0]*V34*V56*V78*V910)
            g_ = jtu.np_like_set(g_, 2, V12*EV_[3]*V56*V78*V910)
            g_ = jtu.np_like_set(g_, 3, V12*EV_[2]*V56*V78*V910)
            g_ = jtu.np_like_set(g_, 4, V12*V34*EV_[5]*V78*V910)
            g_ = jtu.np_like_set(g_, 5, V12*V34*EV_[4]*V78*V910)
            g_ = jtu.np_like_set(g_, 6, V12*V34*V56*EV_[7]*V910)
            g_ = jtu.np_like_set(g_, 7, V12*V34*V56*EV_[6]*V910)
            g_ = jtu.np_like_set(g_, 8, V12*V34*V56*V78*EV_[9])
            g_ = jtu.np_like_set(g_, 9, V12*V34*V56*V78*EV_[8])
            if nargout>2:
                H_ = jnp.zeros((10,10))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), V34*V56*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), EV_[1]*EV_[3]*V56*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), EV_[1]*EV_[2]*V56*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([0,4]), EV_[1]*V34*EV_[5]*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([4,0]), H_[0,4])
                H_ = jtu.np_like_set(H_, jnp.array([0,5]), EV_[1]*V34*EV_[4]*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([5,0]), H_[0,5])
                H_ = jtu.np_like_set(H_, jnp.array([0,6]), EV_[1]*V34*V56*EV_[7]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([6,0]), H_[0,6])
                H_ = jtu.np_like_set(H_, jnp.array([0,7]), EV_[1]*V34*V56*EV_[6]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([7,0]), H_[0,7])
                H_ = jtu.np_like_set(H_, jnp.array([0,8]), EV_[1]*V34*V56*V78*EV_[9])
                H_ = jtu.np_like_set(H_, jnp.array([8,0]), H_[0,8])
                H_ = jtu.np_like_set(H_, jnp.array([0,9]), EV_[1]*V34*V56*V78*EV_[8])
                H_ = jtu.np_like_set(H_, jnp.array([9,0]), H_[0,9])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), EV_[0]*EV_[3]*V56*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), EV_[0]*EV_[2]*V56*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,4]), EV_[0]*V34*EV_[5]*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([4,1]), H_[1,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,5]), EV_[0]*V34*EV_[4]*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([5,1]), H_[1,5])
                H_ = jtu.np_like_set(H_, jnp.array([1,6]), EV_[0]*V34*V56*EV_[7]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([6,1]), H_[1,6])
                H_ = jtu.np_like_set(H_, jnp.array([1,7]), EV_[0]*V34*V56*EV_[6]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([7,1]), H_[1,7])
                H_ = jtu.np_like_set(H_, jnp.array([1,8]), EV_[0]*V34*V56*V78*EV_[9])
                H_ = jtu.np_like_set(H_, jnp.array([8,1]), H_[1,8])
                H_ = jtu.np_like_set(H_, jnp.array([1,9]), EV_[0]*V34*V56*V78*EV_[8])
                H_ = jtu.np_like_set(H_, jnp.array([9,1]), H_[1,9])
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), V12*V56*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,4]), V12*EV_[3]*EV_[5]*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([4,2]), H_[2,4])
                H_ = jtu.np_like_set(H_, jnp.array([2,5]), V12*EV_[3]*EV_[4]*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([5,2]), H_[2,5])
                H_ = jtu.np_like_set(H_, jnp.array([2,6]), V12*EV_[3]*V56*EV_[7]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([6,2]), H_[2,6])
                H_ = jtu.np_like_set(H_, jnp.array([2,7]), V12*EV_[3]*V56*EV_[6]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([7,2]), H_[2,7])
                H_ = jtu.np_like_set(H_, jnp.array([2,8]), V12*EV_[3]*V56*V78*EV_[9])
                H_ = jtu.np_like_set(H_, jnp.array([8,2]), H_[2,8])
                H_ = jtu.np_like_set(H_, jnp.array([2,9]), V12*EV_[3]*V56*V78*EV_[8])
                H_ = jtu.np_like_set(H_, jnp.array([9,2]), H_[2,9])
                H_ = jtu.np_like_set(H_, jnp.array([3,4]), V12*EV_[2]*EV_[5]*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([4,3]), H_[3,4])
                H_ = jtu.np_like_set(H_, jnp.array([3,5]), V12*EV_[2]*EV_[4]*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([5,3]), H_[3,5])
                H_ = jtu.np_like_set(H_, jnp.array([3,6]), V12*EV_[2]*V56*EV_[7]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([6,3]), H_[3,6])
                H_ = jtu.np_like_set(H_, jnp.array([3,7]), V12*EV_[2]*V56*EV_[6]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([7,3]), H_[3,7])
                H_ = jtu.np_like_set(H_, jnp.array([3,8]), V12*EV_[2]*V56*V78*EV_[9])
                H_ = jtu.np_like_set(H_, jnp.array([8,3]), H_[3,8])
                H_ = jtu.np_like_set(H_, jnp.array([3,9]), V12*EV_[2]*V56*V78*EV_[8])
                H_ = jtu.np_like_set(H_, jnp.array([9,3]), H_[3,9])
                H_ = jtu.np_like_set(H_, jnp.array([4,5]), V12*V34*V78*V910)
                H_ = jtu.np_like_set(H_, jnp.array([5,4]), H_[4,5])
                H_ = jtu.np_like_set(H_, jnp.array([4,6]), V12*V34*EV_[5]*EV_[7]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([6,4]), H_[4,6])
                H_ = jtu.np_like_set(H_, jnp.array([4,7]), V12*V34*EV_[5]*EV_[6]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([7,4]), H_[4,7])
                H_ = jtu.np_like_set(H_, jnp.array([4,8]), V12*V34*EV_[5]*V78*EV_[9])
                H_ = jtu.np_like_set(H_, jnp.array([8,4]), H_[4,8])
                H_ = jtu.np_like_set(H_, jnp.array([4,9]), V12*V34*EV_[5]*V78*EV_[8])
                H_ = jtu.np_like_set(H_, jnp.array([9,4]), H_[4,9])
                H_ = jtu.np_like_set(H_, jnp.array([5,6]), V12*V34*EV_[4]*EV_[7]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([6,5]), H_[5,6])
                H_ = jtu.np_like_set(H_, jnp.array([5,7]), V12*V34*EV_[4]*EV_[6]*V910)
                H_ = jtu.np_like_set(H_, jnp.array([7,5]), H_[5,7])
                H_ = jtu.np_like_set(H_, jnp.array([5,8]), V12*V34*EV_[4]*V78*EV_[9])
                H_ = jtu.np_like_set(H_, jnp.array([8,5]), H_[5,8])
                H_ = jtu.np_like_set(H_, jnp.array([5,9]), V12*V34*EV_[4]*V78*EV_[8])
                H_ = jtu.np_like_set(H_, jnp.array([9,5]), H_[5,9])
                H_ = jtu.np_like_set(H_, jnp.array([6,7]), V12*V34*V56*V910)
                H_ = jtu.np_like_set(H_, jnp.array([7,6]), H_[6,7])
                H_ = jtu.np_like_set(H_, jnp.array([6,8]), V12*V34*V56*EV_[7]*EV_[9])
                H_ = jtu.np_like_set(H_, jnp.array([8,6]), H_[6,8])
                H_ = jtu.np_like_set(H_, jnp.array([6,9]), V12*V34*V56*EV_[7]*EV_[8])
                H_ = jtu.np_like_set(H_, jnp.array([9,6]), H_[6,9])
                H_ = jtu.np_like_set(H_, jnp.array([7,8]), V12*V34*V56*EV_[6]*EV_[9])
                H_ = jtu.np_like_set(H_, jnp.array([8,7]), H_[7,8])
                H_ = jtu.np_like_set(H_, jnp.array([7,9]), V12*V34*V56*EV_[6]*EV_[8])
                H_ = jtu.np_like_set(H_, jnp.array([9,7]), H_[7,9])
                H_ = jtu.np_like_set(H_, jnp.array([8,9]), V12*V34*V56*V78)
                H_ = jtu.np_like_set(H_, jnp.array([9,8]), H_[8,9])
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


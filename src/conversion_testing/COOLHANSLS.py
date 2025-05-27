from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class COOLHANSLS:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : COOLHANSLS
#    *********
# 
#    A problem arising from the analysis of a Cooley-Hansen economy with
#    loglinear approximation.  The problem is to solve the matrix equation
#                  A * X * X + B * X + C = 0
#    where A, B and C are known N times N matrices and X an unknown matrix
#    of matching dimension.  The instance considered here has N = 3.
# 
#    Source:
#    S. Ceria, private communication, 1995.
# 
#    SIF input: Ph. Toint, Feb 1995.
#    Least-squares version of COOLHANS.SIF, Nick Gould, Jan 2020.
# 
#    classification = "C-CSUR2-RN-9-0"
# 
#    order of the matrix equation
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'COOLHANSLS'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 3
        v_['A1,1'] = 0.0
        v_['A2,1'] = 0.13725e-6
        v_['A3,1'] = 0.0
        v_['A1,2'] = 0.0
        v_['A2,2'] = 937.62
        v_['A3,2'] = 0.0
        v_['A1,3'] = 0.0
        v_['A2,3'] = -42.207
        v_['A3,3'] = 0.0
        v_['B1,1'] = 0.0060893
        v_['B2,1'] = 0.13880e-6
        v_['B3,1'] = -0.13877e-6
        v_['B1,2'] = -44.292
        v_['B2,2'] = -1886.0
        v_['B3,2'] = 42.362
        v_['B1,3'] = 2.0011
        v_['B2,3'] = 42.362
        v_['B3,3'] = -2.0705
        v_['C1,1'] = 0.0
        v_['C2,1'] = 0.0
        v_['C3,1'] = 0.0
        v_['C1,2'] = 44.792
        v_['C2,2'] = 948.21
        v_['C3,2'] = -42.684
        v_['C1,3'] = 0.0
        v_['C2,3'] = 0.0
        v_['C3,3'] = 0.0
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
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
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for K in range(int(v_['1']),int(v_['N'])+1):
            for L in range(int(v_['1']),int(v_['N'])+1):
                for M in range(int(v_['1']),int(v_['N'])+1):
                    [ig,ig_,_] = jtu.s2mpj_ii('G'+str(K)+','+str(L),ig_)
                    gtype = jtu.arrset(gtype,ig,'<>')
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(M)+','+str(L)]])
                    valA = jtu.append(valA,float(v_['B'+str(K)+','+str(M)]))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for K in range(int(v_['1']),int(v_['N'])+1):
            for L in range(int(v_['1']),int(v_['N'])+1):
                v_['-C'] = -1.0*v_['C'+str(K)+','+str(L)]
                self.gconst = jtu.arrset(self.gconst,ig_['G'+str(K)+','+str(L)],float(v_['-C']))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'en2PR', iet_)
        elftv = jtu.loaset(elftv,it,0,'XX')
        elftv = jtu.loaset(elftv,it,1,'YY')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for K in range(int(v_['1']),int(v_['N'])+1):
            for L in range(int(v_['1']),int(v_['N'])+1):
                for M in range(int(v_['1']),int(v_['N'])+1):
                    ename = 'E'+str(K)+','+str(M)+','+str(L)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
                    ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
                    self.x0 = jnp.zeros((self.n,1))
                    vname = 'X'+str(K)+','+str(M)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='XX')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(M)+','+str(L)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='YY')[0]
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
        for L in range(int(v_['1']),int(v_['N'])+1):
            for P in range(int(v_['1']),int(v_['N'])+1):
                for M in range(int(v_['1']),int(v_['N'])+1):
                    ig = ig_['G'+str(int(v_['1']))+','+str(L)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['E'+str(P)+','+str(M)+','+str(L)]))
                    self.grelw  = (
                          jtu.loaset(self.grelw,ig,posel,float(v_['A'+str(int(v_['1']))+','+str(P)])))
        for L in range(int(v_['1']),int(v_['N'])+1):
            for P in range(int(v_['1']),int(v_['N'])+1):
                for M in range(int(v_['1']),int(v_['N'])+1):
                    ig = ig_['G'+str(int(v_['2']))+','+str(L)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['E'+str(P)+','+str(M)+','+str(L)]))
                    self.grelw  = (
                          jtu.loaset(self.grelw,ig,posel,float(v_['A'+str(int(v_['2']))+','+str(P)])))
        for L in range(int(v_['1']),int(v_['N'])+1):
            for P in range(int(v_['1']),int(v_['N'])+1):
                for M in range(int(v_['1']),int(v_['N'])+1):
                    ig = ig_['G'+str(int(v_['3']))+','+str(L)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['E'+str(P)+','+str(M)+','+str(L)]))
                    self.grelw  = (
                          jtu.loaset(self.grelw,ig,posel,float(v_['A'+str(int(v_['3']))+','+str(P)])))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN                0.0
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSUR2-RN-9-0"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def en2PR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[1])
            g_ = jtu.np_like_set(g_, 1, EV_[0])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
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


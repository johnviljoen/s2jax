import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LEVYMONT9:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : LEVYMONT9
#    *********
#    A global optimization example due to Levy & Montalvo 
#    This problem is one of the parameterised set LEVYMONT5-LEVYMONT10
# 
#    Source:  problem 9 in
# 
#    A. V. Levy and A. Montalvo
#    "The Tunneling Algorithm for the Global Minimization of Functions"
#    SIAM J. Sci. Stat. Comp. 6(1) 1985 15:29 
#    https://doi.org/10.1137/0906002
# 
#    SIF input: Nick Gould, August 2021
# 
#    classification = "C-CSBR2-AY-8-0"
# 
#    N is the number of variables
# 
#           Alternative values for the SIF file parameters:
# IE N                   8              $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LEVYMONT9'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(8);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
        v_['A'] = 1.0
        v_['K'] = 10.0
        v_['L'] = 1.0
        v_['C'] = 0.0
        v_['1'] = 1
        v_['2'] = 2
        v_['PI/4'] = jnp.arctan(1.0)
        v_['PI'] = 4.0*v_['PI/4']
        v_['RN'] = float(v_['N'])
        v_['A-C'] = v_['A']-v_['C']
        v_['PI/N'] = v_['PI']/v_['RN']
        v_['KPI/N'] = v_['K']*v_['PI/N']
        v_['ROOTKPI/N'] = jnp.sqrt(v_['KPI/N'])
        v_['N/PI'] = v_['RN']/v_['PI']
        v_['N/KPI'] = v_['N/PI']/v_['K']
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('Q'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(v_['L']))
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['N/PI']))
            [ig,ig_,_] = jtu.s2mpj_ii('N'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['Q'+str(I)],float(v_['A-C']))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-10.0)
        self.xupper = jnp.full((self.n,1),10.0)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(8.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(-8.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X2'], float(8.0))
        pass
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eS2', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'L')
        elftp = jtu.loaset(elftp,it,1,'C')
        [it,iet_,_] = jtu.s2mpj_ii( 'ePS2', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Z')
        elftp = jtu.loaset(elftp,it,0,'L')
        elftp = jtu.loaset(elftp,it,1,'C')
        elftp = jtu.loaset(elftp,it,2,'A')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'E'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eS2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eS2"])
        ename = 'E'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(-10.0),float(10.0),float(8.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        posep = jnp.where(elftp[ielftype[ie]]=='L')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['L']))
        ename = 'E'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        posep = jnp.where(elftp[ielftype[ie]]=='C')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['C']))
        for I in range(int(v_['2']),int(v_['N'])+1):
            v_['I-1'] = I-v_['1']
            ename = 'E'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'ePS2')
            ielftype = jtu.arrset(ielftype,ie,iet_["ePS2"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(-10.0),float(10.0),float(8.0))
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['I-1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(-10.0),float(10.0),float(8.0))
            posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='L')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['L']))
            posep = jnp.where(elftp[ielftype[ie]]=='C')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['C']))
            posep = jnp.where(elftp[ielftype[ie]]=='A')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['A']))
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['Q'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            ig = ig_['N'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['ROOTKPI/N']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               0.0
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSBR2-AY-8-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def e_globs(self):

        import jax.numpy as jnp
        self.efpar = jnp.array([])
        self.efpar = jtu.arrset( self.efpar,0,4.0*jnp.arctan(1.0e0))
        return pbm

    @staticmethod
    def eS2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        PIL = self.efpar[0]*self.elpar[iel_][0]
        V = PIL*EV_[0]+self.efpar[0]*self.elpar[iel_][1]
        SINV = jnp.sin(V)
        COSV = jnp.cos(V)
        f_   = SINV
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, PIL*COSV)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -PIL*PIL*SINV)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def ePS2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        PIL = self.efpar[0]*self.elpar[iel_][0]
        U = self.elpar[iel_][0]*EV_[1]+self.elpar[iel_][1]-self.elpar[iel_][2]
        V = PIL*EV_[0]+self.efpar[0]*self.elpar[iel_][1]
        SINV = jnp.sin(V)
        COSV = jnp.cos(V)
        f_   = U*SINV
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, PIL*U*COSV)
            g_ = jtu.np_like_set(g_, 1, self.elpar[iel_][0]*SINV)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -PIL*PIL*U*SINV)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), self.elpar[iel_][0]*PIL*COSV)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), 0.0)
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
            g_ = 2.0*GVAR_
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


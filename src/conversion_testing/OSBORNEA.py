import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class OSBORNEA:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : OSBORNEA
#    *********
# 
#    Osborne first problem in 5 variables.
# 
#    This function  is a nonlinear least squares with 33 groups.  Each
#    group has 2 nonlinear elements and one linear element.
# 
#    Source:  Problem 17 in
#    J.J. More', B.S. Garbow and K.E. Hillstrom,
#    "Testing Unconstrained Optimization Software",
#    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.
# 
#    See alos Buckley#32 (p. 77).
# 
#    SIF input: Ph. Toint, Dec 1989.
# 
#    classification = "C-CSUR2-MN-5-0"
# 
#    Number of groups
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'OSBORNEA'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['M'] = 33
        v_['1'] = 1
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('X1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X1')
        [iv,ix_,_] = jtu.s2mpj_ii('X2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X2')
        [iv,ix_,_] = jtu.s2mpj_ii('X3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X3')
        [iv,ix_,_] = jtu.s2mpj_ii('X4',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X4')
        [iv,ix_,_] = jtu.s2mpj_ii('X5',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X5')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['M'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X1']])
            valA = jtu.append(valA,float(1.0))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        self.gconst = jtu.arrset(self.gconst,ig_['G1'],float(0.844))
        self.gconst = jtu.arrset(self.gconst,ig_['G2'],float(0.908))
        self.gconst = jtu.arrset(self.gconst,ig_['G3'],float(0.932))
        self.gconst = jtu.arrset(self.gconst,ig_['G4'],float(0.936))
        self.gconst = jtu.arrset(self.gconst,ig_['G5'],float(0.925))
        self.gconst = jtu.arrset(self.gconst,ig_['G6'],float(0.908))
        self.gconst = jtu.arrset(self.gconst,ig_['G7'],float(0.881))
        self.gconst = jtu.arrset(self.gconst,ig_['G8'],float(0.850))
        self.gconst = jtu.arrset(self.gconst,ig_['G9'],float(0.818))
        self.gconst = jtu.arrset(self.gconst,ig_['G10'],float(0.784))
        self.gconst = jtu.arrset(self.gconst,ig_['G11'],float(0.751))
        self.gconst = jtu.arrset(self.gconst,ig_['G12'],float(0.718))
        self.gconst = jtu.arrset(self.gconst,ig_['G13'],float(0.685))
        self.gconst = jtu.arrset(self.gconst,ig_['G14'],float(0.658))
        self.gconst = jtu.arrset(self.gconst,ig_['G15'],float(0.628))
        self.gconst = jtu.arrset(self.gconst,ig_['G16'],float(0.603))
        self.gconst = jtu.arrset(self.gconst,ig_['G17'],float(0.580))
        self.gconst = jtu.arrset(self.gconst,ig_['G18'],float(0.558))
        self.gconst = jtu.arrset(self.gconst,ig_['G19'],float(0.538))
        self.gconst = jtu.arrset(self.gconst,ig_['G20'],float(0.522))
        self.gconst = jtu.arrset(self.gconst,ig_['G21'],float(0.506))
        self.gconst = jtu.arrset(self.gconst,ig_['G22'],float(0.490))
        self.gconst = jtu.arrset(self.gconst,ig_['G23'],float(0.478))
        self.gconst = jtu.arrset(self.gconst,ig_['G24'],float(0.467))
        self.gconst = jtu.arrset(self.gconst,ig_['G25'],float(0.457))
        self.gconst = jtu.arrset(self.gconst,ig_['G26'],float(0.448))
        self.gconst = jtu.arrset(self.gconst,ig_['G27'],float(0.438))
        self.gconst = jtu.arrset(self.gconst,ig_['G28'],float(0.431))
        self.gconst = jtu.arrset(self.gconst,ig_['G29'],float(0.424))
        self.gconst = jtu.arrset(self.gconst,ig_['G30'],float(0.420))
        self.gconst = jtu.arrset(self.gconst,ig_['G31'],float(0.414))
        self.gconst = jtu.arrset(self.gconst,ig_['G32'],float(0.411))
        self.gconst = jtu.arrset(self.gconst,ig_['G33'],float(0.406))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(0.5))
        self.x0 = jtu.np_like_set(self.x0, ix_['X2'], float(1.5))
        self.x0 = jtu.np_like_set(self.x0, ix_['X3'], float(-1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X4'], float(0.01))
        self.x0 = jtu.np_like_set(self.x0, ix_['X5'], float(0.02))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePEXP', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        elftv = jtu.loaset(elftv,it,1,'V2')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'T')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['M'])+1):
            v_['I-1'] = -1+I
            v_['ITI'] = 10*v_['I-1']
            v_['MTI'] = float(v_['ITI'])
            v_['TI'] = -1.0*v_['MTI']
            ename = 'A'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'ePEXP')
            ielftype = jtu.arrset(ielftype,ie,iet_["ePEXP"])
            vname = 'X2'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X4'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='T')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['TI']))
            ename = 'B'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'ePEXP')
            ielftype = jtu.arrset(ielftype,ie,iet_["ePEXP"])
            vname = 'X3'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X5'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='T')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['TI']))
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
            ig = ig_['G'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['A'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['B'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Least square problems are bounded below by zero
        self.objlower = 0.0
#    Solution
# LO SOLTN               5.46489D-05
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSUR2-MN-5-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePEXP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        EXPA = jnp.exp(self.elpar[iel_][0]*EV_[1])
        V1EXPA = EV_[0]*EXPA
        f_   = V1EXPA
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EXPA)
            g_ = jtu.np_like_set(g_, 1, self.elpar[iel_][0]*V1EXPA)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), self.elpar[iel_][0]*EXPA)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), self.elpar[iel_][0]*self.elpar[iel_][0]*V1EXPA)
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


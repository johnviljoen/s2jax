from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class BOXBODLS:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : BOXBODLS
#    *********
# 
#    NIST Data fitting problem BOXBOD.
# 
#    Fit: y = b1*(1-exp[-b2*x]) + e
# 
#    Source:  Problem from the NIST nonlinear regression test set
#      http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml
# 
# 
#    Reference: Box, G. P., W. G. Hunter, and J. S. Hunter (1978).
#      Statistics for Experimenters, New York, NY: Wiley, pp. 483-487.
# 
#    SIF input: Nick Gould and Tyrone Rees, Oct 2015
# 
#    classification = "C-CSUR2-MN-2-0"
# 
#    Number of data values
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'BOXBODLS'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['M'] = 6
        v_['N'] = 2
        v_['1'] = 1
        v_['X1'] = 1.0
        v_['X2'] = 2.0
        v_['X3'] = 3.0
        v_['X4'] = 5.0
        v_['X5'] = 7.0
        v_['X6'] = 10.0
        v_['Y1'] = 109.0
        v_['Y2'] = 149.0
        v_['Y3'] = 149.0
        v_['Y4'] = 191.0
        v_['Y5'] = 213.0
        v_['Y6'] = 224.0
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('B'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'B'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['M'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('F'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['B1']])
            valA = jtu.append(valA,float(1.0))
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
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['B1'], float(1.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['B2'], float(1.0))
        pass
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eE2', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        elftv = jtu.loaset(elftv,it,1,'V2')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'X')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['M'])+1):
            ename = 'E'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eE2')
            ielftype = jtu.arrset(ielftype,ie,iet_["eE2"])
            vname = 'B1'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'B2'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='X')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['X'+str(I)]))
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
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Least square problems are bounded below by zero
        self.objlower = 0.0
#    Solution
# LO SOLTN               
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSUR2-MN-2-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eE2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        E = jnp.exp(-EV_[1]*self.elpar[iel_][0])
        f_   = EV_[0]*E
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, E)
            g_ = jtu.np_like_set(g_, 1, -EV_[0]*self.elpar[iel_][0]*E)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), -self.elpar[iel_][0]*E)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), EV_[0]*self.elpar[iel_][0]*self.elpar[iel_][0]*E)
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


from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class SNAIL:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : SNAIL
#    *********
# 
#    A 2D problem featuring a spiraling valley.
#    Dedicated to the city of Namur, whose emblem is a snail.
# 
#    Source:
#    J. Engels, private communication.
# 
#    SIF input: Ph. Toint, May 1990.
# 
#    classification = "C-COUR2-AN-2-0"
# 
#    Problem parameters (CUP > CLOW > 0)
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'SNAIL'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['CLOW'] = 1.0
        v_['CUP'] = 2.0
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
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(10.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X2'], float(10.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSPIRAL', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'CL')
        elftp = jtu.loaset(elftp,it,1,'CU')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'E'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSPIRAL')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSPIRAL"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='CL')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CLOW']))
        posep = jnp.where(elftp[ielftype[ie]]=='CU')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['CUP']))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['OBJ']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               0.0
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-COUR2-AN-2-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eSPIRAL(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        A = 0.5*(self.elpar[iel_][1]+self.elpar[iel_][0])
        B = 0.5*(self.elpar[iel_][1]-self.elpar[iel_][0])
        X2 = EV_[0]*EV_[0]
        Y2 = EV_[1]*EV_[1]
        R2 = X2+Y2
        D = 1.0+R2
        D2 = D*D
        D3 = D2*D
        U = R2/D
        DUDX = (EV_[0]+EV_[0])/D2
        DUDY = (EV_[1]+EV_[1])/D2
        D2UDX2 = 2.0*(D-4.0*X2)/D3
        D2UDY2 = 2.0*(D-4.0*Y2)/D3
        D2UDXY = -8.0*EV_[0]*EV_[1]/D3
        THETA = jnp.arctan2(EV_[1],EV_[0])
        DTDX = -EV_[1]/R2
        DTDY = EV_[0]/R2
        R4 = R2*R2
        D2TDX2 = 2.0*EV_[0]*EV_[1]/R4
        D2TDY2 = -2.0*EV_[1]*EV_[0]/R4
        D2TDXY = (Y2-X2)/R4
        R = jnp.sqrt(R2)
        R3 = R*R2
        DRDX = EV_[0]/R
        DRDY = EV_[1]/R
        D2RDX2 = Y2/R3
        D2RDY2 = X2/R3
        D2RDXY = -EV_[0]*EV_[1]/R3
        ARG = R-THETA
        S = B*jnp.sin(ARG)
        C = B*jnp.cos(ARG)
        DCDX = -S*(DRDX-DTDX)
        DCDY = -S*(DRDY-DTDY)
        D2CDX2 = -C*(DRDX-DTDX)**2-S*(D2RDX2-D2TDX2)
        D2CDY2 = -C*(DRDY-DTDY)**2-S*(D2RDY2-D2TDY2)
        D2CDXY = -C*(DRDX-DTDX)*(DRDY-DTDY)-S*(D2RDXY-D2TDXY)
        V = 1.0+A*R-R*C
        DVDX = A*DRDX-DRDX*C-R*DCDX
        DVDY = A*DRDY-DRDY*C-R*DCDY
        D2VDX2 = A*D2RDX2-D2RDX2*C-2.0*DRDX*DCDX-R*D2CDX2
        D2VDY2 = A*D2RDY2-D2RDY2*C-2.0*DRDY*DCDY-R*D2CDY2
        D2VDXY = A*D2RDXY-D2RDXY*C-DRDX*DCDY-DRDY*DCDX-R*D2CDXY
        f_   = U*V
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, DUDX*V+U*DVDX)
            g_ = jtu.np_like_set(g_, 1, DUDY*V+U*DVDY)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), D2UDX2*V+2.0*DUDX*DVDX+U*D2VDX2)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), D2UDXY*V+DUDX*DVDY+DUDY*DVDX+U*D2VDXY)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), D2UDY2*V+2.0*DUDY*DVDY+U*D2VDY2)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class YFITU:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    A nonlinear least-squares problem.  This problem arises in measuring
#    angles and distances to a vibrating beam using a laser-Doppler
#    velocimeter.
#    This is an unconstrained variant of the bounded constrained problem YFIT.
# 
#    Source:
#    an exercize for L. Watson course on LANCELOT in the Spring 1993.
# 
#    SIF input: B. E. Lindholm, Virginia Tech., Spring 1993,
#               modified by Ph. Toint, March 1994.
#               derivatives corrected by Nick Gould, June 2019.
# 
#    classification = "C-CSUR2-MN-3-0"
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'YFITU'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['zero'] = 0
        v_['p'] = 16
        v_['realp'] = 16.0
        v_['y0'] = 21.158931
        v_['y1'] = 17.591719
        v_['y2'] = 14.046854
        v_['y3'] = 10.519732
        v_['y4'] = 7.0058392
        v_['y5'] = 3.5007293
        v_['y6'] = 0.0000000
        v_['y7'] = -3.5007293
        v_['y8'] = -7.0058392
        v_['y9'] = -10.519732
        v_['y10'] = -14.046854
        v_['y11'] = -17.591719
        v_['y12'] = -21.158931
        v_['y13'] = -24.753206
        v_['y14'] = -28.379405
        v_['y15'] = -32.042552
        v_['y16'] = -35.747869
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('alpha',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'alpha')
        [iv,ix_,_] = jtu.s2mpj_ii('beta',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'beta')
        [iv,ix_,_] = jtu.s2mpj_ii('dist',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'dist')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for i in range(int(v_['zero']),int(v_['p'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('diff'+str(i),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for i in range(int(v_['zero']),int(v_['p'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['diff'+str(i)],float(v_['y'+str(i)]))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['alpha'], float(0.60))
        self.x0 = jtu.np_like_set(self.x0, ix_['beta'], float(-0.60))
        self.x0 = jtu.np_like_set(self.x0, ix_['dist'], float(20.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'etanab', iet_)
        elftv = jtu.loaset(elftv,it,0,'a1')
        elftv = jtu.loaset(elftv,it,1,'b1')
        elftv = jtu.loaset(elftv,it,2,'d1')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'point')
        elftp = jtu.loaset(elftp,it,1,'count')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for i in range(int(v_['zero']),int(v_['p'])+1):
            v_['index'] = float(i)
            ename = 'est'+str(i)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'etanab')
            ielftype = jtu.arrset(ielftype,ie,iet_["etanab"])
            vname = 'alpha'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='a1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'beta'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='b1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'dist'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='d1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='point')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['index']))
            posep = jnp.where(elftp[ielftype[ie]]=='count')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['realp']))
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gsquare',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for i in range(int(v_['zero']),int(v_['p'])+1):
            ig = ig_['diff'+str(i)]
            self.grftype = jtu.arrset(self.grftype,ig,'gsquare')
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['est'+str(i)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
# LO SOLUTION            0.0
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSUR2-MN-3-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def etanab(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        frac = self.elpar[iel_][0]/self.elpar[iel_][1]
        ttan = jnp.tan(EV_[0]*(1.0-frac)+EV_[1]*frac)
        tsec = 1.0/jnp.cos(EV_[0]*(1.0-frac)+EV_[1]*frac)
        tsec2 = tsec*tsec
        f_   = EV_[2]*ttan
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[2]*(1.0-frac)*tsec2)
            g_ = jtu.np_like_set(g_, 1, EV_[2]*frac*tsec2)
            g_ = jtu.np_like_set(g_, 2, ttan)
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0*EV_[2]*((1.0-frac)**2)*tsec2*ttan)
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), 2.0*EV_[2]*(frac**2)*tsec2*ttan)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 2.0*EV_[2]*(1.0-frac)*frac*tsec2*ttan)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), (1.0-frac)*tsec2)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), frac*tsec2)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), 0.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gsquare(self,nargout,*args):

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


from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class SENSORS:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : SENSORS
#    *********
# 
#    A problem arising from two-dimensional optimal sensor placement
# 
#    Source:
#    H. Zhang and X. Wang,
#    "Optimal sensor placement",
#    SIAM Review, vol. 35, p. 641, 1993.
# 
#    SIF input: Nick Gould, June 1994
# 
#    classification = "C-COUR2-AN-V-0"
# 
#    Number of unknowns
# 
#           Alternative values for the SIF file parameters:
# IE N                   2              $-PARAMETER
# IE N                   3              $-PARAMETER
# IE N                   10             $-PARAMETER
# IE N                   100            $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'SENSORS'

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
# IE N                   1000           $-PARAMETER
        v_['1'] = 1
        v_['RN'] = float(v_['N'])
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('THETA'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'THETA'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for J in range(int(v_['1']),int(v_['N'])+1):
            for I in range(int(v_['1']),int(v_['N'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('S'+str(I)+','+str(J),ig_)
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['I/N'] = v_['RI']/v_['RN']
            self.x0 = jtu.np_like_set(self.x0, ix_['THETA'+str(I)], float(v_['I/N']))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSINFUN', iet_)
        elftv = jtu.loaset(elftv,it,0,'THETAI')
        elftv = jtu.loaset(elftv,it,1,'THETAJ')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for J in range(int(v_['1']),int(v_['N'])+1):
            for I in range(int(v_['1']),int(v_['N'])+1):
                ename = 'S'+str(I)+','+str(J)
                [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
                if newelt:
                    self.elftype = jtu.arrset(self.elftype,ie,'eSINFUN')
                    ielftype = jtu.arrset(ielftype,ie,iet_['eSINFUN'])
                vname = 'THETA'+str(I)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='THETAI')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'THETA'+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='THETAJ')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gmL2',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for ig in range(0,ngrp):
            self.grftype = jtu.arrset(self.grftype,ig,'gmL2')
        for J in range(int(v_['1']),int(v_['N'])+1):
            for I in range(int(v_['1']),int(v_['N'])+1):
                ig = ig_['S'+str(I)+','+str(J)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(I)+','+str(J)])
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-COUR2-AN-V-0"
        self.objderlvl = 2


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eSINFUN(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        TIMJ = EV_[0]-EV_[1]
        SI = jnp.sin(EV_[0])
        SJ = jnp.sin(EV_[1])
        SIMJ = jnp.sin(TIMJ)
        CI = jnp.cos(EV_[0])
        CJ = jnp.cos(EV_[1])
        CIMJ = jnp.cos(TIMJ)
        CJSIMJ = CJ*SIMJ-SJ*CIMJ
        CJCIMJ = CJ*CIMJ+SJ*SIMJ
        f_   = SI*SJ*SIMJ
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, SJ*(CI*SIMJ+SI*CIMJ))
            g_ = jtu.np_like_set(g_, 1, SI*CJSIMJ)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0*SJ*(CI*CIMJ-SI*SIMJ))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), CI*CJSIMJ+SI*CJCIMJ)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), -2.0*SI*CJCIMJ)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gmL2(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= -GVAR_*GVAR_
        if nargout>1:
            g_ = -GVAR_-GVAR_
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = -2.0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


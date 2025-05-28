import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class SPARSINE:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : SPARSINE
#    *********
# 
#    A sparse problem involving sine functions
# 
#    SIF input: Nick Gould, November 1995
# 
#    classification = "C-COUR2-AN-V-0"
# 
#    The number of variables 
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER
# IE N                   50             $-PARAMETER
# IE N                   100            $-PARAMETER
# IE N                   1000           $-PARAMETER     original value
# IE N                   5000           $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'SPARSINE'

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
# IE N                   10000          $-PARAMETER
        v_['1'] = 1
        v_['2'] = 2
        v_['4'] = 4
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
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(0.5))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSINE', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            ename = 'S'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSINE')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSINE"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.5))
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gSQR',igt_)
        [it,igt_,_] = jtu.s2mpj_ii('gSQR',igt_)
        grftp = []
        grftp = jtu.loaset(grftp,it,0,'P')
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        self.grpar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['OBJ'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gSQR')
            v_['RI'] = float(I)
            posgp = jnp.where(grftp[igt_[self.grftype[ig]]]=='P')[0]
            self.grpar =jtu.loaset(self.grpar,ig,posgp[0],float(v_['RI']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
            v_['J'] = 2*I
            v_['J'] = -1+v_['J']
            v_['K'] = int(jnp.fix(v_['J']/v_['N']))
            v_['K'] = v_['K']*v_['N']
            v_['J'] = v_['J']-v_['K']
            v_['J'] = 1+v_['J']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['J']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
            v_['J'] = 3*I
            v_['J'] = -1+v_['J']
            v_['K'] = int(jnp.fix(v_['J']/v_['N']))
            v_['K'] = v_['K']*v_['N']
            v_['J'] = v_['J']-v_['K']
            v_['J'] = 1+v_['J']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['J']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
            v_['J'] = 5*I
            v_['J'] = -1+v_['J']
            v_['K'] = int(jnp.fix(v_['J']/v_['N']))
            v_['K'] = v_['K']*v_['N']
            v_['J'] = v_['J']-v_['K']
            v_['J'] = 1+v_['J']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['J']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
            v_['J'] = 7*I
            v_['J'] = -1+v_['J']
            v_['K'] = int(jnp.fix(v_['J']/v_['N']))
            v_['K'] = v_['K']*v_['N']
            v_['J'] = v_['J']-v_['K']
            v_['J'] = 1+v_['J']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['J']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
            v_['J'] = 11*I
            v_['J'] = -1+v_['J']
            v_['K'] = int(jnp.fix(v_['J']/v_['N']))
            v_['K'] = v_['K']*v_['N']
            v_['J'] = v_['J']-v_['K']
            v_['J'] = 1+v_['J']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['S'+str(int(v_['J']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLUTION            0.0
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-COUR2-AN-V-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eSINE(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = jnp.sin(EV_[0])
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, jnp.cos(EV_[0]))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -jnp.sin(EV_[0]))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gSQR(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= 0.5*self.grpar[igr_][0]*GVAR_*GVAR_
        if nargout>1:
            g_ = self.grpar[igr_][0]*GVAR_
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = self.grpar[igr_][0]
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class GROWTHLS:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : GROWTHLS
#    *********
#    GROWTH problem in 3 variables
# 
#    Fit the observed growth g(n) from Gaussian Elimination
#    with complete pivoting to a function of the form
#         U1 * n ** ( U2 + LOG(n) * U3 )
# 
#    SIF input: Nick Gould, Nov, 1991, modified by Ph. Toint, March 1994.
# 
#    classification = "C-CSUR2-AN-3-0"
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'GROWTHLS'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 3
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('U1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'U1')
        [iv,ix_,_] = jtu.s2mpj_ii('U2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'U2')
        [iv,ix_,_] = jtu.s2mpj_ii('U3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'U3')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('G8',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G9',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G10',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G11',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G12',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G13',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G14',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G15',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G16',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G18',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G20',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('G25',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        self.objgrps = jnp.arange(ngrp)
        self.m       = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        self.gconst = jtu.arrset(self.gconst,ig_['G8'],float(8.0))
        self.gconst = jtu.arrset(self.gconst,ig_['G9'],float(8.4305))
        self.gconst = jtu.arrset(self.gconst,ig_['G10'],float(9.5294))
        self.gconst = jtu.arrset(self.gconst,ig_['G11'],float(10.4627))
        self.gconst = jtu.arrset(self.gconst,ig_['G12'],float(12.0))
        self.gconst = jtu.arrset(self.gconst,ig_['G13'],float(13.0205))
        self.gconst = jtu.arrset(self.gconst,ig_['G14'],float(14.5949))
        self.gconst = jtu.arrset(self.gconst,ig_['G15'],float(16.1078))
        self.gconst = jtu.arrset(self.gconst,ig_['G16'],float(18.0596))
        self.gconst = jtu.arrset(self.gconst,ig_['G18'],float(20.4569))
        self.gconst = jtu.arrset(self.gconst,ig_['G20'],float(24.25))
        self.gconst = jtu.arrset(self.gconst,ig_['G25'],float(32.9863))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['U1'], float(100.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eFIT', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'RN')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'G8'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(8.0))
        ename = 'G9'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(9.0))
        ename = 'G10'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(10.0))
        ename = 'G11'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(11.0))
        ename = 'G12'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(12.0))
        ename = 'G13'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(13.0))
        ename = 'G14'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(14.0))
        ename = 'G15'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(15.0))
        ename = 'G16'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(16.0))
        ename = 'G18'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(18.0))
        ename = 'G20'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(20.0))
        ename = 'G25'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_['eFIT'])
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='RN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(25.0))
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
        ig = ig_['G8']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G8'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G9']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G9'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G10']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G10'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G11']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G11'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G12']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G12'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G13']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G13'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G14']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G14'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G15']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G15'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G16']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G16'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G18']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G18'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G20']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G20'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['G25']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['G25'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN                0.0
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.pbclass   = "C-CSUR2-AN-3-0"
        self.objderlvl = 2

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eFIT(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        LOGRN = jnp.log(self.elpar[iel_][0])
        POWER = self.elpar[iel_][0]**(EV_[1]+LOGRN*EV_[2])
        f_   = EV_[0]*POWER
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, POWER)
            g_ = jtu.np_like_set(g_, 1, EV_[0]*POWER*LOGRN)
            g_ = jtu.np_like_set(g_, 2, EV_[0]*POWER*LOGRN**2)
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), POWER*LOGRN)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), POWER*LOGRN**2)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), EV_[0]*POWER*LOGRN**2)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), EV_[0]*POWER*LOGRN**3)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), EV_[0]*POWER*LOGRN**4)
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


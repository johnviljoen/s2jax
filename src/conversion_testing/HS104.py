from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HS104:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HS104
#    *********
# 
#    Source: problem 104 in
#    W. Hock and K. Schittkowski,
#    "Test examples for nonlinear programming codes",
#    Lectures Notes in Economics and Mathematical Systems 187, Springer
#    Verlag, Heidelberg, 1981.
# 
#    SIF input: Nick Gould, August 1991.
# 
#    classification = "C-COOR2-AN-8-5"
# 
#    Number of variables
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HS104'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 8
        v_['1'] = 1
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
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(-1.0e+0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(-1.0e+0))
        [ig,ig_,_] = jtu.s2mpj_ii('C1',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'C1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(1.0e-1))
        [ig,ig_,_] = jtu.s2mpj_ii('C2',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'C2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(1.0e-1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(1.0e-1))
        [ig,ig_,_] = jtu.s2mpj_ii('C3',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'C3')
        [ig,ig_,_] = jtu.s2mpj_ii('C4',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'C4')
        [ig,ig_,_] = jtu.s2mpj_ii('C5',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C5')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(-1.0e+0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(-1.0e+0))
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        self.n   = len(ix_)
        ngrp   = len(ig_)
        legrps = jnp.where(gtype=='<=')[0]
        eqgrps = jnp.where(gtype=='==')[0]
        gegrps = jnp.where(gtype=='>=')[0]
        self.nle = len(legrps)
        self.neq = len(eqgrps)
        self.nge = len(gegrps)
        self.m   = self.nle+self.neq+self.nge
        self.congrps = jnp.concatenate((legrps,eqgrps,gegrps))
        self.cnames = cnames[self.congrps]
        self.nob = ngrp-self.m
        self.objgrps = jnp.where(gtype=='<>')[0]
        #%%%%%%%%%%%%%%%%%%  CONSTANTS %%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.full((ngrp,1),1.0e+0)
        self.gconst = jtu.arrset(self.gconst,ig_['OBJ'],float(-1.0e+1))
        self.gconst = jtu.arrset(self.gconst,ig_['C5'],float(-9.0e+0))
        #%%%%%%%%%%%%%%%%%%%%  RANGES %%%%%%%%%%%%%%%%%%%%%%
        grange = jnp.full((ngrp,1),None)
        grange = jtu.np_like_set(grange, legrps, jnp.full((self.nle,1),float('inf')))
        grange = jtu.np_like_set(grange, gegrps, jnp.full((self.nge,1),float('inf')))
        grange = jtu.arrset(grange,ig_['C5'],float(3.2e+0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),1.0e-1)
        self.xupper = jnp.full((self.n,1),1.0e+1)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        if('X1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(6.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X1']),float(6.0)))
        if('X2' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X2'], float(3.0))
        else:
            self.y0 = jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X2'])[0],float(3.0))
        if('X3' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X3'], float(0.4))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X3']),float(0.4)))
        if('X4' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X4'], float(0.2))
        else:
            self.y0 = jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X4'])[0],float(0.2))
        if('X5' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X5'], float(6.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X5']),float(6.0)))
        if('X6' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X6'], float(6.0))
        else:
            self.y0 = jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X6'])[0],float(6.0))
        if('X7' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X7'], float(1.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X7']),float(1.0)))
        if('X8' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X8'], float(0.5))
        else:
            self.y0 = jtu.arrset(self.y0,jnp.where(self.congrps==ig_['X8'])[0],float(0.5))
        pass
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        elftv = jtu.loaset(elftv,it,1,'V2')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'P1')
        elftp = jtu.loaset(elftp,it,1,'P2')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'OE1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.67))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-0.67))
        ename = 'OE2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(0.67))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-0.67))
        ename = 'C1E1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.0))
        ename = 'C2E1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.0))
        ename = 'C3E1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-1.0))
        ename = 'C3E2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-0.71))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-1.0))
        ename = 'C3E3'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-1.3))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.0))
        ename = 'C4E1'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.0))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-1.0))
        ename = 'C4E2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-0.71))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-1.0))
        ename = 'C4E3'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(1.0e-1),float(1.0e+1),None)
        posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='P1')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-1.3))
        posep = jnp.where(elftp[ielftype[ie]]=='P2')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.0))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['OBJ']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['OE1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(4.0e-1))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['OE2'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(4.0e-1))
        ig = ig_['C1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C1E1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(5.88e-2))
        ig = ig_['C2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C2E1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(5.88e-2))
        ig = ig_['C3']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C3E1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(4.0e+0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C3E2'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(2.0e+0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C3E3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(5.88e-2))
        ig = ig_['C4']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C4E1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(4.0e+0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C4E2'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(2.0e+0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['C4E3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(5.88e-2))
        ig = ig_['C5']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['OE1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(4.0e-1))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['OE2'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(4.0e-1))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               3.9511634396
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.arange(self.nle), grange[legrps])
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nge), grange[gegrps])
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COOR2-AN-8-5"
        self.objderlvl = 2
        self.conderlvl = [2]

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
        f_   = (EV_[0]**self.elpar[iel_][0])*(EV_[1]**self.elpar[iel_][1])
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, ()
                  self.elpar[iel_][0]*(EV_[0]**(self.elpar[iel_][0]-1.0))*(EV_[1]**self.elpar[iel_][1]))
            g_ = jtu.np_like_set(g_, 1, ()
                  self.elpar[iel_][1]*(EV_[0]**self.elpar[iel_][0])*(EV_[1]**(self.elpar[iel_][1]-1.0)))
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), ()
                      self.elpar[iel_][0]*(EV_[0]**(self.elpar[iel_][0]-2.0))*(self.elpar[iel_][0]-1.0)*(EV_[1]**self.elpar[iel_][1]))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), ()
                      self.elpar[iel_][0]*(EV_[0]**(self.elpar[iel_][0]-1.0))*self.elpar[iel_][1]*(EV_[1]**(self.elpar[iel_][1]-1.0)))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), ()
                      self.elpar[iel_][1]*(self.elpar[iel_][1]-1.0)*(EV_[0]**self.elpar[iel_][0])*(EV_[1]**(self.elpar[iel_][1]-2.0)))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


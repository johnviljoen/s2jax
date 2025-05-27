from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HS67:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HS67
#    *********
# 
#    Source: problem 67 in
#    W. Hock and K. Schittkowski,
#    "Test examples for nonlinear programming codes",
#    Lectures Notes in Economics and Mathematical Systems 187, Springer
#    Verlag, Heidelberg, 1981.
# 
#    Original Source: problem 8 in
#    A.R. Colville
#    "A comparative study on nonlinear programming"
#    IBM Scientific Center Report 320-2949, New York, 1968.
# 
#    SIF input: A.R. Conn & Nick Gould, April 1991.
# 
#    classification = "C-COOI2-AN-3-14"
# 
#    Set useful parameters
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HS67'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['4'] = 4
        v_['5'] = 5
        v_['6'] = 6
        v_['7'] = 7
        v_['8'] = 8
        v_['9'] = 9
        v_['10'] = 10
        v_['11'] = 11
        v_['12'] = 12
        v_['13'] = 13
        v_['14'] = 14
        v_['A'+str(int(v_['1']))] = 0.0
        v_['A'+str(int(v_['2']))] = 0.0
        v_['A'+str(int(v_['3']))] = 85.0
        v_['A'+str(int(v_['4']))] = 90.0
        v_['A'+str(int(v_['5']))] = 3.0
        v_['A'+str(int(v_['6']))] = 0.01
        v_['A'+str(int(v_['7']))] = 145.0
        v_['A'+str(int(v_['8']))] = 5000.0
        v_['A'+str(int(v_['9']))] = 2000.0
        v_['A'+str(int(v_['10']))] = 93.0
        v_['A'+str(int(v_['11']))] = 95.0
        v_['A'+str(int(v_['12']))] = 12.0
        v_['A'+str(int(v_['13']))] = 4.0
        v_['A'+str(int(v_['14']))] = 162.0
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['3'])+1):
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
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['1']))]])
        valA = jtu.append(valA,float(5.04))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['2']))]])
        valA = jtu.append(valA,float(0.035))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['3']))]])
        valA = jtu.append(valA,float(10.0))
        for I in range(int(v_['1']),int(v_['7'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('AG'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'AG'+str(I))
            [ig,ig_,_] = jtu.s2mpj_ii('AL'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<=')
            cnames = jtu.arrset(cnames,ig,'AL'+str(I))
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
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for I in range(int(v_['1']),int(v_['7'])+1):
            v_['I+7'] = 7+I
            self.gconst = jtu.arrset(self.gconst,ig_['AG'+str(I)],float(v_['A'+str(I)]))
            self.gconst  = (
                  jtu.arrset(self.gconst,ig_['AL'+str(I)],float(v_['A'+str(int(v_['I+7']))])))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),0.00001)
        self.xupper = jnp.full((self.n,1),+float('inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['X'+str(int(v_['1']))], 2000.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X'+str(int(v_['2']))], 16000.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X'+str(int(v_['3']))], 120.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(int(v_['1']))], float(1745.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(int(v_['2']))], float(12000.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(int(v_['3']))], float(110.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eY2Y5', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        [it,iet_,_] = jtu.s2mpj_ii( 'eY2', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        [it,iet_,_] = jtu.s2mpj_ii( 'eY3', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        [it,iet_,_] = jtu.s2mpj_ii( 'eY4', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        [it,iet_,_] = jtu.s2mpj_ii( 'eY5', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        [it,iet_,_] = jtu.s2mpj_ii( 'eY6', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        [it,iet_,_] = jtu.s2mpj_ii( 'eY7', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        [it,iet_,_] = jtu.s2mpj_ii( 'eY8', iet_)
        elftv = jtu.loaset(elftv,it,0,'U1')
        elftv = jtu.loaset(elftv,it,1,'U2')
        elftv = jtu.loaset(elftv,it,2,'U3')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        ename = 'E25'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eY2Y5')
        ielftype = jtu.arrset(ielftype,ie,iet_["eY2Y5"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eY2')
        ielftype = jtu.arrset(ielftype,ie,iet_["eY2"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E3'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eY3')
        ielftype = jtu.arrset(ielftype,ie,iet_["eY3"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E4'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eY4')
        ielftype = jtu.arrset(ielftype,ie,iet_["eY4"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E5'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eY5')
        ielftype = jtu.arrset(ielftype,ie,iet_["eY5"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E6'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eY6')
        ielftype = jtu.arrset(ielftype,ie,iet_["eY6"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E7'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eY7')
        ielftype = jtu.arrset(ielftype,ie,iet_["eY7"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'E8'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eY8')
        ielftype = jtu.arrset(ielftype,ie,iet_["eY8"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.00001),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='U3')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['OBJ']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E25'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-0.063))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E3'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(3.36))
        ig = ig_['AG'+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AL'+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AG'+str(int(v_['2']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AL'+str(int(v_['2']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AG'+str(int(v_['3']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E4'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AL'+str(int(v_['3']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E4'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AG'+str(int(v_['4']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E5'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AL'+str(int(v_['4']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E5'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AG'+str(int(v_['5']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E6'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AL'+str(int(v_['5']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E6'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AG'+str(int(v_['6']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E7'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AL'+str(int(v_['6']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E7'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AG'+str(int(v_['7']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E8'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AL'+str(int(v_['7']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E8'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COOI2-AN-3-14"
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

#    @staticmethod
#    def e_globs(self):

#        import numpy as jnp
#        self.efpar = jnp.array([]);
#        self.efpar = jtu.arrset( self.efpar,0,.FALSE.0)
#        return pbm

    @staticmethod
    def extfunc(self,X1,X2,X3):
    # PAGE 129 Hock and Schittkowski.

        import jax.numpy as jnp
        Y = jnp.zeros((8,1))
        G = jnp.zeros((8,3))
        H = jnp.zeros((8,6))

     #  First approximation to Y1.

        Y = jtu.np_like_set(Y, 1, 1.6 * X1)

     #  Loop until Y1 converges.

        for I in range(1,1000):

            #  Y2.

            Y = jtu.np_like_set(Y, 2, 1.22 * Y[1] - X1)
            G = jtu.np_like_set(G, jnp.array([2,0]), 1.22 * G[1,0] - 1.0)
            G = jtu.np_like_set(G, jnp.array([2,1]), 1.22 * G[1,1])
            G = jtu.np_like_set(G, jnp.array([2,2]), 1.22 * G[1,2])
            H = jtu.np_like_set(H, jnp.array([2,0]), 1.22 * H[1,0])
            H = jtu.np_like_set(H, jnp.array([2,1]), 1.22 * H[1,1])
            H = jtu.np_like_set(H, jnp.array([2,2]), 1.22 * H[1,2])
            H = jtu.np_like_set(H, jnp.array([2,3]), 1.22 * H[1,3])
            H = jtu.np_like_set(H, jnp.array([2,4]), 1.22 * H[1,4])
            H = jtu.np_like_set(H, jnp.array([2,5]), 1.22 * H[1,5])

            #  Y5.

            Y = jtu.np_like_set(Y, 5, ( X2 + Y[2] ) / X1)
            G = jtu.np_like_set(G, jnp.array([5,0]), - Y[5] / X1 + G[2,0] / X1)
            G = jtu.np_like_set(G, jnp.array([5,1]), 1.0  / X1 + G[2,1] / X1)
            G = jtu.np_like_set(G, jnp.array([5,2]), G[2,2] / X1)
            H = jtu.np_like_set(H, jnp.array([5,0]), - G[5,0] / X1 + Y[5]/ X1**2 - G[2,0] / X1**2 + H[2,0] / X1)
            H = jtu.np_like_set(H, jnp.array([5,1]), - G[5,1] / X1 + H[2,1] / X1)
            H = jtu.np_like_set(H, jnp.array([5,2]), - G[5,2] / X1 + H[2,2] / X1)
            H = jtu.np_like_set(H, jnp.array([5,3]), H[2,3] / X1)
            H = jtu.np_like_set(H, jnp.array([5,4]), H[2,4] / X1)
            H = jtu.np_like_set(H, jnp.array([5,5]), H[2,5] / X1)
            Y1C    = 0.01 * X1 * ( 112.0 + 13.167 * Y[5] - 0.6667 * Y[5]**2 );
   
            #  Y1.

            if jnp.abs( Y1C - Y[1] ) > 0.001 :
   
                Y = jtu.np_like_set(Y, 1, Y1C;)
                G = jtu.np_like_set(G, jnp.array([1,0]), ( 0.01 * ( 112.0 + 13.167 * Y[5] - 0.6667 * Y[5]**2 ))
                         +  X1 * 0.13167 * G[5,0] -  X1 * 0.013334 * Y[5] * G[5,0] )
                G = jtu.np_like_set(G, jnp.array([1,1]), X1 * ( 0.13167 * G[5,1] - 0.013334 * Y[5] * G[5,1] ))
                G = jtu.np_like_set(G, jnp.array([1,2]), X1 * ( 0.13167 * G[5,2] - 0.013334 * Y[5] * G[5,2] ))
                H = jtu.np_like_set(H, jnp.array([1,0]), (  0.13167 * G[5,0] - 0.013334 * Y[5] * G[5,0] + 0.13167 * G[5,0])
                           - 0.013334 * Y[5] * G[5,0] + X1 * 0.13167 * H[5,0] - X1 * 0.013334 * G[5,0]**2 
                           - X1 * 0.013334 * Y[5] * H[5,0] )
                H = jtu.np_like_set(H, jnp.array([1,1]), (  0.13167 * G[5,1] - 0.013334 * Y[5] * G[5,1]         + X1 * 0.13167 * H[5,1])
                           - X1 * 0.013334 * G[5,1] * G[5,0]                     - X1 * 0.013334 * Y[5] * H[5,1] )
                H = jtu.np_like_set(H, jnp.array([1,2]), ( 0.13167 * G[5,2] - 0.013334 * Y[5] *  G[5,2]          + X1 * 0.13167 * H[5,2])
                           - X1 * 0.013334 * G[5,2] * G[5,0]                     - X1 * 0.013334 * Y[5] * H[5,2] )
                H = jtu.np_like_set(H, jnp.array([1,3]), X1 * ( 0.13167 * H[5,3] - 0.013334 * G[5,1]**2         - 0.013334 * Y[5] * H[5,3]))
                H = jtu.np_like_set(H, jnp.array([1,4]), X1 * ( 0.13167 * H[5,4] - 0.013334 * G[5,2] * G[5,1]  - 0.013334 * Y[5] * H[5,4]))
                H = jtu.np_like_set(H, jnp.array([1,5]), X1 * ( 0.13167 * H[5,5] - 0.013334 * G[5,2]**2         - 0.013334 * Y[5] * H[5,5]))
            else:
                break

#        print( "Y = ", Y )
#        print( "G = ", G )
#        print( "H = ", H )
#        exit()%D

        #  First approximation to Y3.

        Y = jtu.np_like_set(Y, 3, 93.0)

        #  Loop until Y3 converges.

        for I in range(1,1000):

            #  Y4.

            Y = jtu.np_like_set(Y, 4, 86.35 + 1.098 * Y[5] - 0.038 * Y[5]**2 + 0.325 * ( Y[3] - 89.0 ))
            G = jtu.np_like_set(G, jnp.array([4,0]), 1.098 * G[5,0] - 0.076 * Y[5] * G[5,0] + 0.325 * G[3,0])
            G = jtu.np_like_set(G, jnp.array([4,1]), 1.098 * G[5,1] - 0.076 * Y[5] * G[5,1] + 0.325 * G[3,1])
            G = jtu.np_like_set(G, jnp.array([4,2]), 1.098 * G[5,2] - 0.076 * Y[5] * G[5,2] + 0.325 * G[3,2])
            H = jtu.np_like_set(H, jnp.array([4,0]), 1.098 * H[5,0] - 0.076 * G[5,0] * G[5,0] - 0.076 * Y[5] * H[5,0] + 0.325 * H[3,0])
            H = jtu.np_like_set(H, jnp.array([4,1]), 1.098 * H[5,1] - 0.076 * G[5,0] * G[5,1] - 0.076 * Y[5] * H[5,1] + 0.325 * H[3,1])
            H = jtu.np_like_set(H, jnp.array([4,2]), 1.098 * H[5,2] - 0.076 * G[5,0] * G[5,2] - 0.076 * Y[5] * H[5,2] + 0.325 * H[3,2])
            H = jtu.np_like_set(H, jnp.array([4,3]), 1.098 * H[5,3] - 0.076 * G[5,1] * G[5,1] - 0.076 * Y[5] * H[5,3] + 0.325 * H[3,3])
            H = jtu.np_like_set(H, jnp.array([4,4]), 1.098 * H[5,4] - 0.076 * G[5,1] * G[5,2] - 0.076 * Y[5] * H[5,4] + 0.325 * H[3,4])
            H = jtu.np_like_set(H, jnp.array([4,5]), 1.098 * H[5,5] - 0.076 * G[5,2] * G[5,2] - 0.076 * Y[5] * H[5,5] + 0.325 * H[3,5])

            #  Y7.

            Y = jtu.np_like_set(Y, 7, 3.0 * Y[4] - 133.0)
            G = jtu.np_like_set(G, jnp.array([7,0]), 3.0 * G[4,0])
            G = jtu.np_like_set(G, jnp.array([7,1]), 3.0 * G[4,1])
            G = jtu.np_like_set(G, jnp.array([7,2]), 3.0 * G[4,2])
            H = jtu.np_like_set(H, jnp.array([7,0]), 3.0 * H[4,0])
            H = jtu.np_like_set(H, jnp.array([7,1]), 3.0 * H[4,1])
            H = jtu.np_like_set(H, jnp.array([7,2]), 3.0 * H[4,2])
            H = jtu.np_like_set(H, jnp.array([7,3]), 3.0 * H[4,3])
            H = jtu.np_like_set(H, jnp.array([7,4]), 3.0 * H[4,4])
            H = jtu.np_like_set(H, jnp.array([7,5]), 3.0 * H[4,5])

            #  Y6.

            Y = jtu.np_like_set(Y, 6, 35.82 - 0.222 * Y[7])
            G = jtu.np_like_set(G, jnp.array([6,0]), - 0.222 * G[7,0])
            G = jtu.np_like_set(G, jnp.array([6,1]), - 0.222 * G[7,1])
            G = jtu.np_like_set(G, jnp.array([6,2]), - 0.222 * G[7,2])
            H = jtu.np_like_set(H, jnp.array([6,0]), - 0.222 * H[7,0])
            H = jtu.np_like_set(H, jnp.array([6,1]), - 0.222 * H[7,1])
            H = jtu.np_like_set(H, jnp.array([6,2]), - 0.222 * H[7,2])
            H = jtu.np_like_set(H, jnp.array([6,3]), - 0.222 * H[7,3])
            H = jtu.np_like_set(H, jnp.array([6,4]), - 0.222 * H[7,4])
            H = jtu.np_like_set(H, jnp.array([6,5]), - 0.222 * H[7,5])
            Y1Y6X3 = Y[1] * Y[6] + 1000.0 * X3
            Y3C    = 98000.0 * X3 / Y1Y6X3

#            print( "Y = ", Y )
#            print( "G = ", G )
#            print( "H = ", H )
#            print( "===1===")
#            exit()%D        
                       
            #  Y3.

            if jnp.abs( Y3C - Y[3] ) > 0.001 :
                Y = jtu.np_like_set(Y, 3, Y3C)
                G = jtu.np_like_set(G, jnp.array([3,0]), -  98000.0 * X3 * ( G[1,0] * Y[6] + Y[1] * G[6,0] ) / Y1Y6X3**2)
                G = jtu.np_like_set(G, jnp.array([3,1]), -  98000.0 * X3 * ( G[1,1] * Y[6] + Y[1] * G[6,1] ) / Y1Y6X3**2;)
                G = jtu.np_like_set(G, jnp.array([3,2]), 98000.0 / Y1Y6X3  - 98000.0 * X3 * ( G[1,2] * Y[6] + Y[1] * G[6,2] + 1000.0 ) / Y1Y6X3**2)
                H = jtu.np_like_set(H, jnp.array([3,0]), ( -  98000.0 * X3 * ( H[1,0] * Y[6] + 2.0 * G[1,0] * G[6,0] + Y[1] * H[6,0] ) / Y1Y6X3**2)
                           + 196000.0 * X3 * ( G[1,0] * Y[6] + Y[1] * G[6,0] )**2 / Y1Y6X3**3 )
                H = jtu.np_like_set(H, jnp.array([3,1]), ( -  98000.0 * X3 * ( H[1,1] * Y[6] + G[1,1] * G[6,0] + G[1,0] * G[6,1]  + Y[1] * H[6,1] ) / Y1Y6X3**2)
                           + 196000.0 * X3 * ( G[1,1] * Y[6] + Y[1] * G[6,1] ) * ( G[1,0] * Y[6]  + Y[1] * G[6,0] ) / Y1Y6X3**3 )
                H = jtu.np_like_set(H, jnp.array([3,2]), ( -  98000.0 * ( Y[1] * G[6,0] + Y[6] *  G[1,0] ) / Y1Y6X3**2)
                           -  98000.0 * X3 * ( G[1,2] * G[6,0] + Y[1] * H[6,2] + G[1,0] * G[6,2]
                           + H[1,2] * Y[6] ) / Y1Y6X3**2 + 196000.0 * X3 * ( Y[1] * G[6,0]
                           + Y[6] * G[1,0] ) * ( G[1,2] * Y[6] + Y[1] * G[6,2] + 1000.0 ) / Y1Y6X3**3 )
                H = jtu.np_like_set(H, jnp.array([3,3]), ( - 98000.0 * X3 * ( H[1,3] * Y[6] + 2.0 * G[1,1] * G[6,1] + Y[1] * H[6,3]) / Y1Y6X3**2)
                           + 196000.0 * X3 * ( G[1,1] * Y[6] + Y[1] * G[6,1] )**2 / Y1Y6X3**3 )
                H = jtu.np_like_set(H, jnp.array([3,4]), ( -  98000.0 * ( Y[1] * G[6,1] + Y[6] *  G[1,1]) / Y1Y6X3**2)
                           -  98000.0 * X3 * ( G[1,2] * G[6,1] + Y[1] * H[6,4]
                          + G[1,1] * G[6,2] + H[1,4] * Y[6] ) / Y1Y6X3**2  + 196000.0 * X3 * ( Y[1] * G[6,1]
                          + Y[6] * G[1,1] ) * ( G[1,2] * Y[6] + Y[1] *  G[6,2] + 1000.0 ) / Y1Y6X3**3 )
                H = jtu.np_like_set(H, jnp.array([3,5]), ( - 196000.0 * ( Y[1] * G[6,2] + Y[6] * G[1,2] + 1000.0 ) / Y1Y6X3**2)
                           -  98000.0 * X3 * ( H[1,5] * Y[6] + 2.0 * G[1,2] * G[6,2] + Y[1] * H[6,5] ) / Y1Y6X3**2 
                           + 196000.0 * X3 * ( G[1,2] * Y[6] + Y[1] * G[6,2] + 1000.0 )**2 / Y1Y6X3**3 )
            else:
                break

#        print( "Y = ", Y )
#        print( "G = ", G )
#        print( "H = ", H )
#        print( "===========")

        return Y, G, H
     
    @staticmethod
    def eY2Y5(self,nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        YY, GG, HH = self.extfunc(self,EV_[0],EV_[1],EV_[2])
        f_   = YY[1]*YY[4]
        if not isinstance( f_, float ):
            f_ = f_.item();
        if nargout>1:
            dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, YY[1]*GG[4,0]+YY[4]*GG[1,0])
            g_ = jtu.np_like_set(g_, 1, YY[1]*GG[4,1]+YY[4]*GG[1,1])
            g_ = jtu.np_like_set(g_, 2, YY[1]*GG[4,2]+YY[4]*GG[1,2])
            if nargout>2:
                H_      = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), YY[1]*HH[4,0]+2.0*GG[4,0]*GG[1,0]+YY[4]*HH[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), YY[1]*HH[4,1]+GG[4,0]*GG[1,1]+GG[4,1]*GG[1,0]+YY[4]*HH[1,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), YY[1]*HH[4,2]+GG[4,0]*GG[1,2]+GG[4,2]*GG[1,0]+YY[4]*HH[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), YY[1]*HH[4,3]+2.0*GG[4,1]*GG[1,1]            +YY[4]*HH[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), YY[1]*HH[4,4]+GG[4,1]*GG[1,2]+GG[4,2]*GG[1,1]+YY[4]*HH[1,4])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), YY[1]*HH[4,5]+2.0*GG[4,2]*GG[1,2]            +YY[4]*HH[1,5])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eY2(self,nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        YY, GG, HH = self.extfunc( self, EV_[0], EV_[1], EV_[2] )
        f_   = YY[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, GG[1,0])
            g_ = jtu.np_like_set(g_, 1, GG[1,1])
            g_ = jtu.np_like_set(g_, 2, 0.0)
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), HH[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), HH[1,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), HH[1,3])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eY3(self,nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        YY, GG, HH = self.extfunc( self, EV_[0], EV_[1], EV_[2] )
        f_   = YY[2]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, GG[2,0])
            g_ = jtu.np_like_set(g_, 1, GG[2,1])
            g_ = jtu.np_like_set(g_, 2, 0.0)
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), HH[2,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), HH[2,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), HH[2,3])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eY4(self,nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        YY, GG, HH = self.extfunc( self, EV_[0], EV_[1], EV_[2] )
        f_   = YY[3]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, GG[3,0])
            g_ = jtu.np_like_set(g_, 1, GG[3,1])
            g_ = jtu.np_like_set(g_, 2, GG[3,2])
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), HH[3,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), HH[3,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), HH[3,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), H_[2,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), HH[3,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), HH[3,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), H_[2,1])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), HH[3,5])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eY5(self,nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        YY, GG, HH = self.extfunc( self, EV_[0], EV_[1], EV_[2] )
        f_   = YY[4]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, GG[4,0])
            g_ = jtu.np_like_set(g_, 1, GG[4,1])
            g_ = jtu.np_like_set(g_, 2, GG[4,2])
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), HH[4,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), HH[4,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), HH[4,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), H_[2,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), HH[4,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), HH[4,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), H_[2,1])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), HH[4,5])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eY6(self,nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        YY, GG, HH = self.extfunc( self, EV_[0], EV_[1], EV_[2] )
        f_   = YY[5]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, GG[5,0])
            g_ = jtu.np_like_set(g_, 1, GG[5,1])
            g_ = jtu.np_like_set(g_, 2, 0.0)
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), HH[5,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), HH[5,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), HH[5,3])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eY7(self,nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        YY, GG, HH = self.extfunc( self, EV_[0], EV_[1], EV_[2] )
        f_   = YY[6]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            dim = len(EV_)
            g_  = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, GG[6,0])
            g_ = jtu.np_like_set(g_, 1, GG[6,1])
            g_ = jtu.np_like_set(g_, 2, GG[6,2])
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), HH[6,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), HH[6,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), HH[6,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), H_[2,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), HH[6,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), HH[6,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), H_[2,1])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), HH[6,5])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eY8(self,nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        YY, GG, HH = self.extfunc( self, EV_[0], EV_[1], EV_[2] )
        f_   = YY[7]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, GG[7,0])
            g_ = jtu.np_like_set(g_, 1, GG[7,1])
            g_ = jtu.np_like_set(g_, 2, GG[7,2])
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), HH[7,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), HH[7,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), HH[7,2])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), H_[2,0])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), HH[7,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), HH[7,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), H_[2,1])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), HH[7,5])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


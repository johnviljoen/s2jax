from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class CONCON:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : CONCON
#    *********
# 
#    A small gas network problem.
# 
#    SIF input: Sybille Schachler, Oxford, August 1992.
#               minor correction by Ph. Shott, Jan 1995.
# 
#    classification = "C-CLOI2-MN-15-11"
# 
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'CONCON'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 7
        v_['M'] = 4
        v_['1'] = 1
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('P'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'P'+str(I))
        for I in range(int(v_['1']),int(v_['M'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('Q'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'Q'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('F'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'F'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('OBJECT',ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['P'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['1']),int(v_['M'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('PAN'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'PAN'+str(I))
        [ig,ig_,_] = jtu.s2mpj_ii('MBAL1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBAL1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Q1']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['F3']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('MBAL2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBAL2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Q1']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['F1']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('MBAL3',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBAL3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Q2']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['F1']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('MBAL4',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBAL4')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Q2']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Q3']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('MBAL5',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBAL5')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Q3']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['F2']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('MBAL6',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBAL6')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Q4']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['F2']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('MBAL7',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'MBAL7')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Q4']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['F4']])
        valA = jtu.append(valA,float(-1.0))
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
        v_['DEMAND'] = -1000.0
        self.gconst = jtu.arrset(self.gconst,ig_['MBAL4'],float(v_['DEMAND']))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        v_['PMAX1'] = 914.73
        v_['PMAX2'] = 904.73
        self.xupper = jtu.np_like_set(self.xupper, ix_['P3'], v_['PMAX2'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['P5'], v_['PMAX2'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['P1'], v_['PMAX1'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['P7'], v_['PMAX1'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['F4'], 400)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.x0 = jtu.np_like_set(self.x0, ix_['P'+str(I)], float(965))
        self.x0 = jtu.np_like_set(self.x0, ix_['Q1'], float(100))
        self.x0 = jtu.np_like_set(self.x0, ix_['Q2'], float(100))
        self.x0 = jtu.np_like_set(self.x0, ix_['Q3'], float(-100))
        self.x0 = jtu.np_like_set(self.x0, ix_['Q4'], float(-100))
        self.x0 = jtu.np_like_set(self.x0, ix_['F1'], float(1000))
        self.x0 = jtu.np_like_set(self.x0, ix_['F2'], float(1000))
        self.x0 = jtu.np_like_set(self.x0, ix_['F3'], float(1000))
        self.x0 = jtu.np_like_set(self.x0, ix_['F4'], float(1000))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQR', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        [it,iet_,_] = jtu.s2mpj_ii( 'eFORQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'Y')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            ename = 'PSQ'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQR')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQR"])
            vname = 'P'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['M'])+1):
            ename = 'QTO'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eFORQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eFORQ"])
            vname = 'Q'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        v_['K'] = -0.597053452
        ig = ig_['PAN1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['PSQ1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['PSQ2'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['QTO1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['K']))
        ig = ig_['PAN2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['PSQ3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['PSQ4'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['QTO2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['K']))
        ig = ig_['PAN3']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['PSQ4'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['PSQ5'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['QTO3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['K']))
        ig = ig_['PAN4']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['PSQ6'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['PSQ7'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['QTO4'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['K']))
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CLOI2-MN-15-11"
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eSQR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*jnp.absolute(EV_[0])
        if not isinstance( f_, float ):
            f_   = f_.item();
        PO = EV_[0]>0.0
        if PO!=0:
            GO = 2*EV_[0]
        if PO==0:
            GO = -2*EV_[0]
        if PO!=0:
            HO = 2
        if PO==0:
            HO = -2
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, GO)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), HO)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eFORQ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*jnp.absolute(EV_[0])**0.8539
        if not isinstance( f_, float ):
            f_   = f_.item();
        POS = EV_[0]>0.0
        if POS!=0:
            GG = 1.8539*EV_[0]**0.8539
        if POS==0:
            GG = 1.8539*jnp.absolute(EV_[0])**0.8539
        if POS!=0:
            HH = 1.8539*0.8539*EV_[0]**(-0.1461)
        if POS==0:
            HH = -1.8539*0.8539*jnp.absolute(EV_[0])**(-0.1461)
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, GG)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), HH)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class BATCH:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : BATCH
#    *********
# 
#    Source: Optimal Design of Multiproduct Batch Plant
#    G.R. Kocis & I.E. Grossmann,
#    "Global OPtimization of Nonconvex Mixed Integer Nonlinear Programmming
#     (MINLP) problems in Process Synthesis", Indust. Engng. Chem. Res.,
#    No. 27, pp 1407--1421, 1988.
# 
#    SIF input: S. Leyffer, October 1997
# 
#    classification = "C-COOR2-AN-46-73"
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'BATCH'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['1'] = 1
        v_['M'] = 6
        v_['N'] = 5
        v_['NU'] = 4
        v_['LOGNU'] = jnp.log(4.0)
        v_['VL'] = jnp.log(300.0)
        v_['VU'] = jnp.log(3000.0)
        v_['H'] = 6000.0
        v_['TLO1'] = 0.729961
        v_['TLO2'] = 0.530628
        v_['TLO3'] = 1.09024
        v_['TLO4'] = -0.133531
        v_['TLO5'] = 0.0487901
        v_['TUP1'] = 2.11626
        v_['TUP2'] = 1.91626
        v_['TUP3'] = 2.47654
        v_['TUP4'] = 1.25276
        v_['TUP5'] = 1.43508
        v_['BLO1'] = 4.45966
        v_['BLO2'] = 3.74950
        v_['BLO3'] = 4.49144
        v_['BLO4'] = 3.14988
        v_['BLO5'] = 3.04452
        v_['BUP1'] = 397.747
        v_['BUP2'] = 882.353
        v_['BUP3'] = 833.333
        v_['BUP4'] = 638.298
        v_['BUP5'] = 666.667
        v_['Q1'] = 250000.0
        v_['Q2'] = 150000.0
        v_['Q3'] = 180000.0
        v_['Q4'] = 160000.0
        v_['Q5'] = 120000.0
        v_['LOGI1'] = jnp.log(1.0)
        v_['LOGI2'] = jnp.log(2.0)
        v_['LOGI3'] = jnp.log(3.0)
        v_['LOGI4'] = jnp.log(4.0)
        v_['S1,1'] = jnp.log(7.9)
        v_['S2,1'] = jnp.log(0.7)
        v_['S3,1'] = jnp.log(0.7)
        v_['S4,1'] = jnp.log(4.7)
        v_['S5,1'] = jnp.log(1.2)
        v_['S1,2'] = jnp.log(2.0)
        v_['S2,2'] = jnp.log(0.8)
        v_['S3,2'] = jnp.log(2.6)
        v_['S4,2'] = jnp.log(2.3)
        v_['S5,2'] = jnp.log(3.6)
        v_['S1,3'] = jnp.log(5.2)
        v_['S2,3'] = jnp.log(0.9)
        v_['S3,3'] = jnp.log(1.6)
        v_['S4,3'] = jnp.log(1.6)
        v_['S5,3'] = jnp.log(2.4)
        v_['S1,4'] = jnp.log(4.9)
        v_['S2,4'] = jnp.log(3.4)
        v_['S3,4'] = jnp.log(3.6)
        v_['S4,4'] = jnp.log(2.7)
        v_['S5,4'] = jnp.log(4.5)
        v_['S1,5'] = jnp.log(6.1)
        v_['S2,5'] = jnp.log(2.1)
        v_['S3,5'] = jnp.log(3.2)
        v_['S4,5'] = jnp.log(1.2)
        v_['S5,5'] = jnp.log(1.6)
        v_['S1,6'] = jnp.log(4.2)
        v_['S2,6'] = jnp.log(2.5)
        v_['S3,6'] = jnp.log(2.9)
        v_['S4,6'] = jnp.log(2.5)
        v_['S5,6'] = jnp.log(2.1)
        v_['T1,1'] = jnp.log(6.4)
        v_['T2,1'] = jnp.log(6.8)
        v_['T3,1'] = jnp.log(1.0)
        v_['T4,1'] = jnp.log(3.2)
        v_['T5,1'] = jnp.log(2.1)
        v_['T1,2'] = jnp.log(4.7)
        v_['T2,2'] = jnp.log(6.4)
        v_['T3,2'] = jnp.log(6.3)
        v_['T4,2'] = jnp.log(3.0)
        v_['T5,2'] = jnp.log(2.5)
        v_['T1,3'] = jnp.log(8.3)
        v_['T2,3'] = jnp.log(6.5)
        v_['T3,3'] = jnp.log(5.4)
        v_['T4,3'] = jnp.log(3.5)
        v_['T5,3'] = jnp.log(4.2)
        v_['T1,4'] = jnp.log(3.9)
        v_['T2,4'] = jnp.log(4.4)
        v_['T3,4'] = jnp.log(11.9)
        v_['T4,4'] = jnp.log(3.3)
        v_['T5,4'] = jnp.log(3.6)
        v_['T1,5'] = jnp.log(2.1)
        v_['T2,5'] = jnp.log(2.3)
        v_['T3,5'] = jnp.log(5.7)
        v_['T4,5'] = jnp.log(2.8)
        v_['T5,5'] = jnp.log(3.7)
        v_['T1,6'] = jnp.log(1.2)
        v_['T2,6'] = jnp.log(3.2)
        v_['T3,6'] = jnp.log(6.2)
        v_['T4,6'] = jnp.log(3.4)
        v_['T5,6'] = jnp.log(2.2)
        for J in range(int(v_['1']),int(v_['M'])+1):
            v_['ALPHA'+str(J)] = 250.0
            v_['BETA'+str(J)] = 0.6
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for J in range(int(v_['1']),int(v_['M'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('N'+str(J),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'N'+str(J))
        for J in range(int(v_['1']),int(v_['M'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('V'+str(J),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'V'+str(J))
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('B'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'B'+str(I))
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('TL'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'TL'+str(I))
        for J in range(int(v_['1']),int(v_['M'])+1):
            for K in range(int(v_['1']),int(v_['NU'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('Y'+str(K)+','+str(J),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'Y'+str(K)+','+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('COST',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['M'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('VOL'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'VOL'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['V'+str(J)]])
                valA = jtu.append(valA,float(1.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['B'+str(I)]])
                valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['M'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('CYCL'+str(I)+','+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'CYCL'+str(I)+','+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['N'+str(J)]])
                valA = jtu.append(valA,float(1.0))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['TL'+str(I)]])
                valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('HORIZON',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'HORIZON')
        for J in range(int(v_['1']),int(v_['M'])+1):
            for K in range(int(v_['1']),int(v_['NU'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('NPAR'+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'NPAR'+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(K)+','+str(J)]])
                valA = jtu.append(valA,float(v_['LOGI'+str(K)]))
            [ig,ig_,_] = jtu.s2mpj_ii('NPAR'+str(J),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'NPAR'+str(J))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['N'+str(J)]])
            valA = jtu.append(valA,float(-1.0))
        for J in range(int(v_['1']),int(v_['M'])+1):
            for K in range(int(v_['1']),int(v_['NU'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('SOS1'+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'SOS1'+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['Y'+str(K)+','+str(J)]])
                valA = jtu.append(valA,float(1.0))
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['M'])+1):
                self.gconst  = (                       jtu.arrset(self.gconst,ig_['VOL'+str(I)+','+str(J)],float(v_['S'+str(I)+','+str(J)])))
        for I in range(int(v_['1']),int(v_['N'])+1):
            for J in range(int(v_['1']),int(v_['M'])+1):
                self.gconst  = (                       jtu.arrset(self.gconst,ig_['CYCL'+str(I)+','+str(J)],float(v_['T'+str(I)+','+str(J)])))
        self.gconst = jtu.arrset(self.gconst,ig_['HORIZON'],float(v_['H']))
        for J in range(int(v_['1']),int(v_['M'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['SOS1'+str(J)],float(1.0))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        for J in range(int(v_['1']),int(v_['M'])+1):
            self.xupper = jtu.np_like_set(self.xupper, ix_['N'+str(J)], v_['LOGNU'])
            self.xlower = jtu.np_like_set(self.xlower, ix_['V'+str(J)], v_['VL'])
            self.xupper = jtu.np_like_set(self.xupper, ix_['V'+str(J)], v_['VU'])
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['B'+str(I)], v_['BLO'+str(I)])
            self.xupper = jtu.np_like_set(self.xupper, ix_['B'+str(I)], v_['BUP'+str(I)])
            self.xlower = jtu.np_like_set(self.xlower, ix_['TL'+str(I)], v_['TLO'+str(I)])
            self.xupper = jtu.np_like_set(self.xupper, ix_['TL'+str(I)], v_['TUP'+str(I)])
        for J in range(int(v_['1']),int(v_['M'])+1):
            for K in range(int(v_['1']),int(v_['NU'])+1):
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(K)+','+str(J)]]), 1.0)
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eEXPXAY', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'A')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for J in range(int(v_['1']),int(v_['M'])+1):
            ename = 'EXPO'+str(J)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eEXPXAY')
            ielftype = jtu.arrset(ielftype,ie,iet_["eEXPXAY"])
            self.x0 = jnp.zeros((self.n,1))
            vname = 'N'+str(J)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'V'+str(J)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='A')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['BETA'+str(J)]))
        for I in range(int(v_['1']),int(v_['M'])+1):
            ename = 'EXPC'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eEXPXAY')
            ielftype = jtu.arrset(ielftype,ie,iet_["eEXPXAY"])
            vname = 'TL'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'B'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='A')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(-1.0))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for J in range(int(v_['1']),int(v_['M'])+1):
            ig = ig_['COST']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EXPO'+str(J)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['ALPHA'+str(J)]))
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['HORIZON']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EXPC'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['Q'+str(I)]))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COOR2-AN-46-73"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eEXPXAY(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        FVALUE = jnp.exp(EV_[0]+self.elpar[iel_][0]*EV_[1])
        GYVALU = self.elpar[iel_][0]*FVALUE
        f_   = FVALUE
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, FVALUE)
            g_ = jtu.np_like_set(g_, 1, GYVALU)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), FVALUE)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), GYVALU)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), self.elpar[iel_][0]*GYVALU)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HS111:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HS111
#    *********
# 
#    This problem is a chemical equilibrium problem involving 3 linear
#    equality constraints.
# 
#    Source: problem 111 in
#    W. Hock and K. Schittkowski,
#    "Test examples for nonlinear programming codes",
#    Lectures Notes in Economics and Mathematical Systems 187, Springer
#    Verlag, Heidelberg, 1981.
# 
#    SIF input: Nick Gould, August 1991.
# 
#    classification = "C-COOR2-AN-10-3"
# 
#    N is the number of variables
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HS111'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 10
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
        v_['C1'] = -6.089
        v_['C2'] = -17.164
        v_['C3'] = -34.054
        v_['C4'] = -5.914
        v_['C5'] = -24.721
        v_['C6'] = -14.986
        v_['C7'] = -24.100
        v_['C8'] = -10.708
        v_['C9'] = -26.662
        v_['C10'] = -22.179
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
        [ig,ig_,_] = jtu.s2mpj_ii('CON1',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'CON1')
        [ig,ig_,_] = jtu.s2mpj_ii('CON2',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'CON2')
        [ig,ig_,_] = jtu.s2mpj_ii('CON3',ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'CON3')
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
        self.gconst = jtu.arrset(self.gconst,ig_['CON1'],float(2.0))
        self.gconst = jtu.arrset(self.gconst,ig_['CON2'],float(1.0))
        self.gconst = jtu.arrset(self.gconst,ig_['CON3'],float(1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-100.0)
        self.xupper = jnp.full((self.n,1),100.0)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(-2.3))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eOBJ', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        elftv = jtu.loaset(elftv,it,1,'V2')
        elftv = jtu.loaset(elftv,it,2,'V3')
        elftv = jtu.loaset(elftv,it,3,'V4')
        elftv = jtu.loaset(elftv,it,4,'V5')
        elftv = jtu.loaset(elftv,it,5,'V6')
        elftv = jtu.loaset(elftv,it,6,'V7')
        elftv = jtu.loaset(elftv,it,7,'V8')
        elftv = jtu.loaset(elftv,it,8,'V9')
        elftv = jtu.loaset(elftv,it,9,'V10')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'C')
        [it,iet_,_] = jtu.s2mpj_ii( 'eEXP', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'+str(I)] = float(I)
            v_['RI'+str(I)] = 0.1+v_['RI'+str(I)]
        for I in range(int(v_['1']),int(v_['N'])+1):
            ename = 'O'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eOBJ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eOBJ"])
            v_['TEMP'] = v_['RI'+str(int(v_['1']))]
            v_['RI'+str(int(v_['1']))] = v_['RI'+str(I)]
            v_['RI'+str(I)] = v_['TEMP']
            v_['R'] = v_['RI'+str(int(v_['1']))]
            v_['J'] = int(jnp.fix(v_['R']))
            vname = 'X'+str(int(v_['J']))
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['R'] = v_['RI'+str(int(v_['2']))]
            v_['J'] = int(jnp.fix(v_['R']))
            vname = 'X'+str(int(v_['J']))
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['R'] = v_['RI'+str(int(v_['3']))]
            v_['J'] = int(jnp.fix(v_['R']))
            vname = 'X'+str(int(v_['J']))
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='V3')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['R'] = v_['RI'+str(int(v_['4']))]
            v_['J'] = int(jnp.fix(v_['R']))
            vname = 'X'+str(int(v_['J']))
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='V4')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['R'] = v_['RI'+str(int(v_['5']))]
            v_['J'] = int(jnp.fix(v_['R']))
            vname = 'X'+str(int(v_['J']))
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='V5')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['R'] = v_['RI'+str(int(v_['6']))]
            v_['J'] = int(jnp.fix(v_['R']))
            vname = 'X'+str(int(v_['J']))
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='V6')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['R'] = v_['RI'+str(int(v_['7']))]
            v_['J'] = int(jnp.fix(v_['R']))
            vname = 'X'+str(int(v_['J']))
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='V7')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['R'] = v_['RI'+str(int(v_['8']))]
            v_['J'] = int(jnp.fix(v_['R']))
            vname = 'X'+str(int(v_['J']))
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='V8')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['R'] = v_['RI'+str(int(v_['9']))]
            v_['J'] = int(jnp.fix(v_['R']))
            vname = 'X'+str(int(v_['J']))
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='V9')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            v_['R'] = v_['RI'+str(int(v_['10']))]
            v_['J'] = int(jnp.fix(v_['R']))
            vname = 'X'+str(int(v_['J']))
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='V10')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='C')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['C'+str(I)]))
            ename = 'E'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eEXP')
            ielftype = jtu.arrset(ielftype,ie,iet_["eEXP"])
            vname = 'X'+str(I)
            [iv,ix_]  = (
                  jtu.s2mpj_nlx(self,vname,ix_,1,float(-100.0),float(100.0),float(-2.3)))
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['OBJ']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['O'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['CON1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E1'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E2'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(2.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(2.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E6'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E10'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        ig = ig_['CON2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E4'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E5'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(2.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E6'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E7'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        ig = ig_['CON3']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E3'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E7'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E8'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E9'])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(2.0))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E10'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               -47.707579
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COOR2-AN-10-3"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eEXP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        EX = jnp.exp(EV_[0])
        f_   = EX
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EX)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), EX)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eOBJ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        E1 = jnp.exp(EV_[0])
        E2 = jnp.exp(EV_[1])
        E3 = jnp.exp(EV_[2])
        E4 = jnp.exp(EV_[3])
        E5 = jnp.exp(EV_[4])
        E6 = jnp.exp(EV_[5])
        E7 = jnp.exp(EV_[6])
        E8 = jnp.exp(EV_[7])
        E9 = jnp.exp(EV_[8])
        E10 = jnp.exp(EV_[9])
        SUM = E1+E2+E3+E4+E5+E6+E7+E8+E9+E10
        f_   = E1*(self.elpar[iel_][0]+EV_[0]-jnp.log(SUM))
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, E1*(self.elpar[iel_][0]+EV_[0]-jnp.log(SUM))+E1*(1.0e+0-E1/SUM))
            g_ = jtu.np_like_set(g_, 1, -E1*E2/SUM)
            g_ = jtu.np_like_set(g_, 2, -E1*E3/SUM)
            g_ = jtu.np_like_set(g_, 3, -E1*E4/SUM)
            g_ = jtu.np_like_set(g_, 4, -E1*E5/SUM)
            g_ = jtu.np_like_set(g_, 5, -E1*E6/SUM)
            g_ = jtu.np_like_set(g_, 6, -E1*E7/SUM)
            g_ = jtu.np_like_set(g_, 7, -E1*E8/SUM)
            g_ = jtu.np_like_set(g_, 8, -E1*E9/SUM)
            g_ = jtu.np_like_set(g_, 9, -E1*E10/SUM)
            if nargout>2:
                H_ = jnp.zeros((10,10))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), (E1*(self.elpar[iel_][0]+EV_[0]-jnp.log(SUM))+E1*(1.0e+0-E1/SUM)+)
                     E1*(1.0e+0-E1/SUM)+E1*(-E1/SUM)+E1*(E1**2/SUM**2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), (-1.0e+0+E1/SUM)*E1*E2/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), (-1.0e+0+E2/SUM)*E1*E2/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), (-1.0e+0+E1/SUM)*E1*E3/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), E1*E2*E3/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), (-1.0e+0+E3/SUM)*E1*E3/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), (-1.0e+0+E1/SUM)*E1*E4/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), E1*E2*E4/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), E1*E3*E4/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), (-1.0e+0+E4/SUM)*E1*E4/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([0,4]), (-1.0e+0+E1/SUM)*E1*E5/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([4,0]), H_[0,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,4]), E1*E2*E5/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([4,1]), H_[1,4])
                H_ = jtu.np_like_set(H_, jnp.array([2,4]), E1*E3*E5/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([4,2]), H_[2,4])
                H_ = jtu.np_like_set(H_, jnp.array([3,4]), E1*E4*E5/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([4,3]), H_[3,4])
                H_ = jtu.np_like_set(H_, jnp.array([4,4]), (-1.0e+0+E5/SUM)*E1*E5/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([0,5]), (-1.0e+0+E1/SUM)*E1*E6/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([5,0]), H_[0,5])
                H_ = jtu.np_like_set(H_, jnp.array([1,5]), E1*E2*E6/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([5,1]), H_[1,5])
                H_ = jtu.np_like_set(H_, jnp.array([2,5]), E1*E3*E6/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([5,2]), H_[2,5])
                H_ = jtu.np_like_set(H_, jnp.array([3,5]), E1*E4*E6/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([5,3]), H_[3,5])
                H_ = jtu.np_like_set(H_, jnp.array([4,5]), E1*E5*E6/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([5,4]), H_[4,5])
                H_ = jtu.np_like_set(H_, jnp.array([5,5]), (-1.0e+0+E6/SUM)*E1*E6/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([0,6]), (-1.0e+0+E1/SUM)*E1*E7/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([6,0]), H_[0,6])
                H_ = jtu.np_like_set(H_, jnp.array([1,6]), E1*E2*E7/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([6,1]), H_[1,6])
                H_ = jtu.np_like_set(H_, jnp.array([2,6]), E1*E3*E7/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([6,2]), H_[2,6])
                H_ = jtu.np_like_set(H_, jnp.array([3,6]), E1*E4*E7/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([6,3]), H_[3,6])
                H_ = jtu.np_like_set(H_, jnp.array([4,6]), E1*E5*E7/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([6,4]), H_[4,6])
                H_ = jtu.np_like_set(H_, jnp.array([5,6]), E1*E6*E7/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([6,5]), H_[5,6])
                H_ = jtu.np_like_set(H_, jnp.array([6,6]), (-1.0e+0+E7/SUM)*E1*E7/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([0,7]), (-1.0e+0+E1/SUM)*E1*E8/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([7,0]), H_[0,7])
                H_ = jtu.np_like_set(H_, jnp.array([1,7]), E1*E2*E8/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([7,1]), H_[1,7])
                H_ = jtu.np_like_set(H_, jnp.array([2,7]), E1*E3*E8/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([7,2]), H_[2,7])
                H_ = jtu.np_like_set(H_, jnp.array([3,7]), E1*E4*E8/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([7,3]), H_[3,7])
                H_ = jtu.np_like_set(H_, jnp.array([4,7]), E1*E5*E8/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([7,4]), H_[4,7])
                H_ = jtu.np_like_set(H_, jnp.array([5,7]), E1*E6*E8/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([7,5]), H_[5,7])
                H_ = jtu.np_like_set(H_, jnp.array([6,7]), E1*E7*E8/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([7,6]), H_[6,7])
                H_ = jtu.np_like_set(H_, jnp.array([7,7]), (-1.0e+0+E8/SUM)*E1*E8/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([0,8]), (-1.0e+0+E1/SUM)*E1*E9/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([8,0]), H_[0,8])
                H_ = jtu.np_like_set(H_, jnp.array([1,8]), E1*E2*E9/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([8,1]), H_[1,8])
                H_ = jtu.np_like_set(H_, jnp.array([2,8]), E1*E3*E9/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([8,2]), H_[2,8])
                H_ = jtu.np_like_set(H_, jnp.array([3,8]), E1*E4*E9/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([8,3]), H_[3,8])
                H_ = jtu.np_like_set(H_, jnp.array([4,8]), E1*E5*E9/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([8,4]), H_[4,8])
                H_ = jtu.np_like_set(H_, jnp.array([5,8]), E1*E6*E9/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([8,5]), H_[5,8])
                H_ = jtu.np_like_set(H_, jnp.array([6,8]), E1*E7*E9/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([8,6]), H_[6,8])
                H_ = jtu.np_like_set(H_, jnp.array([7,8]), E1*E8*E9/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([8,7]), H_[7,8])
                H_ = jtu.np_like_set(H_, jnp.array([8,8]), (-1.0e+0+E9/SUM)*E1*E9/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([0,9]), (-1.0e+0+E1/SUM)*E1*E10/SUM)
                H_ = jtu.np_like_set(H_, jnp.array([9,0]), H_[0,9])
                H_ = jtu.np_like_set(H_, jnp.array([1,9]), E1*E2*E10/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([9,1]), H_[1,9])
                H_ = jtu.np_like_set(H_, jnp.array([2,9]), E1*E3*E10/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([9,2]), H_[2,9])
                H_ = jtu.np_like_set(H_, jnp.array([3,9]), E1*E4*E10/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([9,3]), H_[3,9])
                H_ = jtu.np_like_set(H_, jnp.array([4,9]), E1*E5*E10/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([9,4]), H_[4,9])
                H_ = jtu.np_like_set(H_, jnp.array([5,9]), E1*E6*E10/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([9,5]), H_[5,9])
                H_ = jtu.np_like_set(H_, jnp.array([6,9]), E1*E7*E10/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([9,6]), H_[6,9])
                H_ = jtu.np_like_set(H_, jnp.array([7,9]), E1*E8*E10/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([9,7]), H_[7,9])
                H_ = jtu.np_like_set(H_, jnp.array([8,9]), E1*E9*E10/SUM**2)
                H_ = jtu.np_like_set(H_, jnp.array([9,8]), H_[8,9])
                H_ = jtu.np_like_set(H_, jnp.array([9,9]), (-1.0e+0+E10/SUM)*E1*E10/SUM)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HATFLDENE:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HATFLDENE
#    *********
# 
#    An exponential fitting  test problem from the OPTIMA user manual.
# 
#    Source:
#    "The OPTIMA user manual (issue No.8, p. 37)",
#    Numerical Optimization Centre, Hatfield Polytechnic (UK), 1989.
# 
#    SIF input: Ph. Toint, May 1990.
#    Nonlinear-equations version of HATFLDE.SIF, Nick Gould, Jan 2020.
# 
#    classification = "C-CNOR2-AN-3-21"
# 
#    Problem data
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HATFLDENE'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['NG'] = 21
        v_['T1'] = 0.3
        v_['T2'] = 0.35
        v_['T3'] = 0.4
        v_['T4'] = 0.45
        v_['T5'] = 0.5
        v_['T6'] = 0.55
        v_['T7'] = 0.6
        v_['T8'] = 0.65
        v_['T9'] = 0.7
        v_['T10'] = 0.75
        v_['T11'] = 0.8
        v_['T12'] = 0.85
        v_['T13'] = 0.9
        v_['T14'] = 0.95
        v_['T15'] = 1.0
        v_['T16'] = 1.05
        v_['T17'] = 1.1
        v_['T18'] = 1.15
        v_['T19'] = 1.2
        v_['T20'] = 1.25
        v_['T21'] = 1.3
        v_['Z1'] = 1.561
        v_['Z2'] = 1.473
        v_['Z3'] = 1.391
        v_['Z4'] = 1.313
        v_['Z5'] = 1.239
        v_['Z6'] = 1.169
        v_['Z7'] = 1.103
        v_['Z8'] = 1.04
        v_['Z9'] = 0.981
        v_['Z10'] = 0.925
        v_['Z11'] = 0.8721
        v_['Z12'] = 0.8221
        v_['Z13'] = 0.7748
        v_['Z14'] = 0.73
        v_['Z15'] = 0.6877
        v_['Z16'] = 0.6477
        v_['Z17'] = 0.6099
        v_['Z18'] = 0.5741
        v_['Z19'] = 0.5403
        v_['Z20'] = 0.5084
        v_['Z21'] = 0.4782
        v_['1'] = 1
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
        [iv,ix_,_] = jtu.s2mpj_ii('X3',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'X3')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['NG'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'G'+str(I))
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
        for I in range(int(v_['1']),int(v_['NG'])+1):
            v_['-ZI'] = -1.0*v_['Z'+str(I)]
            self.gconst = jtu.arrset(self.gconst,ig_['G'+str(I)],float(v_['-ZI']))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        if('X1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X1'], float(1.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X1']),float(1.0)))
        if('X2' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X2'], float(-1.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X2']),float(-1.0)))
        if('X3' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['X3'], float(0.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X3']),float(0.0)))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eXPEXP', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'T')
        [it,iet_,_] = jtu.s2mpj_ii( 'ePEXP', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftp = jtu.loaset(elftp,it,0,'T')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['NG'])+1):
            ename = 'A'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eXPEXP')
            ielftype = jtu.arrset(ielftype,ie,iet_["eXPEXP"])
            vname = 'X1'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X2'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='T')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['T'+str(I)]))
            ename = 'B'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'ePEXP')
            ielftype = jtu.arrset(ielftype,ie,iet_["ePEXP"])
            vname = 'X3'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='T')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['T'+str(I)]))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['NG'])+1):
            ig = ig_['G'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['A'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['B'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN               5.120377D-07
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CNOR2-AN-3-21"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePEXP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        EX = jnp.exp(self.elpar[iel_][0]*EV_[0])
        f_   = EX
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, self.elpar[iel_][0]*EX)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), self.elpar[iel_][0]*self.elpar[iel_][0]*EX)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eXPEXP(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        EX = jnp.exp(self.elpar[iel_][0]*EV_[1])
        f_   = EV_[0]*EX
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EX)
            g_ = jtu.np_like_set(g_, 1, self.elpar[iel_][0]*EV_[0]*EX)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), self.elpar[iel_][0]*EX)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), self.elpar[iel_][0]*self.elpar[iel_][0]*EV_[0]*EX)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


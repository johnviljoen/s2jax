from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HS268:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HS268
#    *********
# 
#    A quadratic programming problem.
# 
#    Source:
#    K. Schittkowski
#    "More Test Examples for Nonlinear Programming Codes"
#    Springer Verlag, Berlin, Lecture notes in economics and 
#    mathematical systems, volume 282, 1987
# 
#    SIF input: Michel Bierlaire and Annick Sartenaer, October 1992.
#               minor correction by Ph. Shott, Jan 1995.
# 
#    classification = "C-CQLR2-AN-5-5"
# 
#   the number of functions
# 
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HS268'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['D1,1'] = 10197.0
        v_['D1,2'] = -12454.0
        v_['D1,3'] = -1013.0
        v_['D1,4'] = 1948.0
        v_['D1,5'] = 329.0
        v_['D2,1'] = -12454.0
        v_['D2,2'] = 20909.0
        v_['D2,3'] = -1733.0
        v_['D2,4'] = -4914.0
        v_['D2,5'] = -186.0
        v_['D3,1'] = -1013.0
        v_['D3,2'] = -1733.0
        v_['D3,3'] = 1755.0
        v_['D3,4'] = 1089.0
        v_['D3,5'] = -174.0
        v_['D4,1'] = 1948.0
        v_['D4,2'] = -4914.0
        v_['D4,3'] = 1089.0
        v_['D4,4'] = 1515.0
        v_['D4,5'] = -22.0
        v_['D5,1'] = 329.0
        v_['D5,2'] = -186.0
        v_['D5,3'] = -174.0
        v_['D5,4'] = -22.0
        v_['D5,5'] = 27.0
        v_['B1'] = -9170.0
        v_['B2'] = 17099.0
        v_['B3'] = -2271.0
        v_['B4'] = -4336.0
        v_['B5'] = -43.0
        v_['1'] = 1
        v_['5'] = 5
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['5'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('NONL',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for I in range(int(v_['1']),int(v_['5'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('LINEAR',ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(v_['B'+str(I)]))
        [ig,ig_,_] = jtu.s2mpj_ii('LINEAR',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(-0.5))
        [ig,ig_,_] = jtu.s2mpj_ii('C1',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X4']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X5']])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C2',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(10.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(10.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(-3.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X4']])
        valA = jtu.append(valA,float(5.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X5']])
        valA = jtu.append(valA,float(4.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C3',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(-8.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(-2.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X4']])
        valA = jtu.append(valA,float(-5.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X5']])
        valA = jtu.append(valA,float(3.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C4',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C4')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(8.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(-1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(2.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X4']])
        valA = jtu.append(valA,float(5.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X5']])
        valA = jtu.append(valA,float(-3.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C5',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C5')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(-4.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(-2.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(3.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X4']])
        valA = jtu.append(valA,float(-5.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X5']])
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
        self.gconst = jtu.arrset(self.gconst,ig_['C1'],float(-5.0))
        self.gconst = jtu.arrset(self.gconst,ig_['C2'],float(20.0))
        self.gconst = jtu.arrset(self.gconst,ig_['C3'],float(-40.0))
        self.gconst = jtu.arrset(self.gconst,ig_['C4'],float(11.0))
        self.gconst = jtu.arrset(self.gconst,ig_['C5'],float(-30.0))
        self.gconst = jtu.arrset(self.gconst,ig_['NONL'],float(-14463.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(1.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'D')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['5'])+1):
            for J in range(int(v_['1']),int(v_['5'])+1):
                ename = 'E'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
                ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
                vname = 'X'+str(I)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
                posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                posep = jnp.where(elftp[ielftype[ie]]=='D')[0]
                self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['D'+str(I)+','+str(J)]))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['5'])+1):
            for J in range(int(v_['1']),int(v_['5'])+1):
                ig = ig_['NONL']
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CQLR2-AN-5-5"
        self.objderlvl = 2
        self.conderlvl = [2]


    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePROD(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = self.elpar[iel_][0]*EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, self.elpar[iel_][0]*EV_[1])
            g_ = jtu.np_like_set(g_, 1, self.elpar[iel_][0]*EV_[0])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), 0.0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), self.elpar[iel_][0])
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


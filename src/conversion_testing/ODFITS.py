import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class ODFITS:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    A simple Origin/Destination matrix fit using a minimum entropy
#    approach.  The objective is a combination of different aims, namely
#    to be close to an a priori matrix for some entries, to be consistent
#    with some traffic counts (for some entries) and to be small (for entries
#    where nothing else is known).
# 
#    The objective function is of the form
#         SUM   m T [ ln( T / a ) - 1 ] + E   SUM  T [ ln ( T  ) - 1 ]
#        i in I  i i       i   i            i in J  i        i
#                +  g   SUM   q  F [ ln( F / c ) - 1 ]
#                     i in K   i  i       i   i
#    with the constraints that all Ti and Fi be positive and that
#                         F  =  SUM p   T
#                          i     j   ij  j
#    where the pij represent path weights from an a priori assignment.
# 
#    Source: a modification of an example in
#    L.G. Willumsen,
#    "Origin-Destination Matrix: static estimation"
#    in "Concise Encyclopedia of Traffic and Transportation Systems"
#    (M. Papageorgiou, ed.), Pergamon Press, 1991.
# 
#    M. Bierlaire, private communication, 1991.
# 
#    SIF input: Ph Toint, Dec 1991.
# 
#    classification = "C-COLR2-MN-10-6"
# 
#    Number of available traffic counts
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'ODFITS'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['ARCS'] = 6
        v_['TC1'] = 100.0
        v_['TC2'] = 500.0
        v_['TC3'] = 400.0
        v_['TC4'] = 1100.0
        v_['TC5'] = 600.0
        v_['TC6'] = 700.0
        v_['QLT1'] = 1.0
        v_['QLT2'] = 1.0
        v_['QLT3'] = 1.0
        v_['QLT4'] = 1.0
        v_['QLT5'] = 1.0
        v_['QLT6'] = 1.0
        v_['P131'] = 1.0
        v_['P132'] = 0.0
        v_['P133'] = 0.0
        v_['P134'] = 0.0
        v_['P135'] = 0.0
        v_['P136'] = 0.0
        v_['P141'] = 0.0
        v_['P142'] = 1.0
        v_['P143'] = 0.0
        v_['P144'] = 1.0
        v_['P145'] = 0.0
        v_['P146'] = 0.0
        v_['P231'] = 0.0
        v_['P232'] = 0.0
        v_['P233'] = 1.0
        v_['P234'] = 1.0
        v_['P235'] = 1.0
        v_['P236'] = 0.0
        v_['P241'] = 0.0
        v_['P242'] = 0.0
        v_['P243'] = 0.0
        v_['P244'] = 1.0
        v_['P245'] = 1.0
        v_['P246'] = 1.0
        v_['APV13'] = 90.0
        v_['APV14'] = 450.0
        v_['APV23'] = 360.0
        v_['MU13'] = 0.5
        v_['MU14'] = 0.5
        v_['MU23'] = 0.5
        v_['1/MU13'] = 1.0/v_['MU13']
        v_['1/MU14'] = 1.0/v_['MU14']
        v_['1/MU23'] = 1.0/v_['MU23']
        v_['GAMMA'] = 1.5
        v_['ENTROP'] = 0.2
        v_['1/ENTR'] = 1.0/v_['ENTROP']
        v_['1'] = 1
        for I in range(int(v_['1']),int(v_['ARCS'])+1):
            v_['1/QLT'+str(I)] = 1.0/v_['QLT'+str(I)]
            v_['G/QLT'+str(I)] = v_['1/QLT'+str(I)]*v_['GAMMA']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('T13',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'T13')
        [iv,ix_,_] = jtu.s2mpj_ii('T14',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'T14')
        [iv,ix_,_] = jtu.s2mpj_ii('T23',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'T23')
        [iv,ix_,_] = jtu.s2mpj_ii('T24',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'T24')
        for I in range(int(v_['1']),int(v_['ARCS'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('F'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'F'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('AP13',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T13']])
        valA = jtu.append(valA,float(-1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['1/MU13']))
        [ig,ig_,_] = jtu.s2mpj_ii('AP14',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T14']])
        valA = jtu.append(valA,float(-1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['1/MU14']))
        [ig,ig_,_] = jtu.s2mpj_ii('AP23',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T23']])
        valA = jtu.append(valA,float(-1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['1/MU23']))
        [ig,ig_,_] = jtu.s2mpj_ii('AP24',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T24']])
        valA = jtu.append(valA,float(-1.0))
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['1/ENTR']))
        for I in range(int(v_['1']),int(v_['ARCS'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('CP'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['F'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['G/QLT'+str(I)]))
        for I in range(int(v_['1']),int(v_['ARCS'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'C'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['F'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T13']])
            valA = jtu.append(valA,float(v_['P13'+str(I)]))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T14']])
            valA = jtu.append(valA,float(v_['P14'+str(I)]))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T23']])
            valA = jtu.append(valA,float(v_['P23'+str(I)]))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T24']])
            valA = jtu.append(valA,float(v_['P24'+str(I)]))
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
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),0.1)
        self.xupper = jnp.full((self.n,1),+float('inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        self.x0 = jtu.np_like_set(self.x0, ix_['T13'], float(v_['APV13']))
        self.x0 = jtu.np_like_set(self.x0, ix_['T14'], float(v_['APV14']))
        self.x0 = jtu.np_like_set(self.x0, ix_['T23'], float(v_['APV23']))
        self.x0 = jtu.np_like_set(self.x0, ix_['T24'], float(1.0))
        for I in range(int(v_['1']),int(v_['ARCS'])+1):
            self.x0 = jtu.np_like_set(self.x0, ix_['F'+str(I)], float(v_['TC'+str(I)]))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eXLOGX', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'DEN')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        ename = 'TFIT13'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eXLOGX')
        ielftype = jtu.arrset(ielftype,ie,iet_["eXLOGX"])
        vname = 'T13'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.1),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='DEN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['APV13']))
        ename = 'TFIT23'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eXLOGX')
        ielftype = jtu.arrset(ielftype,ie,iet_["eXLOGX"])
        vname = 'T23'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.1),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='DEN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['APV23']))
        ename = 'TFIT14'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eXLOGX')
        ielftype = jtu.arrset(ielftype,ie,iet_["eXLOGX"])
        vname = 'T14'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.1),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='DEN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['APV14']))
        ename = 'TFIT24'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eXLOGX')
        ielftype = jtu.arrset(ielftype,ie,iet_["eXLOGX"])
        vname = 'T24'
        [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.1),None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        posep = jnp.where(elftp[ielftype[ie]]=='DEN')[0]
        self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.0))
        for I in range(int(v_['1']),int(v_['ARCS'])+1):
            ename = 'CFIT'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eXLOGX')
            ielftype = jtu.arrset(ielftype,ie,iet_["eXLOGX"])
            vname = 'F'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(0.1),None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='DEN')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['TC'+str(I)]))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['AP13']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['TFIT13'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AP14']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['TFIT14'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AP23']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['TFIT23'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['AP24']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['TFIT24'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        for I in range(int(v_['1']),int(v_['ARCS'])+1):
            ig = ig_['CP'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['CFIT'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO ODFITS             -2380.026775
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COLR2-MN-10-6"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eXLOGX(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        LOGX = jnp.log(EV_[0]/self.elpar[iel_][0])
        f_   = EV_[0]*LOGX
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 1.0+LOGX)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 1.0/EV_[0])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


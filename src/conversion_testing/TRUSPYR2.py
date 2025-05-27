from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class TRUSPYR2:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    This is a structural optimization problem.
#    The problem is to minimize the weight of a given
#    8-bar truss structure formed as a pyramid for a given external load.
#    There are upper bounds on the normal stresses in the
#    bars and lower bounds on the cross-sectional areas of the bars.
# 
#    Source:
#    K. Svanberg, 
#    "Local and global optima", 
#    Proceedings of the NATO/DFG ASI on Optimization of large structural
#    systems, 
#    G. I. N. Rozvany, ed., Kluwer, 1993, pp. 579-588.
# 
#    SIF input: A. Forsgren, Royal Institute of Technology, December 1993.
# 
#    classification = "C-CLQR2-MN-11-11"
# 
#    Number of bars
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'TRUSPYR2'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['NBAR'] = 8
        v_['NDIM'] = 3
        v_['1'] = 1
        v_['2'] = 2
        v_['NBAR/2'] = int(jnp.fix(v_['NBAR']/v_['2']))
        v_['8.0'] = 8.0
        v_['SQRT17'] = jnp.sqrt(17.0)
        v_['SQRT18'] = jnp.sqrt(18.0)
        v_['P1'] = 40.0
        v_['P2'] = 20.0
        v_['P3'] = 200.0
        for J in range(int(v_['1']),int(v_['NBAR/2'])+1):
            v_['L'+str(J)] = v_['SQRT17']/v_['8.0']
            v_['J+4'] = J+v_['NBAR/2']
            v_['L'+str(int(v_['J+4']))] = v_['SQRT18']/v_['8.0']
        v_['E'] = 21.0
        v_['R1,1'] = 0.250
        v_['R2,1'] = 0.250
        v_['R3,1'] = 0.375
        v_['R1,2'] = 0.250
        v_['R2,2'] = -0.250
        v_['R3,2'] = 0.375
        v_['R1,3'] = -0.250
        v_['R2,3'] = -0.250
        v_['R3,3'] = 0.375
        v_['R1,4'] = -0.250
        v_['R2,4'] = 0.250
        v_['R3,4'] = 0.375
        v_['R1,5'] = 0.375
        v_['R2,5'] = 0.000
        v_['R3,5'] = 0.375
        v_['R1,6'] = 0.000
        v_['R2,6'] = -0.375
        v_['R3,6'] = 0.375
        v_['R1,7'] = -0.375
        v_['R2,7'] = 0.000
        v_['R3,7'] = 0.375
        v_['R1,8'] = 0.000
        v_['R2,8'] = 0.375
        v_['R3,8'] = 0.375
        for J in range(int(v_['1']),int(v_['NBAR'])+1):
            v_['L2'+str(J)] = v_['L'+str(J)]*v_['L'+str(J)]
            v_['L3'+str(J)] = v_['L2'+str(J)]*v_['L'+str(J)]
            v_['GAMMA'+str(J)] = v_['E']/v_['L3'+str(J)]
            v_['DL2'+str(J)] = v_['L2'+str(J)]/v_['E']
            v_['W'+str(J)] = 0.78*v_['L'+str(J)]
            v_['STRUP'+str(J)] = 10.0*v_['DL2'+str(J)]
            for I in range(int(v_['1']),int(v_['NDIM'])+1):
                v_['RG'+str(I)+','+str(J)] = v_['GAMMA'+str(J)]*v_['R'+str(I)+','+str(J)]
                for K in range(int(v_['1']),int(v_['NDIM'])+1):
                    v_['RR'+str(I)+','+str(J)+','+str(K)] = (v_['RG'+str(I)+','+str(J)]*v_['R'+
                         str(K)+','+str(J)])
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for J in range(int(v_['1']),int(v_['NBAR'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('XAREA'+str(J),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'XAREA'+str(J))
        for I in range(int(v_['1']),int(v_['NDIM'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('DISPL'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'DISPL'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for J in range(int(v_['1']),int(v_['NBAR'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('WEIGHT',ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['XAREA'+str(J)]])
            valA = jtu.append(valA,float(v_['W'+str(J)]))
        for K in range(int(v_['1']),int(v_['NDIM'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('EQUIL'+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'EQUIL'+str(K))
        for I in range(int(v_['1']),int(v_['NDIM'])+1):
            for J in range(int(v_['1']),int(v_['NBAR'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('STRES'+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'<=')
                cnames = jtu.arrset(cnames,ig,'STRES'+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['DISPL'+str(I)]])
                valA = jtu.append(valA,float(v_['R'+str(I)+','+str(J)]))
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
        for K in range(int(v_['1']),int(v_['NDIM'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['EQUIL'+str(K)],float(v_['P'+str(K)]))
        for J in range(int(v_['1']),int(v_['NBAR'])+1):
            self.gconst  = (
                  jtu.arrset(self.gconst,ig_['STRES'+str(J)],float(v_['STRUP'+str(J)])))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        for J in range(int(v_['1']),int(v_['NBAR'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['XAREA'+str(J)], 1.0)
        for I in range(int(v_['1']),int(v_['NDIM'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['DISPL'+str(I)], -float('Inf'))
            self.xupper = jtu.np_like_set(self.xupper, ix_['DISPL'+str(I)], +float('Inf'))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'en2PR', iet_)
        elftv = jtu.loaset(elftv,it,0,'U')
        elftv = jtu.loaset(elftv,it,1,'X')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['NDIM'])+1):
            for J in range(int(v_['1']),int(v_['NBAR'])+1):
                ename = 'UX'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
                ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
                self.x0 = jnp.zeros((self.n,1))
                vname = 'DISPL'+str(I)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'XAREA'+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['NDIM'])+1):
            for J in range(int(v_['1']),int(v_['NBAR'])+1):
                for K in range(int(v_['1']),int(v_['NDIM'])+1):
                    ig = ig_['EQUIL'+str(K)]
                    posel = len(self.grelt[ig])
                    self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['UX'+str(I)+','+str(J)])
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw  = (
                          jtu.loaset(self.grelw,ig,posel,float(v_['RR'+str(I)+','+str(J)+','+str(K)])))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Objective function value corresponding to the local minimizer above
        self.objlower = 1.2287408808
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CLQR2-MN-11-11"
        self.x0        = jnp.zeros((self.n,1))
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def en2PR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[1])
            g_ = jtu.np_like_set(g_, 1, EV_[0])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 1.0)
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


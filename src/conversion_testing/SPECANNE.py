import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class SPECANNE:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : SPECANNE
#    *********
# 
#    Source: a problem in spectral analysis suggested
#    by J. Eriksson and P. Lindstrom in "A Parallel Algorithm
#    for Bound Constrained Nonlinear Least Squares", UMEA TR S-901 87
# 
#    SIF input: Michael Ferris, July 1993
#    Bound-constrained nonlinear equations version: Nick Gould, June 2019.
# 
#    classification = "C-CNOR2-AN-V-V"
# 
#    Number of Gaussians
# 
#           Alternative values for the SIF file parameters:
# IE K                   1              $-PARAMETER
# IE K                   2              $-PARAMETER
# IE K                   3              $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'SPECANNE'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['K'] = int(3);  #  SIF file default value
        else:
            v_['K'] = int(args[0])
        v_['N'] = 3
        v_['M'] = 5000
        v_['RealM'] = float(v_['M'])
        v_['H'] = 25.0/v_['RealM']
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['ONE'] = 1.0
        v_['ROOTP5'] = jnp.sqrt(0.5)
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for p in range(int(v_['1']),int(v_['K'])+1):
            for j in range(int(v_['1']),int(v_['N'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('X'+str(p)+','+str(j),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'X'+str(p)+','+str(j))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for p in range(int(v_['1']),int(v_['K'])+1):
            for I in range(int(v_['1']),int(v_['M'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('OBJ'+str(p)+','+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'OBJ'+str(p)+','+str(I))
                self.gscale = jtu.arrset(self.gscale,ig,float(v_['ROOTP5']))
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
        v_['SOLN1,1'] = 19.0
        v_['SOLN1,2'] = 4.2
        v_['SOLN1,3'] = 1.2
        v_['SOLN2,1'] = 8.0
        v_['SOLN2,2'] = 2.5
        v_['SOLN2,3'] = 4.6
        v_['SOLN3,1'] = 10.0
        v_['SOLN3,2'] = 2.0
        v_['SOLN3,3'] = 2.6
        for I in range(int(v_['1']),int(v_['M'])+1):
            v_['RI'] = float(I)
            v_['IH'] = v_['H']*v_['RI']
            v_['TI'] = v_['ONE']+v_['IH']
            v_['Differ'] = v_['TI']-v_['SOLN1,2']
            v_['Numer'] = v_['Differ']*v_['Differ']
            v_['Denom'] = v_['SOLN1,3']*v_['SOLN1,3']
            v_['Differ'] = v_['Numer']/v_['Denom']
            v_['Ratio'] = 0.0-v_['Differ']
            v_['ERat'] = jnp.exp(v_['Ratio'])
            v_['Yi1'] = v_['SOLN1,1']*v_['ERat']
            v_['Differ'] = v_['TI']-v_['SOLN2,2']
            v_['Numer'] = v_['Differ']*v_['Differ']
            v_['Denom'] = v_['SOLN2,3']*v_['SOLN2,3']
            v_['Differ'] = v_['Numer']/v_['Denom']
            v_['Ratio'] = 0.0-v_['Differ']
            v_['ERat'] = jnp.exp(v_['Ratio'])
            v_['Yi2'] = v_['SOLN2,1']*v_['ERat']
            v_['Differ'] = v_['TI']-v_['SOLN3,2']
            v_['Numer'] = v_['Differ']*v_['Differ']
            v_['Denom'] = v_['SOLN3,3']*v_['SOLN3,3']
            v_['Differ'] = v_['Numer']/v_['Denom']
            v_['Ratio'] = 0.0-v_['Differ']
            v_['ERat'] = jnp.exp(v_['Ratio'])
            v_['Yi3'] = v_['SOLN3,1']*v_['ERat']
            for p in range(int(v_['1']),int(v_['K'])+1):
                self.gconst  = (                       jtu.arrset(self.gconst,ig_['OBJ'+str(p)+','+str(I)],float(v_['Yi'+str(p)])))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        v_['LOWER1,1'] = 15.0
        v_['LOWER1,2'] = 3.5
        v_['LOWER1,3'] = 0.3
        v_['LOWER2,1'] = 5.0
        v_['LOWER2,2'] = 2.2
        v_['LOWER2,3'] = 2.6
        v_['LOWER3,1'] = 5.0
        v_['LOWER3,2'] = 1.2
        v_['LOWER3,3'] = 1.3
        v_['UPPER1,1'] = 31.0
        v_['UPPER1,2'] = 6.3
        v_['UPPER1,3'] = 3.7
        v_['UPPER2,1'] = 15.0
        v_['UPPER2,2'] = 5.3
        v_['UPPER2,3'] = 6.2
        v_['UPPER3,1'] = 14.0
        v_['UPPER3,2'] = 3.3
        v_['UPPER3,3'] = 2.8
        for p in range(int(v_['1']),int(v_['K'])+1):
            for j in range(int(v_['1']),int(v_['N'])+1):
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(p)+','+str(j)]]), v_['LOWER'+str(p)+','+str(j)])
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(p)+','+str(j)]]), v_['UPPER'+str(p)+','+str(j)])
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        v_['START1,1'] = 25.0
        v_['START1,2'] = 5.2
        v_['START1,3'] = 3.2
        v_['START2,1'] = 7.0
        v_['START2,2'] = 4.1
        v_['START2,3'] = 3.6
        v_['START3,1'] = 11.6
        v_['START3,2'] = 1.9
        v_['START3,3'] = 2.2
        for p in range(int(v_['1']),int(v_['K'])+1):
            for j in range(int(v_['1']),int(v_['N'])+1):
                self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(p)+','+str(j)]]), float(v_['START'+str(p)+','+str(j)]))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eEXPSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'U')
        elftv = jtu.loaset(elftv,it,1,'V')
        elftv = jtu.loaset(elftv,it,2,'W')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'T')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for p in range(int(v_['1']),int(v_['K'])+1):
            for I in range(int(v_['1']),int(v_['M'])+1):
                ename = 'E'+str(p)+','+str(I)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eEXPSQ')
                ielftype = jtu.arrset(ielftype,ie,iet_["eEXPSQ"])
                vname = 'X'+str(p)+','+str(int(v_['1']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(p)+','+str(int(v_['2']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                vname = 'X'+str(p)+','+str(int(v_['3']))
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                v_['RI'] = float(I)
                v_['IH'] = v_['H']*v_['RI']
                v_['TI'] = v_['ONE']+v_['IH']
                posep = jnp.where(elftp[ielftype[ie]]=='T')[0]
                self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['TI']))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for p in range(int(v_['1']),int(v_['K'])+1):
            for I in range(int(v_['1']),int(v_['M'])+1):
                ig = ig_['OBJ'+str(p)+','+str(I)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(p)+','+str(I)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CNOR2-AN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eEXPSQ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        R = (self.elpar[iel_][0]-EV_[1])**2
        S = EV_[2]**2
        E = jnp.exp(-R/S)
        f_   = EV_[0]*E
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, E)
            g_ = jtu.np_like_set(g_, 1, 2.0*(self.elpar[iel_][0]-EV_[1])*EV_[0]*E/S)
            g_ = jtu.np_like_set(g_, 2, 2.0*R*EV_[0]*E/(S*EV_[2]))
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 2.0*(self.elpar[iel_][0]-EV_[1])*E/S)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), 2.0*R*E/(S*EV_[2]))
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), (2.0*EV_[0]*E/S)*(2.0*R/S-1.0))
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), 4.0*(self.elpar[iel_][0]-EV_[1])*EV_[0]*E/(S*EV_[2])*(R/S-1.0))
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), 2.0*R*EV_[0]*E/(S**3)*(2.0*R-3.0*S))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


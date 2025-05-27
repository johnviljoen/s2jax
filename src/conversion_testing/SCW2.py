from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class SCW2:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : SCW2
#    *********
# 
#    Source: a discretization of an infinite-demsional problem proposed 
#    by Simon Chandler-Wilde (U. Reading):
# 
#    Given a function u in C[0,2 pi] with ||u||_infty <= 1, jtu.find the 
#    supremum of c^2(u) + s^2(u), where
#      c(u) = int_0^2 pi cos(t)u(t) dt and
#      s(u) = int_0^2 pi sin(t)u(t) dt      
# 
#    The discretized version ignores the required continuity, and 
#    posits a piecewise constant solution that varies anywhere between
#    plus and minus one. The anticipated solution is -16.
# 
#    SIF input: Nick Gould, July 2020
# 
#    classification = "C-CSLR2-MN-V-V"
# 
#    Number of internal knots
# 
#           Alternative values for the SIF file parameters:
# IE K                   1              $-PARAMETER
# IE K                   10             $-PARAMETER
# IE K                   100            $-PARAMETER
# IE K                   1000           $-PARAMETER     original value
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'SCW2'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['K'] = int(10);  #  SIF file default value
        else:
            v_['K'] = int(args[0])
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['ONE'] = 1.0
        v_['K+1'] = 1+v_['K']
        v_['RK'] = float(v_['K'])
        v_['RK+1'] = float(v_['K+1'])
        v_['PI/4'] = jnp.arctan(1.0)
        v_['PI'] = 4.0*v_['PI/4']
        v_['2PI'] = 2.0*v_['PI']
        v_['2PI/K+1'] = v_['2PI']/v_['RK+1']
        v_['1/K'] = v_['ONE']/v_['RK']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['0']),int(v_['K+1'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('T'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'T'+str(I))
        for I in range(int(v_['0']),int(v_['K'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('U'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'U'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('S',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('C',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for I in range(int(v_['0']),int(v_['K'])+1):
            v_['I+1'] = 1+I
            [ig,ig_,_] = jtu.s2mpj_ii('CON'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'CON'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(int(v_['I+1']))]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
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
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['T'+str(int(v_['0']))], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['T'+str(int(v_['0']))], 0.0)
        for I in range(int(v_['1']),int(v_['K'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['T'+str(I)], 0.0)
            self.xupper = jtu.np_like_set(self.xupper, ix_['T'+str(I)], v_['2PI'])
        self.xlower = jtu.np_like_set(self.xlower, ix_['T'+str(int(v_['K+1']))], v_['2PI'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['T'+str(int(v_['K+1']))], v_['2PI'])
        for I in range(int(v_['0']),int(v_['K'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['U'+str(I)], -1.0)
            self.xupper = jtu.np_like_set(self.xupper, ix_['U'+str(I)], 1.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        if('T'+str(int(v_['1'])) in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['T'+str(int(v_['1']))], float(1.0))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['T'+str(int(v_['1']))]),float(1.0)))
        for I in range(int(v_['2']),int(v_['K'])+1):
            v_['RI'] = float(I)
            v_['START'] = v_['RI']*v_['2PI/K+1']
            if('T'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['T'+str(I)], float(0.0))
            else:
                self.y0  = (
                      jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['T'+str(I)]),float(0.0)))
        for I in range(int(v_['0']),int(v_['K'])+1):
            v_['RI'] = float(I)
            v_['START'] = v_['RI']*v_['1/K']
            if('U'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['U'+str(I)], float(v_['START']))
            else:
                self.y0  = (
                      jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['U'+str(I)]),float(v_['START'])))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eUSINT', iet_)
        elftv = jtu.loaset(elftv,it,0,'T')
        elftv = jtu.loaset(elftv,it,1,'U')
        [it,iet_,_] = jtu.s2mpj_ii( 'eUCOST', iet_)
        elftv = jtu.loaset(elftv,it,0,'T')
        elftv = jtu.loaset(elftv,it,1,'U')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['0']),int(v_['K'])+1):
            v_['I+1'] = 1+I
            ename = 'US'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eUSINT')
            ielftype = jtu.arrset(ielftype,ie,iet_["eUSINT"])
            vname = 'T'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='T')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'U'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'USP'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eUSINT')
            ielftype = jtu.arrset(ielftype,ie,iet_["eUSINT"])
            vname = 'T'+str(int(v_['I+1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='T')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'U'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'UC'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eUCOST')
            ielftype = jtu.arrset(ielftype,ie,iet_["eUCOST"])
            vname = 'T'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='T')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'U'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'UCP'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eUCOST')
            ielftype = jtu.arrset(ielftype,ie,iet_["eUCOST"])
            vname = 'T'+str(int(v_['I+1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='T')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'U'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gMAXSQ',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['C']
        self.grftype = jtu.arrset(self.grftype,ig,'gMAXSQ')
        for I in range(int(v_['0']),int(v_['K'])+1):
            ig = ig_['C']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['USP'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['US'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['S']
        self.grftype = jtu.arrset(self.grftype,ig,'gMAXSQ')
        for I in range(int(v_['0']),int(v_['K'])+1):
            ig = ig_['S']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['UCP'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['UC'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
# LO SCW                 0.0
#    Solution
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
        self.pbclass   = "C-CSLR2-MN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eUSINT(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        S = jnp.sin(EV_[0])
        C = jnp.cos(EV_[0])
        f_   = EV_[1]*S
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[1]*C)
            g_ = jtu.np_like_set(g_, 1, S)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -EV_[1]*S)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), C)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eUCOST(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        S = jnp.sin(EV_[0])
        C = jnp.cos(EV_[0])
        f_   = EV_[1]*C
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -EV_[1]*S)
            g_ = jtu.np_like_set(g_, 1, C)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -EV_[1]*C)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), -S)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gMAXSQ(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= -GVAR_*GVAR_
        if nargout>1:
            g_ = -GVAR_-GVAR_
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = -2.0e+0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


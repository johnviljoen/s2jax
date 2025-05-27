from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class DIXCHLNV:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : DIXCHLNV
#    *********
# 
#    A variant of a constrained problem set as a challenge for SQP methods
#    by L.C.W. Dixon at the APMOD91 Conference.
#    The variant from the original problem is that the variables have
#    been constrained to be above 1.0D-15, which in turn allows the
#    constraints to be expressed in terms of logarithms.
# 
#    Source: a modification (by Ph. Toint) of
#    L.C.W. Dixon, personnal communication, Jan 1991.
# 
#    SIF input: Ph. Toint, Feb 1991.
# 
#    classification = "C-CSOR2-AN-V-V"
# 
#    Number of variables
#    (variable, but must be even and at least equal to 4)
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER
# IE N                   50             $-PARAMETER
# IE N                   100            $-PARAMETER     original value
# IE N                   1000           $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'DIXCHLNV'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(10);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
# IE N                   5000           $-PARAMETER
        v_['1'] = 1
        v_['2'] = 2
        v_['N-3'] = -3+v_['N']
        v_['N-1'] = -1+v_['N']
        v_['90.0'] = 90.0
        v_['10.1'] = 10.1
        v_['19.8'] = 19.8
        v_['1/90.0'] = 1.0/v_['90.0']
        v_['1/10.1'] = 1.0/v_['10.1']
        v_['1/19.8'] = 1.0/v_['19.8']
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
        for I in range(int(v_['1']),int(v_['N-3'])+1):
            v_['I+1'] = 1+I
            v_['I+2'] = 2+I
            v_['I+3'] = 3+I
            [ig,ig_,_] = jtu.s2mpj_ii('A'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['I+1']))]])
            valA = jtu.append(valA,float(1.0))
            self.gscale = jtu.arrset(self.gscale,ig,float(0.01))
            [ig,ig_,_] = jtu.s2mpj_ii('B'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['I+3']))]])
            valA = jtu.append(valA,float(1.0))
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['1/90.0']))
            [ig,ig_,_] = jtu.s2mpj_ii('D'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['I+2']))]])
            valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('E'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['I+1']))]])
            valA = jtu.append(valA,float(1.0))
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['1/10.1']))
            [ig,ig_,_] = jtu.s2mpj_ii('F'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['I+3']))]])
            valA = jtu.append(valA,float(1.0))
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['1/10.1']))
            [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['1/19.8']))
        for I in range(int(v_['2']),int(v_['N'])+1,int(v_['2'])):
            [ig,ig_,_] = jtu.s2mpj_ii('P'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'P'+str(I))
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
        for I in range(int(v_['1']),int(v_['N-3'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['B'+str(I)],float(1.0))
            self.gconst = jtu.arrset(self.gconst,ig_['D'+str(I)],float(1.0))
            self.gconst = jtu.arrset(self.gconst,ig_['E'+str(I)],float(1.0))
            self.gconst = jtu.arrset(self.gconst,ig_['F'+str(I)],float(1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),1.0e-8)
        self.xupper = jnp.full((self.n,1),+float('inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        v_['X0'] = 2.0
        for I in range(int(v_['1']),int(v_['N-1'])+1,int(v_['2'])):
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(v_['X0']))
            v_['1/X0'] = 1.0/v_['X0']
            v_['I+1'] = 1+I
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(int(v_['I+1']))], float(v_['1/X0']))
            v_['X0'] = 1.0+v_['X0']
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        [it,iet_,_] = jtu.s2mpj_ii( 'eS2PR', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        elftv = jtu.loaset(elftv,it,1,'W')
        [it,iet_,_] = jtu.s2mpj_ii( 'eLOGV', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            ename = 'XSQ'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(1.0e-8),None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['N-3'])+1):
            v_['I+1'] = 1+I
            v_['I+3'] = 3+I
            ename = 'PR'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eS2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["eS2PR"])
            vname = 'X'+str(int(v_['I+1']))
            [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(1.0e-8),None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['I+3']))
            [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(1.0e-8),None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for I in range(int(v_['1']),int(v_['N'])+1):
            ename = 'LOGX'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eLOGV')
            ielftype = jtu.arrset(ielftype,ie,iet_["eLOGV"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self, vname,ix_,1,float(1.0e-8),None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gL2',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N-3'])+1):
            v_['I+2'] = 2+I
            ig = ig_['A'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
            ig = ig_['B'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            ig = ig_['C'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(int(v_['I+2']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
            ig = ig_['D'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            ig = ig_['E'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            ig = ig_['F'+str(I)]
            self.grftype = jtu.arrset(self.grftype,ig,'gL2')
            ig = ig_['G'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['PR'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        for I in range(int(v_['2']),int(v_['N'])+1,int(v_['2'])):
            for J in range(int(v_['1']),int(I)+1):
                ig = ig_['P'+str(I)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['LOGX'+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               0.0
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
        self.pbclass   = "C-CSOR2-AN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eSQ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[0]+EV_[0])
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eLOGV(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = jnp.log(EV_[0])
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 1.0/EV_[0])
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -1.0/(EV_[0]*EV_[0]))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eS2PR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = (EV_[0]-1.0)*(EV_[1]-1.0)
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[1]-1.0)
            g_ = jtu.np_like_set(g_, 1, EV_[0]-1.0)
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

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gL2(self,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= GVAR_*GVAR_
        if nargout>1:
            g_ = GVAR_+GVAR_
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = 2.0
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


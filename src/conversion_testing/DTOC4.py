import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class DTOC4:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : DTOC4
#    *********
# 
#    This is a discrete time optimal control (DTOC) problem.  
#    The system has N time periods, 1 control variable and 2 state variables.
# 
#    The problem is not convex.
# 
#    Sources: problem 4 in
#    T.F. Coleman and A. Liao,
#    "An Efficient Trust Region Method for Unconstrained Discret-Time Optimal
#    Control Problems",
#    Tech. Report, ctc93tr144,  Advanced Computing Research Institute, 
#    Cornell University, 1992.
# 
#    G. Di Pillo, L. Grippo and F. Lampariello,
#    "A class of structures quasi-Newton algorithms for optimal control
#    problems",
#    in H.E. Rauch, ed., IFAC Applications of nonlinear programming to
#    optimization and control, pp. 101-107, IFAC, Pergamon Press, 1983.
# 
#    SIF input: Ph. Toint, August 1993
# 
#    classification = "C-CQOR2-AN-V-V"
# 
#    Problem variants: they are identified by the value of the parameter N.
# 
#    The problem has 3N-1  variables (of which 2 are fixed),
#    and 2(N-1) constraints
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER  n=   29,m= 18 original value
# IE N                   50             $-PARAMETER  n=  149,m= 98
# IE N                   100            $-PARAMETER  n=  299,m=198
# IE N                   500            $-PARAMETER  n= 1499,m=998
# IE N                   1000           $-PARAMETER  n= 2999,m=1998
# IE N                   1500           $-PARAMETER  n= 4499,m=2998
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'DTOC4'

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
# IE N                   5000           $-PARAMETER  n=14999,m=9998
        v_['N-1'] = -1+v_['N']
        v_['1'] = 1
        v_['2'] = 2
        v_['RN'] = float(v_['N'])
        v_['H'] = 1.0/v_['RN']
        v_['5H'] = 5.0*v_['H']
        v_['1/5H'] = 1.0/v_['5H']
        v_['1+5H'] = 1.0+v_['5H']
        v_['-5H'] = -1.0*v_['5H']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for T in range(int(v_['1']),int(v_['N-1'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(T),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(T))
        for T in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('Y'+str(T)+','+str(int(v_['1'])),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'Y'+str(T)+','+str(int(v_['1'])))
            [iv,ix_,_] = jtu.s2mpj_ii('Y'+str(T)+','+str(int(v_['2'])),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'Y'+str(T)+','+str(int(v_['2'])))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        self.gscale = jtu.arrset(self.gscale,ig,float(v_['1/5H']))
        for T in range(int(v_['1']),int(v_['N-1'])+1):
            v_['T+1'] = 1+T
            [ig,ig_,_] = jtu.s2mpj_ii('TT'+str(T)+','+str(int(v_['1'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'TT'+str(T)+','+str(int(v_['1'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Y'+str(int(v_['T+1']))+','+str(int(v_['1']))]])
            valA = jtu.append(valA,float(-1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('TT'+str(T)+','+str(int(v_['1'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'TT'+str(T)+','+str(int(v_['1'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Y'+str(T)+','+str(int(v_['1']))]])
            valA = jtu.append(valA,float(v_['1+5H']))
            [ig,ig_,_] = jtu.s2mpj_ii('TT'+str(T)+','+str(int(v_['1'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'TT'+str(T)+','+str(int(v_['1'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Y'+str(T)+','+str(int(v_['2']))]])
            valA = jtu.append(valA,float(v_['-5H']))
            [ig,ig_,_] = jtu.s2mpj_ii('TT'+str(T)+','+str(int(v_['1'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'TT'+str(T)+','+str(int(v_['1'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(T)]])
            valA = jtu.append(valA,float(v_['5H']))
            [ig,ig_,_] = jtu.s2mpj_ii('TT'+str(T)+','+str(int(v_['2'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'TT'+str(T)+','+str(int(v_['2'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Y'+str(int(v_['T+1']))+','+str(int(v_['2']))]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Y'+str(T)+','+str(int(v_['2']))]])
            valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('TT'+str(T)+','+str(int(v_['2'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'TT'+str(T)+','+str(int(v_['2'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Y'+str(T)+','+str(int(v_['1']))]])
            valA = jtu.append(valA,float(v_['5H']))
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
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['1']))]]), 0.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['1']))]]), 0.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['2']))]]), 1.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['2']))]]), 1.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['1']))]]), float(0.0))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['2']))]]), float(1.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'Z')
        [it,iet_,_] = jtu.s2mpj_ii( 'eAAB', iet_)
        elftv = jtu.loaset(elftv,it,0,'A')
        elftv = jtu.loaset(elftv,it,1,'B')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        ename = 'Y1SQ'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
        ename = 'Y1SQ'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        vname = 'Y'+str(int(v_['1']))+','+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'Y2SQ'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
        ename = 'Y2SQ'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        vname = 'Y'+str(int(v_['1']))+','+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'XSQ'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
        ename = 'XSQ'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for T in range(int(v_['2']),int(v_['N-1'])+1):
            ename = 'Y1SQ'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'Y'+str(T)+','+str(int(v_['1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'Y2SQ'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'Y'+str(T)+','+str(int(v_['2']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'XSQ'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'X'+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'Y1SQ'+str(int(v_['N']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
        ename = 'Y1SQ'+str(int(v_['N']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        vname = 'Y'+str(int(v_['N']))+','+str(int(v_['1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'Y2SQ'+str(int(v_['N']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
        ename = 'Y2SQ'+str(int(v_['N']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        vname = 'Y'+str(int(v_['N']))+','+str(int(v_['2']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for T in range(int(v_['1']),int(v_['N-1'])+1):
            ename = 'E'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eAAB')
            ielftype = jtu.arrset(ielftype,ie,iet_["eAAB"])
            vname = 'Y'+str(T)+','+str(int(v_['2']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='A')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'Y'+str(T)+','+str(int(v_['1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='B')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['OBJ']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['Y1SQ'+str(int(v_['1']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(0.5))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['Y2SQ'+str(int(v_['1']))])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(0.5))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(int(v_['1']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
        for T in range(int(v_['2']),int(v_['N-1'])+1):
            ig = ig_['OBJ']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['Y1SQ'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['Y2SQ'+str(T)])
            self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['OBJ']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['Y1SQ'+str(int(v_['N']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(0.5))
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['Y2SQ'+str(int(v_['N']))])
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(0.5))
        for T in range(int(v_['1']),int(v_['N-1'])+1):
            ig = ig_['TT'+str(T)+','+str(int(v_['1']))]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-5H']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
# LO SOLUTION(  10)      3.75078392210
# LO SOLUTION(  50)      3.02963141755
# LO SOLUTION( 100)      2.94726711402
# LO SOLUTION( 500)      2.87827434035
# LO SOLUTION(1000)      2.87483889886
# LO SOLUTION(5000)      2.86386891514
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
        self.pbclass   = "C-CQOR2-AN-V-V"
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
            f_   = f_.item()
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
    def eAAB(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 2.0*EV_[0]*EV_[1])
            g_ = jtu.np_like_set(g_, 1, EV_[0]*EV_[0])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), EV_[1]+EV_[1])
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), EV_[0]+EV_[0])
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


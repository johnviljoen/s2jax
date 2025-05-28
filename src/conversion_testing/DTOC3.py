import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class DTOC3:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : DTOC3
#    *********
# 
#    This is a discrete time optimal control (DTOC) problem.  
#    The system has N time periods, 1 control variable and 2 state variables.
# 
#    The problem is convex.
# 
#    Sources: problem 3 in
#    T.F. Coleman and A. Liao,
#    "An Efficient Trust Region Method for Unconstrained Discret-Time Optimal
#    Control Problems",
#    Tech. Report, ctc93tr144,  Advanced Computing Research Institute, 
#    Cornell University, 1992.
# 
#    D.P. Bertsekas,
#    "Projected Newton methods for optimization problems with simple
#    constraints", 
#    SIAM J. Control and Optimization 20, pp. 221-246, 1982.
# 
#    SIF input: Ph. Toint, August 1993
# 
#    classification = "C-CQLR2-AN-V-V"
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

    name = 'DTOC3'

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
        v_['S'] = 1.0/v_['RN']
        v_['2/S'] = 2.0/v_['S']
        v_['-S'] = -1.0*v_['S']
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
            for I in range(int(v_['1']),int(v_['2'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('Y'+str(T)+','+str(I),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'Y'+str(T)+','+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for T in range(int(v_['1']),int(v_['N-1'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('O'+str(T),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['2/S']))
        for T in range(int(v_['1']),int(v_['N-1'])+1):
            v_['T+1'] = 1+T
            [ig,ig_,_] = jtu.s2mpj_ii('TT'+str(T)+','+str(int(v_['1'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'TT'+str(T)+','+str(int(v_['1'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Y'+str(int(v_['T+1']))+','+str(int(v_['1']))]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Y'+str(T)+','+str(int(v_['1']))]])
            valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('TT'+str(T)+','+str(int(v_['1'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'TT'+str(T)+','+str(int(v_['1'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Y'+str(T)+','+str(int(v_['2']))]])
            valA = jtu.append(valA,float(v_['S']))
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
            valA = jtu.append(valA,float(v_['-S']))
            [ig,ig_,_] = jtu.s2mpj_ii('TT'+str(T)+','+str(int(v_['2'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'TT'+str(T)+','+str(int(v_['2'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(T)]])
            valA = jtu.append(valA,float(v_['S']))
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
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['1']))]]), 15.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['1']))]]), 15.0)
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['2']))]]), 5.0)
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['2']))]]), 5.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['1']))]]), float(15.0))
        self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['Y'+str(int(v_['1']))+','+str(int(v_['2']))]]), float(5.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'YY')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for T in range(int(v_['2']),int(v_['N'])+1):
            ename = 'Y1SQ'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'Y'+str(T)+','+str(int(v_['1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='YY')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'Y2SQ'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'Y'+str(T)+','+str(int(v_['2']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='YY')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        for T in range(int(v_['1']),int(v_['N-1'])+1):
            ename = 'XSQ'+str(T)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'X'+str(T)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='YY')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for T in range(int(v_['1']),int(v_['N-1'])+1):
            v_['T+1'] = 1+T
            ig = ig_['O'+str(T)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['Y1SQ'+str(int(v_['T+1']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(2.0))
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['Y2SQ'+str(int(v_['T+1']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(1.0))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(T)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(6.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
# LO SOLUTION(  10)      224.590381002
# LO SOLUTION(  50)      233.278523083
# LO SOLUTION( 100)      234.286202920
# LO SOLUTION( 500)      235.084407947
# LO SOLUTION(1000)      235.182824435
# LO SOLUTION(5000)      235.154640099
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
        self.pbclass   = "C-CQLR2-AN-V-V"
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


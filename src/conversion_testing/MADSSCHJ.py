import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class MADSSCHJ:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : MADSSCHJ
#    *********
# 
#    A nonlinear minmax problem with variable dimension.
#    The Jacobian of the constraints is dense.
# 
#    Source:
#    K. Madsen and H. Schjaer-Jacobsen,
#    "Linearly Constrained Minmax Optimization",
#    Mathematical Programming 14, pp. 208-223, 1978.
# 
#    SIF input: Ph. Toint, August 1993.
# 
#    classification = "C-CLQR2-AN-V-V"
# 
#    N is the number of variables - 1, and must be even and at least 4.
#    The number of inequality constraints is 2*N - 2.
# 
#           Alternative values for the SIF file parameters:
# IE N                   4              $-PARAMETER  n=  5, m=  6
# IE N                   10             $-PARAMETER  n= 11, m= 18  original value
# IE N                   20             $-PARAMETER  n= 21, m= 38
# IE N                   30             $-PARAMETER  n= 31, m= 58
# IE N                   40             $-PARAMETER  n= 41, m= 78
# IE N                   50             $-PARAMETER  n= 51, m= 98
# IE N                   60             $-PARAMETER  n= 61, m=118
# IE N                   70             $-PARAMETER  n= 71, m=138
# IE N                   80             $-PARAMETER  n= 81, m=158
# IE N                   90             $-PARAMETER  n= 91, m=178
# IE N                   100            $-PARAMETER  n=101, m=198
# IE N                   200            $-PARAMETER  n=201, m=398
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'MADSSCHJ'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(4);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['4'] = 4
        v_['N-1'] = -1+v_['N']
        v_['2N'] = v_['N']+v_['N']
        v_['M'] = -2+v_['2N']
        v_['M-1'] = -1+v_['M']
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
        [iv,ix_,_] = jtu.s2mpj_ii('Z',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Z')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Z']])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C1',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Z']])
        valA = jtu.append(valA,float(1.0))
        for I in range(int(v_['2']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('C1',ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'C1')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C2',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Z']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['3']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('C2',ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'C2')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C3',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Z']])
        valA = jtu.append(valA,float(1.0))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['3']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('C3',ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'C3')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        for K in range(int(v_['4']),int(v_['M-1'])+1,int(v_['2'])):
            v_['K+1'] = 1+K
            v_['K+2'] = 2+K
            v_['J'] = int(jnp.fix(v_['K+2']/v_['2']))
            v_['J-1'] = -1+v_['J']
            v_['J+1'] = 1+v_['J']
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(K),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'C'+str(K))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Z']])
            valA = jtu.append(valA,float(1.0))
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(int(v_['K+1'])),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'C'+str(int(v_['K+1'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Z']])
            valA = jtu.append(valA,float(1.0))
            for I in range(int(v_['1']),int(v_['J-1'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('C'+str(K),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'C'+str(K))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)]])
                valA = jtu.append(valA,float(-1.0))
                [ig,ig_,_] = jtu.s2mpj_ii('C'+str(int(v_['K+1'])),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'C'+str(int(v_['K+1'])))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)]])
                valA = jtu.append(valA,float(-1.0))
            for I in range(int(v_['J+1']),int(v_['N'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('C'+str(K),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'C'+str(K))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)]])
                valA = jtu.append(valA,float(-1.0))
                [ig,ig_,_] = jtu.s2mpj_ii('C'+str(int(v_['K+1'])),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'C'+str(int(v_['K+1'])))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(I)]])
                valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('C'+str(int(v_['M'])),ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'C'+str(int(v_['M'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Z']])
        valA = jtu.append(valA,float(1.0))
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(int(v_['M'])),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'C'+str(int(v_['M'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
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
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        for K in range(int(v_['1']),int(v_['M'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['C'+str(K)],float(-1.0))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(10.0))
        self.x0 = jtu.np_like_set(self.x0, ix_['Z'], float(0.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            ename = 'XSQ'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(10.0))
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['C'+str(int(v_['1']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(int(v_['1']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['C'+str(int(v_['2']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(int(v_['2']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['C'+str(int(v_['3']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(int(v_['2']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-2.0))
        for K in range(int(v_['4']),int(v_['M-1'])+1,int(v_['2'])):
            v_['K+1'] = 1+K
            v_['K+2'] = 2+K
            v_['J'] = int(jnp.fix(v_['K+2']/v_['2']))
            ig = ig_['C'+str(K)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(int(v_['J']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
            ig = ig_['C'+str(int(v_['K+1']))]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(int(v_['J']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-2.0))
        ig = ig_['C'+str(int(v_['M']))]
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['XSQ'+str(int(v_['N']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN(4)             -2.6121094144
# LO SOLTN(10)            -12.814452425
# LO SOLTN(20)            -49.869888156
# LO SOLTN(30)            -111.93545559
# LO SOLTN(40)            -199.00371592
# LO SOLTN(50)            -311.07308068
# LO SOLTN(60)            -448.14300524
# LO SOLTN(70)            -610.21325256
# LO SOLTN(80)            -797.28370289
# LO SOLTN(90)            -1009.3542892
# LO SOLTN(100)           -1246.4249710
# LO SOLTN(200)           -4992.1339031
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CLQR2-AN-V-V"
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


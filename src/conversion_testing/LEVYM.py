import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LEVYM:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
#    Problem : LEVYM
#    *********
#    A global optimization example due to Levy & Montalvo 
#    This problem is one of the parameterised set LEVYMONT5-LEVYMONT10
#    f(x_1,...,x_n) = \sum_{i=1}^n (pi/n)*(L*x_i)^2 +
#                     ( K*pi/n) * [ s(x1,L,C)^2 + \sum_{i=2}^n  p(x_{i-1},x_i,L,C,A)^2 ]
#    where
#    s(x,L,C )     = sin( pi( L*x + C ) )
#    p(x,y,L,C,A ) = ( L* y + C - A) * sin( pi( L*x + C ) )
# 
#    Source:  problem 8 in
# 
#    A. V. Levy and A. Montalvo
#    "The Tunneling Algorithm for the Global Minimization of Functions"
#    SIAM J. Sci. Stat. Comp. 6(1) 1985 15:29 
#    https://doi.org/10.1137/0906002
# 
#    SIF input: Nick Gould, August 2021
# 
#    classification = "SBR2-AY-5-0"
# 
#    N is the number of variables
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LEVYM'

    def __init__(self, *args): 
        import jax.numpy as jnp
        pbm      = jtu.structtype()
        pb       = jtu.structtype()
        pb.name  = self.name
        pb.sifpbname = 'LEVYM'
        pbm.name = self.name
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(5);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
        v_['A'] = 1.0
        v_['K'] = 10.0
        v_['L'] = 1.0
        v_['C'] = 0.0
        v_['1'] = 1
        v_['2'] = 2
        v_['PI/4'] = jnp.arctan(1.0)
        v_['PI'] = 4.0*v_['PI/4']
        v_['RN'] = float(v_['N'])
        v_['A-C'] = v_['A']-v_['C']
        v_['PI/N'] = v_['PI']/v_['RN']
        v_['KPI/N'] = v_['K']*v_['PI/N']
        v_['ROOTKPI/N'] = jnp.sqrt(v_['KPI/N'])
        v_['N/PI'] = v_['RN']/v_['PI']
        v_['N/KPI'] = v_['N/PI']/v_['K']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        pb.xnames = jnp.array([])
        xscale    = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            pb.xnames=jtu.arrset(pb.xnames,iv,'X'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        pbm.A       = spu.bcoo_zeros([1000000, 1000000], dtype=jnp.int64)
        pbm.gscale  = jnp.array([])
        pbm.grnames = jnp.array([])
        cnames      = jnp.array([])
        pb.cnames   = jnp.array([])
        gtype       = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('Q'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            iv = ix_['X'+str(I)]
            pbm.A = jtu.np_like_set(pbm.A, jnp.array([ig,iv]), float(v_['L'])+pbm.A[ig,iv])
            pbm.gscale = jtu.arrset(pbm.gscale,ig,float(v_['N/PI']))
            [ig,ig_,_] = jtu.s2mpj_ii('N'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
        #%%%%%%%%%%%%%% GLOBAL DIMENSIONS %%%%%%%%%%%%%%%%%
        pb.n   = len(ix_)
        ngrp   = len(ig_)
        pbm.objgrps = jnp.arange(ngrp)
        pb.m        = 0
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        pbm.gconst = jnp.zeros((ngrp,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            pbm.gconst = jtu.arrset(pbm.gconst,ig_['Q'+str(I)],v_['A-C'])
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        pb.xlower = jnp.full((pb.n,1),-10.0)
        pb.xupper = jnp.full((pb.n,1),10.0)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        pb.x0 = jnp.full((pb.n,1),float(8.0))
        pb.x0 = jtu.np_like_set(pb.x0, ix_['X1'], float(-8.0))
        pb.x0 = jtu.np_like_set(pb.x0, ix_['X2'], float(8.0))
        pass
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eS2', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'L')
        elftp = jtu.loaset(elftp,it,1,'C')
        [it,iet_,_] = jtu.s2mpj_ii( 'ePS2', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Z')
        elftp = jtu.loaset(elftp,it,0,'L')
        elftp = jtu.loaset(elftp,it,1,'C')
        elftp = jtu.loaset(elftp,it,2,'A')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        pbm.elftype = jnp.array([])
        ielftype    = jnp.array([])
        pbm.elvar   = []
        pbm.elpar   = []
        ename = 'E'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        pbm.elftype = jtu.arrset(pbm.elftype,ie,'eS2')
        ielftype = jtu.arrset(ielftype, ie, iet_["eS2"])
        ename = 'E'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        vname = 'X'+str(int(v_['1']))
        [iv,ix_,pb] = jtu.s2mpj_nlx(vname,ix_,pb,1,-10.0,10.0,8.0)
        posev = jtu.find(elftv[ielftype[ie]],lambda x:x=='X')
        pbm.elvar = jtu.loaset(pbm.elvar,ie,posev[0],iv)
        ename = 'E'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        posep = jtu.find(elftp[ielftype[ie]],lambda x:x=='L')
        pbm.elpar = jtu.loaset(pbm.elpar,ie,posep[0],float(v_['L']))
        ename = 'E'+str(int(v_['1']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        posep = jtu.find(elftp[ielftype[ie]],lambda x:x=='C')
        pbm.elpar = jtu.loaset(pbm.elpar,ie,posep[0],float(v_['C']))
        for I in range(int(v_['2']),int(v_['N'])+1):
            v_['I-1'] = I-v_['1']
            ename = 'E'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            pbm.elftype = jtu.arrset(pbm.elftype,ie,'ePS2')
            ielftype = jtu.arrset(ielftype, ie, iet_["ePS2"])
            vname = 'X'+str(I)
            [iv,ix_,pb] = jtu.s2mpj_nlx(vname,ix_,pb,1,-10.0,10.0,8.0)
            posev = jtu.find(elftv[ielftype[ie]],lambda x:x=='X')
            pbm.elvar = jtu.loaset(pbm.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['I-1']))
            [iv,ix_,pb] = jtu.s2mpj_nlx(vname,ix_,pb,1,-10.0,10.0,8.0)
            posev = jtu.find(elftv[ielftype[ie]],lambda x:x=='Z')
            pbm.elvar = jtu.loaset(pbm.elvar,ie,posev[0],iv)
            posep = jtu.find(elftp[ielftype[ie]],lambda x:x=='L')
            pbm.elpar = jtu.loaset(pbm.elpar,ie,posep[0],float(v_['L']))
            posep = jtu.find(elftp[ielftype[ie]],lambda x:x=='C')
            pbm.elpar = jtu.loaset(pbm.elpar,ie,posep[0],float(v_['C']))
            posep = jtu.find(elftp[ielftype[ie]],lambda x:x=='A')
            pbm.elpar = jtu.loaset(pbm.elpar,ie,posep[0],float(v_['A']))
        #%%%%%%%%%%%%%%%%%%%%% GRFTYPE %%%%%%%%%%%%%%%%%%%%
        igt_ = {}
        [it,igt_,_] = jtu.s2mpj_ii('gL2',igt_)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        pbm.grelt   = []
        for ig in jnp.arange(0,ngrp):
            pbm.grelt.append(jnp.array([]))
        pbm.grftype = jnp.array([])
        pbm.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['Q'+str(I)]
            pbm.grftype = jtu.arrset(pbm.grftype,ig,'gL2')
            ig = ig_['N'+str(I)]
            pbm.grftype = jtu.arrset(pbm.grftype,ig,'gL2')
            posel = len(pbm.grelt[ig])
            pbm.grelt = jtu.loaset(pbm.grelt,ig,posel,ie_['E'+str(I)])
            pbm.grelw = jtu.loaset(pbm.grelw,ig,posel,float(v_['ROOTKPI/N']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        pb.objlower = 0.0
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%%%%%%  RESIZE A %%%%%%%%%%%%%%%%%%%%%%
        pbm.A.resize(ngrp,pb.n)
        pbm.A      = pbm.A.tocsr()
        sA1,sA2    = pbm.A.shape
        pbm.Ashape = [ sA1, sA2 ]
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        pb.pbclass = "SBR2-AY-5-0"
        self.pb = pb; self.pbm = pbm

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def e_globs(pbm):

        import jax.numpy as jnp
        pbm.efpar = jnp.array([])
        pbm.efpar = jtu.arrset( pbm.efpar,0,4.0*jnp.arctan(1.0e0))
        return pbm

    @staticmethod
    def eS2(pbm,nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        PIL = pbm.efpar[0]*pbm.elpar[iel_][0]
        V = PIL*EV_[0]+pbm.efpar[0]*pbm.elpar[iel_][1]
        SINV = jnp.sin(V)
        COSV = jnp.cos(V)
        f_   = SINV
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, PIL*COSV)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -PIL*PIL*SINV)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def ePS2(pbm,nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        PIL = pbm.efpar[0]*pbm.elpar[iel_][0]
        U = pbm.elpar[iel_][0]*EV_[1]+pbm.elpar[iel_][1]-pbm.elpar[iel_][2]
        V = PIL*EV_[0]+pbm.efpar[0]*pbm.elpar[iel_][1]
        SINV = jnp.sin(V)
        COSV = jnp.cos(V)
        f_   = U*SINV
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, PIL*U*COSV)
            g_ = jtu.np_like_set(g_, 1, pbm.elpar[iel_][0]*SINV)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -PIL*PIL*U*SINV)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), pbm.elpar[iel_][0]*PIL*COSV)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), 0.0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    #%%%%%%%%%%%%%%%%% NONLINEAR GROUPS  %%%%%%%%%%%%%%%

    @staticmethod
    def gL2(pbm,nargout,*args):

        GVAR_ = args[0]
        igr_  = args[1]
        f_= GVAR_*GVAR_
        if nargout>1:
            g_ = 2.0*GVAR_
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


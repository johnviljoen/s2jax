from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class HS88:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : HS88
#    *********
# 
#    A time-optimal heat conduction problem.
# 
#    Source: problem 88 in
#    W. Hock and K. Schittkowski,
#    "Test examples for nonlinear programming codes",
#    Lectures Notes in Economics and Mathematical Systems 187, Springer
#    Verlag, Heidelberg, 1981.
# 
#    SIF input: Nick Gould, September 1991.
#      SAVEs removed December 3rd 2014
# 
#    classification = "C-CQOR2-MN-2-1"
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'HS88'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 2
        v_['EPS'] = 0.01
        v_['EPSSQR'] = v_['EPS']*v_['EPS']
        v_['-EPSSQR'] = -1.0*v_['EPSSQR']
        v_['1'] = 1
        v_['2'] = 2
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
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        [ig,ig_,_] = jtu.s2mpj_ii('CON',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'CON')
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
        self.gconst = jtu.arrset(self.gconst,ig_['CON'],float(v_['-EPSSQR']))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['N'])+1,int(v_['2'])):
            if('X'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(0.5))
            else:
                self.y0  = (
                      jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X'+str(I)]),float(0.5)))
        for I in range(int(v_['2']),int(v_['N'])+1,int(v_['2'])):
            if('X'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(-0.5))
            else:
                self.y0  = (
                      jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X'+str(I)]),float(-0.5)))
        pass
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQR', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        [it,iet_,_] = jtu.s2mpj_ii( 'eH', iet_)
        elftv = jtu.loaset(elftv,it,0,'X1')
        elftv = jtu.loaset(elftv,it,1,'X2')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            ename = 'O'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQR')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQR"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'H'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eH')
        ielftype = jtu.arrset(ielftype,ie,iet_["eH"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X1')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X2')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            ig = ig_['OBJ']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['O'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['CON']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['H'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CQOR2-MN-5-1"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%
    
    @staticmethod
    def extfunc(self,x):
        # A translation of the Fortran code present in the SIF file.
        import jax.numpy as jnp
        n     = len(x)
        g     = jnp.zeros((n,1))
        H     = jnp.zeros((n,n))
        A     = jnp.zeros((30,1))
        R     = jnp.zeros((30,30))
        S     = jnp.zeros((30,1))
        RHO   = jnp.zeros((30,1))
        DRHO  = jnp.zeros((30,n))
        D2RHO = jnp.zeros((30,n,n))
        P     = jnp.zeros((n+1,1))
        
        mu = jnp.array([ 8.6033358901938017e-01, 3.4256184594817283e+00, 6.4372981791719468e+00, 9.5293344053619631e+00,
                        1.2645287223856643e+01, 1.5771284874815882e+01, 1.8902409956860023e+01, 2.2036496727938566e+01, 
                        2.5172446326646664e+01, 2.8309642854452012e+01, 3.1447714637546234e+01, 3.4586424215288922e+01, 
                        3.7725612827776501e+01, 4.0865170330488070e+01, 4.4005017920830845e+01, 4.7145097736761031e+01, 
                        5.0285366337773652e+01, 5.3425790477394663e+01, 5.6566344279821521e+01, 5.9707007305335459e+01, 
                        6.2847763194454451e+01, 6.5988598698490392e+01, 6.9129502973895256e+01, 7.2270467060308960e+01, 
                        7.5411483488848148e+01, 7.8552545984242926e+01, 8.1693649235601683e+01, 8.4834788718042290e+01, 
                        8.7975960552493220e+01, 9.1117161394464745e+01 ] )

        T = 2.0 / 15.0
        for i in jnp.arange(0,30):
            MUI    = mu[i]
            SMUI   = jnp.sin(MUI)
            CMUI   = jnp.cos(MUI)
            AI     = 2.0*SMUI/(MUI+SMUI*CMUI)
            A = jtu.np_like_set(A, i, AI)
            S = jtu.np_like_set(S, i, 2.0*AI*(CMUI-SMUI/MUI))
            AIMUI2 = AI*MUI**2
            for j in jnp.arange(i+1):
                if i == j:
                    R = jtu.np_like_set(R, jnp.array([i,i]), 0.5*(1.0+0.5*jnp.sin(MUI+MUI)/MUI)*AIMUI2**2)
                else:
                    MUJ    = mu[j]
                    R = jtu.np_like_set(R, jnp.array([i,j]), 0.5*(jnp.sin(MUI+MUJ )/(MUI+MUJ)+jnp.sin(MUI-MUJ )/(MUI-MUJ))*AIMUI2*A[j]*MUJ**2)
                    R = jtu.np_like_set(R, jnp.array([j,i]), R[i,j])
        
        #                                  n   2
        #  Calculate the functions p(x) = SUM x .
        #                           j     i=j  i

        for k in jnp.arange(n-1,-1,-1):
            P = jtu.np_like_set(P, k, P[k+1]+x[k]**2)

        #  Calculate the functions rho.

        for j in jnp.arange(30):
            MUJ2 = mu[j]*mu[j]
            U    = jnp.exp(-MUJ2*P[0])
            for k in jnp.arange(n):
                 DRHO = jtu.np_like_set(DRHO, jnp.array([j,k]), 2.0*U*x[k])
                 for l in jnp.arange(k,n):
                    D2RHO = jtu.np_like_set(D2RHO, jnp.array([j,k,l]), -4.0*MUJ2*U*x[k]*x[l])
                    if l == k:
                        D2RHO = jtu.np_like_set(D2RHO, jnp.array([j,k,l]), D2RHO[j,k,l]+2.0*U)
            ALPHA = -2.0
            for i in jnp.arange(1,n):
                EU = ALPHA*jnp.exp(-MUJ2*P[i])
                U  = U+EU
                for k in jnp.arange(i,n):
                     DRHO = jtu.np_like_set(DRHO, jnp.array([j,k]), DRHO[j,k]+2.0*EU*x[k])
                     for l in range(k,n):
                         D2RHO = jtu.np_like_set(D2RHO, jnp.array([j,k,l]), D2RHO[j,k,l]-4.0*MUJ2*EU*x[k]*x[l])
                         if l == k :
                             D2RHO = jtu.np_like_set(D2RHO, jnp.array([j,k,l]), D2RHO[j,k,l]+2.0*EU)
                ALPHA = -ALPHA
            U      = U+0.5*ALPHA
            RHO = jtu.np_like_set(RHO, j, -U/MUJ2)
            
        #  Evaluate the function and derivatives.

        f = T;
        for i in jnp.arange(30):
            SI   = S[i]
            RHOI = RHO[i]
            f    = f+SI*RHOI
            for k in jnp.arange(n):
                g = jtu.np_like_set(g, k, g[k]+SI*DRHO[i,k])
                for l in jnp.arange(k,n):
                     H = jtu.np_like_set(H, jnp.array([k,l]), H[k,l]+SI*D2RHO[i,k,l])
            for j in jnp.arange(30):
                RIJ  = R[i,j]
                RHOJ = RHO[j]
                f    = f+RIJ*RHOI*RHOJ
                for k in jnp.arange(n):
                    g = jtu.np_like_set(g, k, g[k]+RIJ*(RHOI*DRHO[j,k]+RHOJ*DRHO[i,k]))
                    for l in range(k,n):
                        H = jtu.np_like_set(H, jnp.array([k,l]), H[k,l]+RIJ*(RHOI*D2RHO[j,k,l]+RHOJ*D2RHO[i,k,l]+DRHO[i,k]*DRHO[j,l]+DRHO[j,k]*DRHO[i,l]))

        #   Symmetrize the Hessian.

        for k in jnp.arange(n):
            for l in jnp.arange(k+1,n):
                H = jtu.np_like_set(H, jnp.array([l,k]), H[k,l])
 
        return f, g, H
 
    @staticmethod
    def eSQR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[0]
        if not isinstance( f_, float ):
            f_ = f_.item();
        if nargout>1:
            dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[0]+EV_[0])
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0e+0)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eH(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_, g_, H_ = self.extfunc(self, EV_ )
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


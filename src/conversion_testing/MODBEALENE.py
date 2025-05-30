import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class MODBEALENE:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : MODBEALENE
#    *********
#    A variation on Beale's problem in 2 variables
#    This is a nonlinear equation variant of MODBEALE
# 
#    Source: An adaptation by Ph. Toint of Problem 5 in
#    J.J. More', B.S. Garbow and K.E. Hillstrom,
#    "Testing Unconstrained Optimization Software",
#    ACM Transactions on Mathematical Software, vol. 7(1), pp. 17-41, 1981.
# 
#    See also Buckley#89.
#    SIF input: Ph. Toint, Mar 2003.
#               Nick Gould (nonlinear equation version), Jan 2019
# 
#    classification = "C-CNOR2-AN-V-V"
# 
#    The number of variables is  2 * N/2
# 
#           Alternative values for the SIF file parameters:
# IE N/2                 1              $-PARAMETER     original value
# IE N/2                 2              $-PARAMETER
# IE N/2                 5              $-PARAMETER
# IE N/2                 100            $-PARAMETER
# IE N/2                 1000           $-PARAMETER
# IE N/2                 10000          $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'MODBEALENE'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N/2'] = int(5);  #  SIF file default value
        else:
            v_['N/2'] = int(args[0])
        if nargin<2:
            v_['ALPHA'] = float(50.0);  #  SIF file default value
        else:
            v_['ALPHA'] = float(args[1])
        v_['1'] = 1
        v_['N'] = v_['N/2']+v_['N/2']
        v_['N/2-1'] = -1+v_['N/2']
        v_['ALPHINV'] = 1.0/v_['ALPHA']
        v_['RALPHINV'] = jnp.sqrt(v_['ALPHINV'])
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for J in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(J),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['N/2-1'])+1):
            v_['I-1'] = -1+I
            v_['2I-1'] = v_['I-1']+v_['I-1']
            v_['J'] = 1+v_['2I-1']
            v_['J+1'] = 1+v_['J']
            v_['J+2'] = 2+v_['J']
            [ig,ig_,_] = jtu.s2mpj_ii('BA'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'BA'+str(I))
            [ig,ig_,_] = jtu.s2mpj_ii('BB'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'BB'+str(I))
            [ig,ig_,_] = jtu.s2mpj_ii('BC'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'BC'+str(I))
            [ig,ig_,_] = jtu.s2mpj_ii('L'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'L'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['J+1']))]])
            valA = jtu.append(valA,float(6.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(int(v_['J+2']))]])
            valA = jtu.append(valA,float(-1.0))
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['RALPHINV']))
        [ig,ig_,_] = jtu.s2mpj_ii('BA'+str(int(v_['N/2'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'BA'+str(int(v_['N/2'])))
        [ig,ig_,_] = jtu.s2mpj_ii('BB'+str(int(v_['N/2'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'BB'+str(int(v_['N/2'])))
        [ig,ig_,_] = jtu.s2mpj_ii('BC'+str(int(v_['N/2'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'BC'+str(int(v_['N/2'])))
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
        for I in range(int(v_['1']),int(v_['N/2'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['BA'+str(I)],float(1.5))
            self.gconst = jtu.arrset(self.gconst,ig_['BB'+str(I)],float(2.25))
            self.gconst = jtu.arrset(self.gconst,ig_['BC'+str(I)],float(2.625))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(1.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'ePRODB', iet_)
        elftv = jtu.loaset(elftv,it,0,'V1')
        elftv = jtu.loaset(elftv,it,1,'V2')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'POW')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['N/2'])+1):
            v_['I-1'] = -1+I
            v_['2I-1'] = v_['I-1']+v_['I-1']
            v_['J'] = 1+v_['2I-1']
            v_['J+1'] = 1+v_['J']
            ename = 'AE'+str(I)
            [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = jtu.arrset(self.elftype,ie,'ePRODB')
                ielftype = jtu.arrset(ielftype,ie,iet_['ePRODB'])
            vname = 'X'+str(int(v_['J']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['J+1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='POW')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(1.0))
            ename = 'BE'+str(I)
            [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = jtu.arrset(self.elftype,ie,'ePRODB')
                ielftype = jtu.arrset(ielftype,ie,iet_['ePRODB'])
            vname = 'X'+str(int(v_['J']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['J+1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='POW')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(2.0))
            ename = 'CE'+str(I)
            [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = jtu.arrset(self.elftype,ie,'ePRODB')
                ielftype = jtu.arrset(ielftype,ie,iet_['ePRODB'])
            vname = 'X'+str(int(v_['J']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['J+1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='V2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='POW')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(3.0))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N/2'])+1):
            ig = ig_['BA'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['AE'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            ig = ig_['BB'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['BE'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            ig = ig_['BC'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['CE'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
# LO SOLTN                0.0
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
        self.pbclass   = "C-CNOR2-AN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePRODB(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        T = 1.0-EV_[1]**self.elpar[iel_][0]
        POWM1 = self.elpar[iel_][0]-1.0
        W = -self.elpar[iel_][0]*EV_[1]**POWM1
        f_   = EV_[0]*T
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, T)
            g_ = jtu.np_like_set(g_, 1, EV_[0]*W)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 0.0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), W)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), (                       -EV_[0]*self.elpar[iel_][0]*POWM1*EV_[1]**(self.elpar[iel_][0]-2.0)))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


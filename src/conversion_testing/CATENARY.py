from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class CATENARY:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    An erroneous but interesting version of the classical problem of the hanging
#    catenary.  Here the catenary consists of N+1 beams of length BL,
#    with the first beam fixed at the origin and the final bean fixed
#    at a fraction FRACT of the total length of all beams.
# 
#    The correct version of the same problem is given by problem CATENA.SIF
# 
#    The problem is non-convex.
# 
#    Source: 
#    K. Veselic,
#    "De forma catenarum in campo gravitatis pendentium",
#    Klasicna Gimnazija u Zagrebu, Zagreb, 1987.
# 
#    SIF input: Ph. L. Toint, May 1993.
# 
#    classification = "C-CLQR2-AY-V-V"
# 
#    Number of beams = N+1 ; the number of variables is 3*(N+2)
# 
#           Alternative values for the SIF file parameters:
# IE N+1                 4              $-PARAMETER n = 15
# IE N+1                 10             $-PARAMETER n = 33
# IE N+1                 32             $-PARAMETER n = 99
# IE N+1                 166            $-PARAMETER n = 501     original value
# IE N+1                 1000           $-PARAMETER n = 3003
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'CATENARY'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N+1'] = int(4);  #  SIF file default value
        else:
            v_['N+1'] = int(args[0])
        v_['GAMMA'] = 9.81
        v_['TMASS'] = 500.0
        v_['BL'] = 1.0
        v_['FRACT'] = 0.6
        v_['0'] = 0
        v_['1'] = 1
        v_['N'] = -1+v_['N+1']
        v_['RN+1'] = float(v_['N+1'])
        v_['TMP'] = v_['BL']*v_['RN+1']
        v_['LENGTH'] = v_['TMP']*v_['FRACT']
        v_['MASS'] = v_['TMASS']/v_['RN+1']
        v_['MG'] = v_['MASS']*v_['GAMMA']
        v_['MG/2'] = 0.5*v_['MG']
        v_['BLSQ'] = v_['BL']*v_['BL']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['0']),int(v_['N+1'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('Y'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'Y'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('Z'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'Z'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Y'+str(int(v_['0']))]])
        valA = jtu.append(valA,float(v_['MG/2']))
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Y'+str(I)]])
            valA = jtu.append(valA,float(v_['MG']))
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['Y'+str(int(v_['N+1']))]])
        valA = jtu.append(valA,float(v_['MG/2']))
        for I in range(int(v_['1']),int(v_['N+1'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'C'+str(I))
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
        for I in range(int(v_['1']),int(v_['N+1'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['C'+str(I)],float(v_['BLSQ']))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['X0'], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['X0'], 0.0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['Y0'], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Y0'], 0.0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['Z0'], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['Z0'], 0.0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['X'+str(int(v_['N+1']))], v_['LENGTH'])
        self.xupper = jtu.np_like_set(self.xupper, ix_['X'+str(int(v_['N+1']))], v_['LENGTH'])
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        v_['TMP'] = v_['LENGTH']/v_['RN+1']
        for I in range(int(v_['1']),int(v_['N+1'])+1):
            v_['RI'] = float(I)
            v_['VAL'] = v_['TMP']*v_['RI']
            self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(v_['TMP']))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eISQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'V')
        elftv = jtu.loaset(elftv,it,1,'W')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['1']),int(v_['N+1'])+1):
            v_['I-1'] = -1+I
            ename = 'EX'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'X'+str(int(v_['I-1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EY'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
            vname = 'Y'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'Y'+str(int(v_['I-1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'EZ'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eISQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eISQ"])
            vname = 'X'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'Z'+str(int(v_['I-1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N+1'])+1):
            ig = ig_['C'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EX'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EY'+str(I)])
            self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EZ'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
# LO SOL(15)             -8184.81057777
# LO SOL(33)             -20837.3763330
# LO SOL(99)             -67050.9978802
# LO SOL(501)            -348403.164505
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
        self.pbclass   = "C-CLQR2-AY-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eISQ(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((1,2))
        IV_ = jnp.zeros(1)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]-1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        f_   = IV_[0]*IV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, IV_[0]+IV_[0])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0)
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


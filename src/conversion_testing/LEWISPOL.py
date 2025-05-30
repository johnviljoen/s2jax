import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LEWISPOL:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ********
# 
#    Adrian Lewis Polynomial Problem,
#    The problem is a transformation of a number theory integer
#    programming problem.
# 
#    Source:
#    A. Lewis, private communication.
# 
#    SIF input: A.R. Conn and Ph. Toint, March 1990.
# 
#    classification = "C-CQOR2-AN-6-9"
# 
#    Number of variables
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LEWISPOL'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['N'] = 6
        v_['DEG'] = 3
        v_['PEN'] = 1.0e4
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['DEG-1'] = -1+v_['DEG']
        v_['N-1'] = -1+v_['N']
        v_['N+1'] = 1+v_['N']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for J in range(int(v_['0']),int(v_['N-1'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('A'+str(J),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'A'+str(J))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for J in range(int(v_['0']),int(v_['N-1'])+1):
            v_['C'+str(int(v_['0']))+','+str(J)] = 1.0
            [ig,ig_,_] = jtu.s2mpj_ii('D0',ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'D0')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A'+str(J)]])
            valA = jtu.append(valA,float(v_['C'+str(int(v_['0']))+','+str(J)]))
        for I in range(int(v_['1']),int(v_['DEG-1'])+1):
            v_['I-1'] = -1+I
            for J in range(int(I),int(v_['N-1'])+1):
                v_['RJ'] = float(J)
                v_['C'+str(I)+','+str(J)] = v_['C'+str(int(v_['I-1']))+','+str(J)]*v_['RJ']
                [ig,ig_,_] = jtu.s2mpj_ii('D'+str(I),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'D'+str(I))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['A'+str(J)]])
                valA = jtu.append(valA,float(v_['C'+str(I)+','+str(J)]))
        for J in range(int(v_['0']),int(v_['N-1'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('INT'+str(J),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'INT'+str(J))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['A'+str(J)]])
            valA = jtu.append(valA,float(-1.0))
            self.gscale = jtu.arrset(self.gscale,ig,float(v_['PEN']))
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
        v_['CT'+str(int(v_['0']))] = -1.0
        self.gconst  = (               jtu.arrset(self.gconst,ig_['D0'],float(v_['CT'+str(int(v_['0']))])))
        for I in range(int(v_['1']),int(v_['DEG-1'])+1):
            v_['I-1'] = -1+I
            v_['-I'] = -1*I
            v_['N+1-I'] = v_['N+1']+v_['-I']
            v_['VAL'] = float(v_['N+1-I'])
            v_['CT'+str(I)] = v_['CT'+str(int(v_['I-1']))]*v_['VAL']
            self.gconst = jtu.arrset(self.gconst,ig_['D'+str(I)],float(v_['CT'+str(I)]))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-10.0)
        self.xupper = jnp.full((self.n,1),10.0)
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        if('A0' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['A0'], float(-1.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['A0']),float(-1.0)))
        if('A1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['A1'], float(1.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['A1']),float(1.0)))
        if('A2' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['A2'], float(1.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['A2']),float(1.0)))
        if('A3' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['A3'], float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['A3']),float(0.0)))
        if('A4' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['A4'], float(1.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['A4']),float(1.0)))
        if('A5' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['A5'], float(-1.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['A5']),float(-1.0)))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQ', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        [it,iet_,_] = jtu.s2mpj_ii( 'eCB', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for J in range(int(v_['0']),int(v_['N-1'])+1):
            ename = 'O'+str(J)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eSQ')
            ielftype = jtu.arrset(ielftype,ie,iet_["eSQ"])
            vname = 'A'+str(J)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(-10.0),float(10.0),None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'E'+str(J)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eCB')
            ielftype = jtu.arrset(ielftype,ie,iet_["eCB"])
            vname = 'A'+str(J)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,float(-10.0),float(10.0),None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for J in range(int(v_['0']),int(v_['N-1'])+1):
            ig = ig_['OBJ']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['O'+str(J)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            ig = ig_['INT'+str(J)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(J)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
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
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CQOR2-AN-6-9"
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
    def eCB(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]**3
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 3.0*EV_[0]**2)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 6.0*EV_[0])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


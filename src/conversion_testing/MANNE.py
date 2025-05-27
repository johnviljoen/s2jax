from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class MANNE:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : MANNE
#    *********
# 
#    A variable dimension econometric equilibrium problem
#    suggested by A. Manne
# 
#    Source:
#    B. Murtagh and M. Saunders,
#    Mathematical Programming Studies 16, pp. 84-117,
#    (example 5.12).
# 
#    SIF input: N. Gould and Ph. Toint, March 1990.
# 
#    classification = "C-COOR2-MN-V-V"
# 
#    Number of periods
#    The number of variables in the problem N = 3*T
# 
#           Alternative values for the SIF file parameters:
# IE T                   100            $-PARAMETER n = 300    original value
# IE T                   365            $-PARAMETER n = 995
# IE T                   1000           $-PARAMETER n = 3000
# IE T                   2000           $-PARAMETER n = 6000
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'MANNE'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['T'] = int(4);  #  SIF file default value
        else:
            v_['T'] = int(args[0])
        v_['GROW'] = 0.03
        v_['BETA'] = 0.95
        v_['XK0'] = 3.0
        v_['XC0'] = 0.95
        v_['XI0'] = 0.05
        v_['B'] = 0.25
        v_['BPROB'] = 1.0
        v_['1'] = 1
        v_['2'] = 2
        v_['T-1'] = -1+v_['T']
        v_['T-2'] = -2+v_['T']
        v_['LOGXK'] = jnp.log(v_['XK0'])
        v_['BLOGX'] = v_['LOGXK']*v_['B']
        v_['XK0**B'] = jnp.exp(v_['BLOGX'])
        v_['NUM'] = v_['XC0']+v_['XI0']
        v_['A'] = v_['NUM']/v_['XK0**B']
        v_['1-B'] = 1.0-v_['B']
        v_['1+G'] = 1.0+v_['GROW']
        v_['LOG1+G'] = jnp.log(v_['1+G'])
        v_['SOME'] = v_['LOG1+G']*v_['1-B']
        v_['GFAC'] = jnp.exp(v_['SOME'])
        v_['AT1'] = v_['A']*v_['GFAC']
        v_['BT1'] = 0.0+v_['BETA']
        for J in range(int(v_['2']),int(v_['T'])+1):
            v_['J-1'] = -1+J
            v_['AT'+str(J)] = v_['AT'+str(int(v_['J-1']))]*v_['GFAC']
            v_['BT'+str(J)] = v_['BT'+str(int(v_['J-1']))]*v_['BETA']
        v_['1-BETA'] = 1.0-v_['BETA']
        v_['1/1-BETA'] = 1.0/v_['1-BETA']
        v_['BT'+str(int(v_['T']))] = v_['BT'+str(int(v_['T']))]*v_['1/1-BETA']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['T'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('C'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'C'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('I'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'I'+str(I))
            [iv,ix_,_] = jtu.s2mpj_ii('K'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'K'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for I in range(int(v_['1']),int(v_['T'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('NL'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'NL'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['C'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['I'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        for I in range(int(v_['1']),int(v_['T-1'])+1):
            v_['I+1'] = 1+I
            [ig,ig_,_] = jtu.s2mpj_ii('L'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<=')
            cnames = jtu.arrset(cnames,ig,'L'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['K'+str(int(v_['I+1']))]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['K'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['I'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('L'+str(int(v_['T'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'L'+str(int(v_['T'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['K'+str(int(v_['T']))]])
        valA = jtu.append(valA,float(v_['GROW']))
        [ig,ig_,_] = jtu.s2mpj_ii('L'+str(int(v_['T'])),ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'L'+str(int(v_['T'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['I'+str(int(v_['T']))]])
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
        #%%%%%%%%%%%%%%%%%%%%  RANGES %%%%%%%%%%%%%%%%%%%%%%
        grange = jnp.full((ngrp,1),None)
        grange = jtu.np_like_set(grange, legrps, jnp.full((self.nle,1),float('inf')))
        grange = jtu.np_like_set(grange, gegrps, jnp.full((self.nge,1),float('inf')))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['K1'], 3.05)
        self.xupper = jtu.np_like_set(self.xupper, ix_['K1'], 3.05)
        for I in range(int(v_['2']),int(v_['T'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['K'+str(I)], 3.05)
        v_['1.04**T'] = 0.05
        for I in range(int(v_['1']),int(v_['T'])+1):
            v_['1.04**T'] = 1.04*v_['1.04**T']
            self.xlower = jtu.np_like_set(self.xlower, ix_['C'+str(I)], 0.95)
            self.xlower = jtu.np_like_set(self.xlower, ix_['I'+str(I)], 0.05)
            self.xupper = jtu.np_like_set(self.xupper, ix_['I'+str(I)], v_['1.04**T'])
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        if('K1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['K1'], float(3.05))
        else:
            self.y0  = (
                  jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['K1']),float(3.05)))
        for I in range(int(v_['2']),int(v_['T'])+1):
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            v_['I-1/10'] = 0.1*v_['RI-1']
            v_['VAL'] = 3.0+v_['I-1/10']
            if('K'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['K'+str(I)], float(v_['VAL']))
            else:
                self.y0  = (
                      jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['K'+str(I)]),float(v_['VAL'])))
        for I in range(int(v_['1']),int(v_['T'])+1):
            if('C'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['C'+str(I)], float(0.95))
            else:
                self.y0  = (
                      jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['C'+str(I)]),float(0.95)))
            if('I'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['I'+str(I)], float(0.05))
            else:
                self.y0  = (
                      jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['I'+str(I)]),float(0.05)))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eLOGS', iet_)
        elftv = jtu.loaset(elftv,it,0,'C')
        [it,iet_,_] = jtu.s2mpj_ii( 'ePOWER', iet_)
        elftv = jtu.loaset(elftv,it,0,'K')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'B')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['T'])+1):
            ename = 'LOGC'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eLOGS')
            ielftype = jtu.arrset(ielftype,ie,iet_["eLOGS"])
            vname = 'C'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='C')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'KS'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'ePOWER')
            ielftype = jtu.arrset(ielftype,ie,iet_["ePOWER"])
            vname = 'K'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='K')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='B')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['B']))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['T'])+1):
            ig = ig_['OBJ']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['LOGC'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['BT'+str(I)]))
            ig = ig_['NL'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['KS'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['AT'+str(I)]))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               -9.7457259D-01
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COOR2-MN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eLOGS(self, nargout,*args):

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
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), -1.0/EV_[0]**2)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def ePOWER(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]**self.elpar[iel_][0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, self.elpar[iel_][0]*EV_[0]**(self.elpar[iel_][0]-1.0))
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), ()
                      self.elpar[iel_][0]*(self.elpar[iel_][0]-1.0)*EV_[0]**(self.elpar[iel_][0]-2.0))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


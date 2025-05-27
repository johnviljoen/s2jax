from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class CHEMRCTA:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : CHEMRCTA
#    *********
# 
#    The tubular chemical reactor model problem by Poore, using a
#    finite difference approximation to the steady state solutions.
# 
#    Source: Problem 8, eqs (8.6)--(8.9) in
#    J.J. More',
#    "A collection of nonlinear model problems"
#    Proceedings of the AMS-SIAM Summer seminar on the Computational
#    Solution of Nonlinear Systems of Equations, Colorado, 1988.
#    Argonne National Laboratory MCS-P60-0289, 1989.
# 
#    SIF input: Ph. Toint, Dec 1989.
#               minor correction by Ph. Shott, Jan 1995 and F Ruediger, Mar 1997.
# 
#    classification = "C-CNOR2-MN-V-V"
# 
#    The axial coordinate interval is [0,1]
# 
#    Number of discretized point for the interval [0,1].
#    The number of variables is 2N.
# 
#           Alternative values for the SIF file parameters:
# IE N                   5              $-PARAMETER n = 10
# IE N                   25             $-PARAMETER n = 50
# IE N                   50             $-PARAMETER n = 100
# IE N                   250            $-PARAMETER n = 500    original value
# IE N                   500            $-PARAMETER n = 1000
# IE N                   2500           $-PARAMETER n = 5000
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'CHEMRCTA'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['N'] = int(5);  #  SIF file default value
        else:
            v_['N'] = int(args[0])
        if nargin<2:
            v_['PEM'] = float(1.0);  #  SIF file default value
        else:
            v_['PEM'] = float(args[1])
        if nargin<3:
            v_['PEH'] = float(5.0);  #  SIF file default value
        else:
            v_['PEH'] = float(args[2])
        if nargin<4:
            v_['D'] = float(0.135);  #  SIF file default value
        else:
            v_['D'] = float(args[3])
        if nargin<5:
            v_['B'] = float(0.5);  #  SIF file default value
        else:
            v_['B'] = float(args[4])
        if nargin<6:
            v_['BETA'] = float(2.0);  #  SIF file default value
        else:
            v_['BETA'] = float(args[5])
        if nargin<7:
            v_['GAMMA'] = float(25.0);  #  SIF file default value
        else:
            v_['GAMMA'] = float(args[6])
        v_['1'] = 1
        v_['2'] = 2
        v_['1.0'] = 1.0
        v_['N-1'] = -1+v_['N']
        v_['1/H'] = float(v_['N-1'])
        v_['-1/H'] = -1.0*v_['1/H']
        v_['H'] = v_['1.0']/v_['1/H']
        v_['1/H2'] = v_['1/H']*v_['1/H']
        v_['-D'] = -1.0*v_['D']
        v_['1/PEM'] = v_['1.0']/v_['PEM']
        v_['1/H2PEM'] = v_['1/PEM']*v_['1/H2']
        v_['-1/H2PM'] = -1.0*v_['1/H2PEM']
        v_['HPEM'] = v_['PEM']*v_['H']
        v_['-HPEM'] = -1.0*v_['HPEM']
        v_['-2/H2PM'] = v_['-1/H2PM']+v_['-1/H2PM']
        v_['CU1'] = 1.0*v_['-HPEM']
        v_['CUI-1'] = v_['1/H2PEM']+v_['1/H']
        v_['CUI'] = v_['-2/H2PM']+v_['-1/H']
        v_['BD'] = v_['B']*v_['D']
        v_['-BETA'] = -1.0*v_['BETA']
        v_['1/PEH'] = v_['1.0']/v_['PEH']
        v_['1/H2PEH'] = v_['1/PEH']*v_['1/H2']
        v_['-1/H2PH'] = -1.0*v_['1/H2PEH']
        v_['HPEH'] = v_['PEH']*v_['H']
        v_['-HPEH'] = -1.0*v_['HPEH']
        v_['-2/H2PH'] = v_['-1/H2PH']+v_['-1/H2PH']
        v_['CT1'] = 1.0*v_['-HPEH']
        v_['CTI-1'] = v_['1/H2PEH']+v_['1/H']
        v_['CTI'] = v_['-2/H2PH']+v_['-1/H']
        v_['CTI'] = v_['CTI']+v_['-BETA']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('T'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'T'+str(I))
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('U'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'U'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('GU'+str(int(v_['1'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'GU'+str(int(v_['1'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['U'+str(int(v_['1']))]])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('GU'+str(int(v_['1'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'GU'+str(int(v_['1'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['U'+str(int(v_['2']))]])
        valA = jtu.append(valA,float(v_['CU1']))
        [ig,ig_,_] = jtu.s2mpj_ii('GT'+str(int(v_['1'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'GT'+str(int(v_['1'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T'+str(int(v_['1']))]])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('GT'+str(int(v_['1'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'GT'+str(int(v_['1'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T'+str(int(v_['2']))]])
        valA = jtu.append(valA,float(v_['CT1']))
        for I in range(int(v_['2']),int(v_['N-1'])+1):
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            [ig,ig_,_] = jtu.s2mpj_ii('GU'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'GU'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['CUI-1']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(I)]])
            valA = jtu.append(valA,float(v_['CUI']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['U'+str(int(v_['I+1']))]])
            valA = jtu.append(valA,float(v_['1/H2PEM']))
            [ig,ig_,_] = jtu.s2mpj_ii('GT'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'GT'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
            valA = jtu.append(valA,float(v_['BETA']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['CTI-1']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(I)]])
            valA = jtu.append(valA,float(v_['CTI']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['T'+str(int(v_['I+1']))]])
            valA = jtu.append(valA,float(v_['1/H2PEH']))
        [ig,ig_,_] = jtu.s2mpj_ii('GU'+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'GU'+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['U'+str(int(v_['N-1']))]])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('GU'+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'GU'+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['U'+str(int(v_['N']))]])
        valA = jtu.append(valA,float(1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('GT'+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'GT'+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T'+str(int(v_['N-1']))]])
        valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('GT'+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'GT'+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['T'+str(int(v_['N']))]])
        valA = jtu.append(valA,float(1.0))
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
        self.gconst  = (
              jtu.arrset(self.gconst,ig_['GU'+str(int(v_['1']))],float(v_['-HPEM'])))
        self.gconst  = (
              jtu.arrset(self.gconst,ig_['GT'+str(int(v_['1']))],float(v_['-HPEH'])))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['T'+str(I)], 0.0000001)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(1.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eREAC', iet_)
        elftv = jtu.loaset(elftv,it,0,'U')
        elftv = jtu.loaset(elftv,it,1,'T')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'G')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['2']),int(v_['N-1'])+1):
            ename = 'EU'+str(I)
            [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = jtu.arrset(self.elftype,ie,'eREAC')
                ielftype = jtu.arrset(ielftype,ie,iet_['eREAC'])
            vname = 'U'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'T'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='T')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='G')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['GAMMA']))
            ename = 'ET'+str(I)
            [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
            if newelt:
                self.elftype = jtu.arrset(self.elftype,ie,'eREAC')
                ielftype = jtu.arrset(ielftype,ie,iet_['eREAC'])
            vname = 'U'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'T'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(1.0))
            posev = jnp.where(elftv[ielftype[ie]]=='T')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='G')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['GAMMA']))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['2']),int(v_['N-1'])+1):
            ig = ig_['GU'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EU'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-D']))
            ig = ig_['GT'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['ET'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['BD']))
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
        self.pbclass   = "C-CNOR2-MN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eREAC(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        DADT = self.elpar[iel_][0]/(EV_[1]*EV_[1])
        D2ADT2 = -2.0*DADT/EV_[1]
        EX = jnp.exp(self.elpar[iel_][0]-self.elpar[iel_][0]/EV_[1])
        UEX = EX*EV_[0]
        f_   = UEX
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EX)
            g_ = jtu.np_like_set(g_, 1, UEX*DADT)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), EX*DADT)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), UEX*(DADT*DADT+D2ADT2))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class MANCINONE:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : MANCINONE
#    *********
# 
#    Mancino's function with variable dimension.
#    This is a nonlinear equation variant of MANCINO
# 
#    Source:
#    E. Spedicato,
#    "Computational experience with quasi-Newton algorithms for
#    minimization problems of moderate size",
#    Report N-175, CISE, Milano, 1975.
# 
#    See also Buckley #51 (p. 72), Schittkowski #391 (for N = 30)
# 
#    SIF input: Ph. Toint, Dec 1989.
#               correction by Ph. Shott, January, 1995.
#               Nick Gould (nonlinear equation version), Jan 2019
#               correction by S. Gratton & Ph. Toint, May 2024
# 
#    classification = "C-CNOR2-AN-V-V"
# 
#    The definitions
#      s_{i,j} = \sin \log v_{i,j}   and s_{i,j} = \cos \log v_{i,j}
#    have been used.  It seems that the additional exponent ALPHA
#    in Buckley is a typo.
# 
#    Number of variables
# 
#           Alternative values for the SIF file parameters:
# IE N                   10             $-PARAMETER
# IE N                   20             $-PARAMETER
# IE N                   30             $-PARAMETER Schittkowski #391
# IE N                   50             $-PARAMETER
# IE N                   100            $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'MANCINONE'

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
        if nargin<2:
            v_['ALPHA'] = int(5);  #  SIF file default value
        else:
            v_['ALPHA'] = int(args[1])
        if nargin<3:
            v_['BETA'] = float(14.0);  #  SIF file default value
        else:
            v_['BETA'] = float(args[2])
        if nargin<4:
            v_['GAMMA'] = int(3);  #  SIF file default value
        else:
            v_['GAMMA'] = int(args[3])
        v_['RALPHA'] = float(v_['ALPHA'])
        v_['RN'] = float(v_['N'])
        v_['N-1'] = -1+v_['N']
        v_['RN-1'] = float(v_['N-1'])
        v_['N-1SQ'] = v_['RN-1']*v_['RN-1']
        v_['BETAN'] = v_['BETA']*v_['RN']
        v_['BETAN2'] = v_['BETAN']*v_['BETAN']
        v_['AL+1'] = 1.0+v_['RALPHA']
        v_['A1SQ'] = v_['AL+1']*v_['AL+1']
        v_['F0'] = v_['A1SQ']*v_['N-1SQ']
        v_['F1'] = -1.0*v_['F0']
        v_['F2'] = v_['BETAN2']+v_['F1']
        v_['F3'] = 1.0/v_['F2']
        v_['F4'] = v_['BETAN']*v_['F3']
        v_['A'] = -1.0*v_['F4']
        v_['-N/2'] = -0.5*v_['RN']
        v_['1'] = 1
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('G'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'G'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['X'+str(I)]])
            valA = jtu.append(valA,float(v_['BETAN']))
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
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['I-N/2'] = v_['RI']+v_['-N/2']
            v_['CI'] = 1.0
            for J in range(int(v_['1']),int(v_['GAMMA'])+1):
                v_['CI'] = v_['CI']*v_['I-N/2']
            self.gconst = jtu.arrset(self.gconst,ig_['G'+str(I)],float(v_['CI']))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            v_['RI'] = float(I)
            v_['H'] = 0.0
            for J in range(int(v_['1']),int(v_['I-1'])+1):
                v_['RJ'] = float(J)
                v_['1/J'] = 1.0/v_['RJ']
                v_['I/J'] = v_['RI']*v_['1/J']
                v_['SQI/J'] = jnp.sqrt(v_['I/J'])
                v_['LIJ'] = jnp.log(v_['SQI/J'])
                v_['SIJ'] = jnp.sin(v_['LIJ'])
                v_['CIJ'] = jnp.cos(v_['LIJ'])
                v_['SA'] = 1.0
                v_['CA'] = 1.0
                for K in range(int(v_['1']),int(v_['ALPHA'])+1):
                    v_['SA'] = v_['SA']*v_['SIJ']
                    v_['CA'] = v_['CA']*v_['CIJ']
                v_['SCA'] = v_['SA']+v_['CA']
                v_['HIJ'] = v_['SQI/J']*v_['SCA']
                v_['H'] = v_['H']+v_['HIJ']
            v_['I+1'] = 1+I
            for J in range(int(v_['I+1']),int(v_['N'])+1):
                v_['RJ'] = float(J)
                v_['1/J'] = 1.0/v_['RJ']
                v_['I/J'] = v_['RI']*v_['1/J']
                v_['SQI/J'] = jnp.sqrt(v_['I/J'])
                v_['LIJ'] = jnp.log(v_['SQI/J'])
                v_['SIJ'] = jnp.sin(v_['LIJ'])
                v_['CIJ'] = jnp.cos(v_['LIJ'])
                v_['SA'] = 1.0
                v_['CA'] = 1.0
                for K in range(int(v_['1']),int(v_['ALPHA'])+1):
                    v_['SA'] = v_['SA']*v_['SIJ']
                    v_['CA'] = v_['CA']*v_['CIJ']
                v_['SCA'] = v_['SA']+v_['CA']
                v_['HIJ'] = v_['SQI/J']*v_['SCA']
                v_['H'] = v_['H']+v_['HIJ']
            v_['I-N/2'] = v_['RI']+v_['-N/2']
            v_['CI'] = 1.0
            for J in range(int(v_['1']),int(v_['GAMMA'])+1):
                v_['CI'] = v_['CI']*v_['I-N/2']
            v_['TMP'] = v_['H']+v_['CI']
            v_['XI0'] = v_['TMP']*v_['A']
            if('X'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['X'+str(I)], float(v_['XI0']))
            else:
                self.y0  = (                       jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['X'+str(I)]),float(v_['XI0'])))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eMANC', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'II')
        elftp = jtu.loaset(elftp,it,1,'JJ')
        elftp = jtu.loaset(elftp,it,2,'AL')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['RI'] = float(I)
            v_['I-1'] = -1+I
            for J in range(int(v_['1']),int(v_['I-1'])+1):
                v_['RJ'] = float(J)
                ename = 'E'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eMANC')
                ielftype = jtu.arrset(ielftype,ie,iet_["eMANC"])
                vname = 'X'+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                posep = jnp.where(elftp[ielftype[ie]]=='II')[0]
                self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['RI']))
                posep = jnp.where(elftp[ielftype[ie]]=='JJ')[0]
                self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['RJ']))
                posep = jnp.where(elftp[ielftype[ie]]=='AL')[0]
                self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['RALPHA']))
            v_['I+1'] = 1+I
            for J in range(int(v_['I+1']),int(v_['N'])+1):
                v_['RJ'] = float(J)
                ename = 'E'+str(I)+','+str(J)
                [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                self.elftype = jtu.arrset(self.elftype,ie,'eMANC')
                ielftype = jtu.arrset(ielftype,ie,iet_["eMANC"])
                vname = 'X'+str(J)
                [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
                self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                posep = jnp.where(elftp[ielftype[ie]]=='II')[0]
                self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['RI']))
                posep = jnp.where(elftp[ielftype[ie]]=='JJ')[0]
                self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['RJ']))
                posep = jnp.where(elftp[ielftype[ie]]=='AL')[0]
                self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['RALPHA']))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            v_['I-1'] = -1+I
            for J in range(int(v_['1']),int(v_['I-1'])+1):
                ig = ig_['G'+str(I)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
            v_['I+1'] = 1+I
            for J in range(int(v_['I+1']),int(v_['N'])+1):
                ig = ig_['G'+str(I)]
                posel = len(self.grelt[ig])
                self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['E'+str(I)+','+str(J)])
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Least square problems are bounded below by zero
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
        self.pbclass   = "C-CNOR2-AN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eMANC(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        IAL = self.elpar[iel_][2]
        IA1 = IAL-1
        A2 = self.elpar[iel_][2]-2.0
        IA2 = IAL-2
        IA3 = IAL-3
        INVIJ = EV_[0]*EV_[0]+self.elpar[iel_][0]/self.elpar[iel_][1]
        VIJ = jnp.sqrt(INVIJ)
        V2 = VIJ*VIJ
        DVIJ = EV_[0]/VIJ
        LIJ = jnp.log(VIJ)
        SIJ = jnp.sin(LIJ)
        CIJ = jnp.cos(LIJ)
        DSDX = CIJ*DVIJ/VIJ
        DCDX = -SIJ*DVIJ/VIJ
        SUMAL = SIJ**IAL+CIJ**IAL
        DSUMAL = self.elpar[iel_][2]*(DSDX*SIJ**IA1+DCDX*CIJ**IA1)
        SCIJ = SIJ*CIJ
        DSCIJ = SIJ*DCDX+DSDX*CIJ
        SAL = SIJ**IA2-CIJ**IA2
        DSAL = A2*(DSDX*SIJ**IA3-DCDX*CIJ**IA3)
        B = SUMAL+self.elpar[iel_][2]*SCIJ*SAL
        DBDX = DSUMAL+self.elpar[iel_][2]*(DSCIJ*SAL+SCIJ*DSAL)
        f_   = VIJ*SUMAL
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[0]*B/VIJ)
            if nargout>2:
                H_ = jnp.zeros((1,1))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), (B+EV_[0]*DBDX)/VIJ-B*EV_[0]*DVIJ/V2)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


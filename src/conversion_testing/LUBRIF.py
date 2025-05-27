from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class LUBRIF:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : LUBRIF
#    *********
# 
#    The elastodynamic lubrification problem by Kostreva.
#    NB. This version has an error. 
#    See LUBRIFC for the correct formulation.
# 
#    Source:
#    M.M. Kostreva,
#    "Elasto-hydrodynamic lubrification: a non-linear
#    complementarity problem",
#    International Journal for Numerical Methods in Fluids,
#    4: 377-397, 1984.
# 
#    This problem is problem #5 in More's test set.
# 
#    SIF input: Ph. Toint, June 1990.
# 
#    classification = "C-CQOR2-MN-V-V"
# 
#    Number of discretized points per unit length
# 
#           Alternative values for the SIF file parameters:
# IE NN                  10             $-PARAMETER n = 151    original value
# IE NN                  50             $-PARAMETER n = 751
# IE NN                  250            $-PARAMETER n = 3751
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'LUBRIF'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        if nargin<1:
            v_['NN'] = int(5);  #  SIF file default value
        else:
            v_['NN'] = int(args[0])
        v_['ALPHA'] = 1.838
        v_['LAMBDA'] = 1.642
        v_['XA'] = -3.0
        v_['XF'] = 2.0
        v_['N'] = 5*v_['NN']
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['4'] = 4
        v_['PI'] = 3.1415926535
        v_['2N'] = 2*v_['N']
        v_['2N-2'] = -2+v_['2N']
        v_['2N-1'] = -1+v_['2N']
        v_['2N+2'] = 2+v_['2N']
        v_['-XA'] = -1.0*v_['XA']
        v_['LEN'] = v_['XF']+v_['-XA']
        v_['1/PI'] = 1.0/v_['PI']
        v_['1/2PI'] = 0.5*v_['1/PI']
        v_['RN'] = float(v_['N'])
        v_['1/N'] = 1.0/v_['RN']
        v_['DX'] = v_['LEN']*v_['1/N']
        v_['1/DX'] = 1.0/v_['DX']
        v_['L/DX'] = v_['LAMBDA']*v_['1/DX']
        v_['-L/DX'] = -1.0*v_['L/DX']
        v_['1/DX2'] = v_['1/DX']*v_['1/DX']
        v_['-1/DX2'] = -1.0*v_['1/DX2']
        v_['DX/PI'] = v_['DX']*v_['1/PI']
        v_['2DX/PI'] = 2.0*v_['DX/PI']
        v_['DX/2'] = 0.5*v_['DX']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('K',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'K')
        for I in range(int(v_['0']),int(v_['2N'])+1,int(v_['2'])):
            [iv,ix_,_] = jtu.s2mpj_ii('P'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'P'+str(I))
        for J in range(int(v_['1']),int(v_['2N-1'])+1,int(v_['2'])):
            [iv,ix_,_] = jtu.s2mpj_ii('H'+str(J),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'H'+str(J))
        for I in range(int(v_['2']),int(v_['2N-2'])+1,int(v_['2'])):
            [iv,ix_,_] = jtu.s2mpj_ii('R'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'R'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['2']),int(v_['2N-2'])+1,int(v_['2'])):
            [ig,ig_,_] = jtu.s2mpj_ii('R'+str(int(v_['0'])),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'R'+str(int(v_['0'])))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['P'+str(I)]])
            valA = jtu.append(valA,float(v_['2DX/PI']))
        [ig,ig_,_] = jtu.s2mpj_ii('R'+str(int(v_['0'])),ig_)
        gtype = jtu.arrset(gtype,ig,'==')
        cnames = jtu.arrset(cnames,ig,'R'+str(int(v_['0'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['P'+str(int(v_['2N']))]])
        valA = jtu.append(valA,float(v_['DX/PI']))
        [ig,ig_,_] = jtu.s2mpj_ii('COMPL',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for I in range(int(v_['2']),int(v_['2N-2'])+1,int(v_['2'])):
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            [ig,ig_,_] = jtu.s2mpj_ii('DR'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'DR'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['H'+str(int(v_['I+1']))]])
            valA = jtu.append(valA,float(v_['L/DX']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['H'+str(int(v_['I-1']))]])
            valA = jtu.append(valA,float(v_['-L/DX']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['R'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        for J in range(int(v_['1']),int(v_['2N-1'])+1,int(v_['2'])):
            v_['-J'] = -1*J
            [ig,ig_,_] = jtu.s2mpj_ii('DH'+str(J),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'DH'+str(J))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['K']])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['H'+str(J)]])
            valA = jtu.append(valA,float(-1.0))
            for I in range(int(v_['2']),int(v_['2N'])+1):
                v_['C'+str(I)] = 0.0
            v_['RI-J'] = float(v_['-J'])
            v_['I-JDX'] = v_['RI-J']*v_['DX/2']
            v_['ALN'] = jnp.absolute(v_['I-JDX'])
            v_['LN'] = jnp.log(v_['ALN'])
            v_['T1'] = v_['I-JDX']*v_['LN']
            v_['COEFF'] = v_['T1']*v_['1/2PI']
            v_['C'+str(int(v_['2']))] = v_['C'+str(int(v_['2']))]+v_['COEFF']
            v_['I-J'] = 2+v_['-J']
            v_['RI-J'] = float(v_['I-J'])
            v_['I-JDX'] = v_['RI-J']*v_['DX/2']
            v_['ALN'] = jnp.absolute(v_['I-JDX'])
            v_['LN'] = jnp.log(v_['ALN'])
            v_['T1'] = v_['I-JDX']*v_['LN']
            v_['COEFF'] = v_['T1']*v_['1/PI']
            v_['C'+str(int(v_['4']))] = v_['C'+str(int(v_['4']))]+v_['COEFF']
            for I in range(int(v_['4']),int(v_['2N-2'])+1,int(v_['2'])):
                v_['I-2'] = -2+I
                v_['I+2'] = 2+I
                v_['I-J'] = I+v_['-J']
                v_['RI-J'] = float(v_['I-J'])
                v_['I-JDX'] = v_['RI-J']*v_['DX/2']
                v_['ALN'] = jnp.absolute(v_['I-JDX'])
                v_['LN'] = jnp.log(v_['ALN'])
                v_['T1'] = v_['I-JDX']*v_['LN']
                v_['COEFF'] = v_['T1']*v_['1/PI']
                v_['C'+str(int(v_['I+2']))] = v_['C'+str(int(v_['I+2']))]+v_['COEFF']
                v_['-COEFF'] = -1.0*v_['COEFF']
                v_['C'+str(int(v_['I-2']))] = v_['C'+str(int(v_['I-2']))]+v_['-COEFF']
            v_['I-J'] = v_['2N']+v_['-J']
            v_['RI-J'] = float(v_['I-J'])
            v_['I-JDX'] = v_['RI-J']*v_['DX/2']
            v_['ALN'] = jnp.absolute(v_['I-JDX'])
            v_['LN'] = jnp.log(v_['ALN'])
            v_['T1'] = v_['I-JDX']*v_['LN']
            v_['COEFF'] = v_['T1']*v_['1/2PI']
            v_['-COEFF'] = -1.0*v_['COEFF']
            v_['C'+str(int(v_['2N-2']))] = v_['C'+str(int(v_['2N-2']))]+v_['-COEFF']
            for I in range(int(v_['2']),int(v_['2N-2'])+1,int(v_['2'])):
                [ig,ig_,_] = jtu.s2mpj_ii('DH'+str(J),ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'DH'+str(J))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['P'+str(I)]])
                valA = jtu.append(valA,float(v_['C'+str(I)]))
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
        self.gconst = jtu.arrset(self.gconst,ig_['R'+str(int(v_['0']))],float(1.0))
        for J in range(int(v_['1']),int(v_['2N-1'])+1,int(v_['2'])):
            v_['RJ'] = float(J)
            v_['JDX'] = v_['RJ']*v_['DX/2']
            v_['XJ'] = v_['XA']+v_['JDX']
            v_['XJSQ'] = v_['XJ']*v_['XJ']
            v_['XJSQ+1'] = 1.0+v_['XJSQ']
            v_['RHS'] = -1.0*v_['XJSQ+1']
            self.gconst = jtu.arrset(self.gconst,ig_['DH'+str(J)],float(v_['RHS']))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['K'], -float('Inf'))
        self.xupper = jtu.np_like_set(self.xupper, ix_['K'], +float('Inf'))
        for I in range(int(v_['2']),int(v_['2N-2'])+1,int(v_['2'])):
            self.xupper = jtu.np_like_set(self.xupper, ix_['P'+str(I)], 3.0)
            self.xlower = jtu.np_like_set(self.xlower, ix_['P'+str(I)], 0.0)
        for I in range(int(v_['1']),int(v_['2N-1'])+1,int(v_['2'])):
            self.xlower = jtu.np_like_set(self.xlower, ix_['H'+str(I)], -float('Inf'))
            self.xupper = jtu.np_like_set(self.xupper, ix_['H'+str(I)], +float('Inf'))
        self.xlower = jtu.np_like_set(self.xlower, ix_['P'+str(int(v_['0']))], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['P'+str(int(v_['0']))], 0.0)
        self.xlower = jtu.np_like_set(self.xlower, ix_['P'+str(int(v_['2N']))], 0.0)
        self.xupper = jtu.np_like_set(self.xupper, ix_['P'+str(int(v_['2N']))], 0.0)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(0.0))
        v_['2NN'] = v_['NN']+v_['NN']
        v_['4NN'] = 4*v_['NN']
        for I in range(int(v_['2']),int(v_['4NN'])+1,int(v_['2'])):
            v_['RI'] = float(I)
            v_['IDX'] = v_['RI']*v_['DX/2']
            v_['XI'] = v_['XA']+v_['IDX']
            v_['LIN'] = 0.02*v_['XI']
            v_['PI0'] = 0.06+v_['LIN']
            if('P'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['P'+str(I)], float(v_['PI0']))
            else:
                self.y0  = (                       jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['P'+str(I)]),float(v_['PI0'])))
        v_['4NN+2'] = 2+v_['4NN']
        v_['8NN'] = 8*v_['NN']
        for I in range(int(v_['4NN+2']),int(v_['8NN'])+1,int(v_['2'])):
            v_['RI'] = float(I)
            v_['IDX'] = v_['RI']*v_['DX/2']
            v_['XI'] = v_['XA']+v_['IDX']
            v_['XISQ'] = v_['XI']*v_['XI']
            v_['-XISQ'] = -1.0*v_['XISQ']
            v_['1-XISQ'] = 1.0+v_['-XISQ']
            v_['PI0'] = jnp.sqrt(v_['1-XISQ'])
            if('P'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['P'+str(I)], float(v_['PI0']))
            else:
                self.y0  = (                       jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['P'+str(I)]),float(v_['PI0'])))
        v_['8NN+2'] = 2+v_['8NN']
        for I in range(int(v_['8NN+2']),int(v_['2N'])+1,int(v_['2'])):
            if('P'+str(I) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['P'+str(I)], float(0.0))
            else:
                self.y0  = (                       jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['P'+str(I)]),float(0.0)))
        for J in range(int(v_['1']),int(v_['2N-1'])+1,int(v_['2'])):
            v_['RJ'] = float(J)
            v_['JDX'] = v_['RJ']*v_['DX/2']
            v_['XJ'] = v_['XA']+v_['JDX']
            v_['XJSQ'] = v_['XJ']*v_['XJ']
            if('H'+str(J) in ix_):
                self.x0 = jtu.np_like_set(self.x0, ix_['H'+str(J)], float(v_['XJSQ']))
            else:
                self.y0  = (                       jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['H'+str(J)]),float(v_['XJSQ'])))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eREY', iet_)
        elftv = jtu.loaset(elftv,it,0,'PA')
        elftv = jtu.loaset(elftv,it,1,'PB')
        elftv = jtu.loaset(elftv,it,2,'H')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'A')
        [it,iet_,_] = jtu.s2mpj_ii( 'en2PR', iet_)
        elftv = jtu.loaset(elftv,it,0,'P')
        elftv = jtu.loaset(elftv,it,1,'R')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for J in range(int(v_['1']),int(v_['2N-1'])+1,int(v_['2'])):
            v_['I+'] = 1+J
            v_['I-'] = -1+J
            ename = 'ER'+str(J)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eREY')
            ielftype = jtu.arrset(ielftype,ie,iet_["eREY"])
            vname = 'P'+str(int(v_['I-']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
            posev = jnp.where(elftv[ielftype[ie]]=='PA')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'P'+str(int(v_['I+']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
            posev = jnp.where(elftv[ielftype[ie]]=='H')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'H'+str(J)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
            posev = jnp.where(elftv[ielftype[ie]]=='PB')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='A')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['ALPHA']))
        for I in range(int(v_['2']),int(v_['2N-2'])+1,int(v_['2'])):
            ename = 'EC'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'en2PR')
            ielftype = jtu.arrset(ielftype,ie,iet_["en2PR"])
            vname = 'P'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
            posev = jnp.where(elftv[ielftype[ie]]=='P')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'R'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
            posev = jnp.where(elftv[ielftype[ie]]=='R')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['2']),int(v_['2N-2'])+1,int(v_['2'])):
            ig = ig_['COMPL']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EC'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        for I in range(int(v_['2']),int(v_['2N-2'])+1,int(v_['2'])):
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            ig = ig_['DR'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['ER'+str(int(v_['I-1']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/DX2']))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['ER'+str(int(v_['I+1']))])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-1/DX2']))
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
        self.pbclass   = "C-CQOR2-MN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def en2PR(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, EV_[1])
            g_ = jtu.np_like_set(g_, 1, EV_[0])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eREY(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        HA = -0.5*self.elpar[iel_][0]
        EARG = HA*(EV_[0]+EV_[1])
        E = jnp.exp(EARG)
        PAMPB = EV_[0]-EV_[1]
        T1 = PAMPB*HA+1.0
        T2 = PAMPB*HA-1.0
        HSQ = EV_[2]*EV_[2]
        HCB = HSQ*EV_[2]
        f_   = PAMPB*HCB*E
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, T1*HCB*E)
            g_ = jtu.np_like_set(g_, 1, T2*HCB*E)
            g_ = jtu.np_like_set(g_, 2, 3.0*PAMPB*HSQ*E)
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), HCB*E*HA*(T1+1.0))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), HCB*E*HA*(T1-1.0))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), 3.0*T1*HSQ*E)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), HCB*E*HA*(T2-1.0))
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), 3.0*T2*HSQ*E)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), 6.0*EV_[2]*PAMPB*E)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


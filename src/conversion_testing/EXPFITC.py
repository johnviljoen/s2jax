from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class EXPFITC:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : EXPFITC
#    *********
# 
#    One sided rational approximation to the exponential function, as
#    described by Powell.
# 
#    Source:
#    M.J.D. Powell,
#    "A tolerant algorithm for linearly constrained optimization
#    calculations"'
#    Mathematical Programming 45(3), pp.547--566, 1989.
# 
#    SIF input: Ph. Toint and N. Gould, May 1990.
# 
#    classification = "C-COLR2-AN-5-502"
# 
#    Number of fitting points
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'EXPFITC'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['R'] = 251
        v_['1'] = 1
        v_['5.0'] = 5.0
        v_['R-1'] = -1+v_['R']
        v_['RR-1'] = float(v_['R-1'])
        v_['5/R-1'] = v_['5.0']/v_['RR-1']
        for I in range(int(v_['1']),int(v_['R'])+1):
            v_['I-1'] = -1+I
            v_['RI-1'] = float(v_['I-1'])
            v_['T'+str(I)] = v_['RI-1']*v_['5/R-1']
            v_['ET'+str(I)] = jnp.exp(v_['T'+str(I)])
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        [iv,ix_,_] = jtu.s2mpj_ii('P0',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'P0')
        [iv,ix_,_] = jtu.s2mpj_ii('P1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'P1')
        [iv,ix_,_] = jtu.s2mpj_ii('P2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'P2')
        [iv,ix_,_] = jtu.s2mpj_ii('Q1',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q1')
        [iv,ix_,_] = jtu.s2mpj_ii('Q2',ix_)
        self.xnames=jtu.arrset(self.xnames,iv,'Q2')
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        for I in range(int(v_['1']),int(v_['R'])+1):
            v_['TM5'] = -5.0+v_['T'+str(I)]
            v_['TM5SQ'] = v_['TM5']*v_['TM5']
            v_['QC1'] = v_['TM5']*v_['ET'+str(I)]
            v_['QC2'] = v_['TM5SQ']*v_['ET'+str(I)]
            v_['-QC1'] = -1.0*v_['QC1']
            v_['-QC2'] = -1.0*v_['QC2']
            v_['2T'] = v_['T'+str(I)]*v_['T'+str(I)]
            [ig,ig_,_] = jtu.s2mpj_ii('C'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'C'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['P0']])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['P1']])
            valA = jtu.append(valA,float(v_['T'+str(I)]))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['P2']])
            valA = jtu.append(valA,float(v_['2T']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Q1']])
            valA = jtu.append(valA,float(v_['-QC1']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Q2']])
            valA = jtu.append(valA,float(v_['-QC2']))
            [ig,ig_,_] = jtu.s2mpj_ii('B'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'B'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Q1']])
            valA = jtu.append(valA,float(v_['TM5']))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['Q2']])
            valA = jtu.append(valA,float(v_['TM5SQ']))
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
        for I in range(int(v_['1']),int(v_['R'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['C'+str(I)],float(v_['ET'+str(I)]))
            self.gconst = jtu.arrset(self.gconst,ig_['B'+str(I)],float(-0.99999))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        if('P0' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['P0'], float(1.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['P0']),float(1.0)))
        if('P1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['P1'], float(1.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['P1']),float(1.0)))
        if('P2' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['P2'], float(6.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['P2']),float(6.0)))
        if('Q1' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['Q1'], float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['Q1']),float(0.0)))
        if('Q2' in ix_):
            self.x0 = jtu.np_like_set(self.x0, ix_['Q2'], float(0.0))
        else:
            self.y0  = (                   jtu.arrset(self.y0,findfirst(self.congrps,lambda x:x==ig_['Q2']),float(0.0)))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eFIT', iet_)
        elftv = jtu.loaset(elftv,it,0,'P0')
        elftv = jtu.loaset(elftv,it,1,'P1')
        elftv = jtu.loaset(elftv,it,2,'P2')
        elftv = jtu.loaset(elftv,it,3,'Q1')
        elftv = jtu.loaset(elftv,it,4,'Q2')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'T')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for I in range(int(v_['1']),int(v_['R'])+1):
            ename = 'F'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'eFIT')
            ielftype = jtu.arrset(ielftype,ie,iet_["eFIT"])
            vname = 'P0'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='P0')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'P1'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='P1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'P2'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='P2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'Q1'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Q1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'Q2'
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Q2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            posep = jnp.where(elftp[ielftype[ie]]=='T')[0]
            self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['T'+str(I)]))
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['1']),int(v_['R'])+1):
            ig = ig_['OBJ']
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['F'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
        self.objlower = 0.0
#    Solution
        pass
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-COLR2-AN-5-502"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eFIT(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        TM5 = self.elpar[iel_][0]-5.0
        TM5SQ = TM5*TM5
        T2 = self.elpar[iel_][0]*self.elpar[iel_][0]
        ET = jnp.exp(self.elpar[iel_][0])
        QT = 1.0+EV_[3]*TM5+EV_[4]*TM5SQ
        ETQT = ET*QT
        ETQT2 = ETQT*QT
        ETQT3 = ETQT2*QT
        PT = EV_[0]+EV_[1]*self.elpar[iel_][0]+EV_[2]*T2
        F = PT/ETQT-1.0
        TWOF = F+F
        DFDP0 = 1.0/ETQT
        DFDP1 = self.elpar[iel_][0]/ETQT
        DFDP2 = T2/ETQT
        DFDQ1 = -PT*TM5/ETQT2
        DFDQ2 = -PT*TM5SQ/ETQT2
        D2P0Q1 = -TM5/ETQT2
        D2P0Q2 = -TM5SQ/ETQT2
        D2P1Q1 = -self.elpar[iel_][0]*TM5/ETQT2
        D2P1Q2 = -self.elpar[iel_][0]*TM5SQ/ETQT2
        D2P2Q1 = -T2*TM5/ETQT2
        D2P2Q2 = -T2*TM5SQ/ETQT2
        D2Q1Q1 = 2.0*PT*TM5SQ/ETQT3
        D2Q1Q2 = 2.0*PT*TM5SQ*TM5/ETQT3
        D2Q2Q2 = 2.0*PT*TM5SQ*TM5SQ/ETQT3
        f_   = F*F
        if not isinstance( f_, float ):
            f_   = f_.item()
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, TWOF*DFDP0)
            g_ = jtu.np_like_set(g_, 1, TWOF*DFDP1)
            g_ = jtu.np_like_set(g_, 2, TWOF*DFDP2)
            g_ = jtu.np_like_set(g_, 3, TWOF*DFDQ1)
            g_ = jtu.np_like_set(g_, 4, TWOF*DFDQ2)
            if nargout>2:
                H_ = jnp.zeros((5,5))
                H_ = jtu.np_like_set(H_, jnp.array([0,0]), 2.0*DFDP0*DFDP0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 2.0*DFDP0*DFDP1)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), 2.0*DFDP0*DFDP2)
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), 2.0*DFDP1*DFDP1)
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), 2.0*DFDP1*DFDP2)
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), 2.0*DFDP2*DFDP2)
                H_ = jtu.np_like_set(H_, jnp.array([0,3]), TWOF*D2P0Q1+2.0*DFDP0*DFDQ1)
                H_ = jtu.np_like_set(H_, jnp.array([3,0]), H_[0,3])
                H_ = jtu.np_like_set(H_, jnp.array([0,4]), TWOF*D2P0Q2+2.0*DFDP0*DFDQ2)
                H_ = jtu.np_like_set(H_, jnp.array([4,0]), H_[0,4])
                H_ = jtu.np_like_set(H_, jnp.array([1,3]), TWOF*D2P1Q1+2.0*DFDP1*DFDQ1)
                H_ = jtu.np_like_set(H_, jnp.array([3,1]), H_[1,3])
                H_ = jtu.np_like_set(H_, jnp.array([1,4]), TWOF*D2P1Q2+2.0*DFDP1*DFDQ2)
                H_ = jtu.np_like_set(H_, jnp.array([4,1]), H_[1,4])
                H_ = jtu.np_like_set(H_, jnp.array([2,3]), TWOF*D2P2Q1+2.0*DFDP2*DFDQ1)
                H_ = jtu.np_like_set(H_, jnp.array([3,2]), H_[2,3])
                H_ = jtu.np_like_set(H_, jnp.array([2,4]), TWOF*D2P2Q2+2.0*DFDP2*DFDQ2)
                H_ = jtu.np_like_set(H_, jnp.array([4,2]), H_[2,4])
                H_ = jtu.np_like_set(H_, jnp.array([3,3]), TWOF*D2Q1Q1+2.0*DFDQ1*DFDQ1)
                H_ = jtu.np_like_set(H_, jnp.array([3,4]), TWOF*D2Q1Q2+2.0*DFDQ1*DFDQ2)
                H_ = jtu.np_like_set(H_, jnp.array([4,3]), H_[3,4])
                H_ = jtu.np_like_set(H_, jnp.array([4,4]), TWOF*D2Q2Q2+2.0*DFDQ2*DFDQ2)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


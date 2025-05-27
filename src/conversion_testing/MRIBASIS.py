from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class MRIBASIS:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem:
#    ******** 
# 
#    An optmization problem arising in the design of medical apparatus.
# 
#    Source:
#    Contribution from a LANCELOT user.
# 
#    SIF input: Arie Quist, TU Delft (NL), 1994.
#    Adaptation for CUTE: Ph. Toint, November 1994.
# 
#    classification = "C-CLOR2-MY-36-55"
# 
#    useful constants
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'MRIBASIS'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['3'] = 3
        v_['R2'] = float(v_['2'])
        v_['k1'] = 22.8443
        v_['k2'] = 12.4402
        v_['k3'] = 5.23792
        v_['k4'] = 5.12238
        v_['k5'] = 6.44999
        v_['k6'] = 5.32383
        v_['k7'] = 0.58392
        v_['k8'] = 3.94584
        v_['k9'] = -2.75584
        v_['k10'] = 32.0669
        v_['k11'] = 18.2179
        v_['k12'] = 31.7496
        v_['S1,1'] = -0.377126
        v_['S1,2'] = 0.919679
        v_['S1,3'] = 0.109389
        v_['S2,1'] = 0.634857
        v_['S2,2'] = 0.170704
        v_['S2,3'] = 0.753536
        v_['S3,1'] = 0.674338
        v_['S3,2'] = 0.353624
        v_['S3,3'] = -0.648242
        v_['xlo1'] = 1*v_['k7']
        v_['k10/2'] = v_['k10']/v_['R2']
        v_['k8/2'] = v_['k8']/v_['R2']
        v_['xup1'] = v_['k10/2']-v_['k8/2']
        v_['xlo2'] = v_['k10/2']+v_['k8/2']
        v_['k13'] = v_['k7']*v_['k5']
        v_['k14'] = v_['k8/2']*v_['k6']
        v_['xup2'] = 1*v_['k12']
        v_['-k1'] = -1*v_['k1']
        v_['-k2'] = -1*v_['k2']
        v_['k4-'] = v_['k4']-v_['k14']
        v_['-k3'] = -1*v_['k3']
        v_['-S1,1'] = -1*v_['S1,1']
        v_['-S1,2'] = -1*v_['S1,2']
        v_['-S1,3'] = -1*v_['S1,3']
        v_['-S2,1'] = -1*v_['S2,1']
        v_['-S2,2'] = -1*v_['S2,2']
        v_['-S2,3'] = -1*v_['S2,3']
        v_['-S3,1'] = -1*v_['S3,1']
        v_['-S3,2'] = -1*v_['S3,2']
        v_['-S3,3'] = -1*v_['S3,3']
        v_['2S1,1'] = 2*v_['S1,1']
        v_['2S1,2'] = 2*v_['S1,2']
        v_['2S1,3'] = 2*v_['S1,3']
        v_['2S2,1'] = 2*v_['S2,1']
        v_['2S2,2'] = 2*v_['S2,2']
        v_['2S2,3'] = 2*v_['S2,3']
        v_['2S3,1'] = 2*v_['S3,1']
        v_['2S3,2'] = 2*v_['S3,2']
        v_['2S3,3'] = 2*v_['S3,3']
        v_['-2S1,1'] = -2*v_['S1,1']
        v_['-2S1,2'] = -2*v_['S1,2']
        v_['-2S1,3'] = -2*v_['S1,3']
        v_['-2S2,1'] = -2*v_['S2,1']
        v_['-2S2,2'] = -2*v_['S2,2']
        v_['-2S2,3'] = -2*v_['S2,3']
        v_['-2S3,1'] = -2*v_['S3,1']
        v_['-2S3,2'] = -2*v_['S3,2']
        v_['-2S3,3'] = -2*v_['S3,3']
        v_['Llo1,1'] = v_['S1,1']*v_['k5']
        v_['Llo1,2'] = v_['S1,2']*v_['k5']
        v_['Llo1,3'] = v_['S1,3']*v_['k5']
        v_['Lup1,1'] = v_['S1,1']*v_['k6']
        v_['Lup1,2'] = v_['S1,2']*v_['k6']
        v_['Lup1,3'] = v_['S1,3']*v_['k6']
        v_['Llo2,1'] = v_['S1,1']*v_['k6']
        v_['Llo2,2'] = v_['S1,2']*v_['k6']
        v_['Llo2,3'] = v_['S1,3']*v_['k6']
        v_['4'] = 4
        v_['xm'] = 6
        v_['Lm'] = 4
        v_['xm-'] = -1+v_['xm']
        v_['xm-2'] = -2+v_['xm']
        v_['Lm-'] = -1+v_['Lm']
        v_['Lm-2'] = -2+v_['Lm']
        v_['R12'] = 12
        v_['1/12'] = 1/v_['R12']
        v_['-1/12'] = -1*v_['1/12']
        v_['R0'] = float(v_['0'])
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for j in range(int(v_['1']),int(v_['2'])+1):
            for k in range(int(v_['1']),int(v_['xm'])+1):
                [iv,ix_,_] = jtu.s2mpj_ii('X'+str(j)+','+str(k),ix_)
                self.xnames=jtu.arrset(self.xnames,iv,'X'+str(j)+','+str(k))
        for i in range(int(v_['1']),int(v_['3'])+1):
            for j in range(int(v_['1']),int(v_['2'])+1):
                for k in range(int(v_['1']),int(v_['Lm'])+1):
                    [iv,ix_,_] = jtu.s2mpj_ii('L'+str(i)+','+str(j)+','+str(k),ix_)
                    self.xnames=jtu.arrset(self.xnames,iv,'L'+str(i)+','+str(j)+','+str(k))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('Object',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['2']))+','+str(int(v_['xm']))]])
        valA = jtu.append(valA,float(1))
        for j in range(int(v_['1']),int(v_['2'])+1):
            for k in range(int(v_['2']),int(v_['xm-2'])+1):
                v_['k+'] = 1+k
                [ig,ig_,_] = jtu.s2mpj_ii('PS'+str(j)+','+str(k),ig_)
                gtype = jtu.arrset(gtype,ig,'>=')
                cnames = jtu.arrset(cnames,ig,'PS'+str(j)+','+str(k))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(j)+','+str(int(v_['k+']))]])
                valA = jtu.append(valA,float(1))
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['X'+str(j)+','+str(k)]])
                valA = jtu.append(valA,float(-1))
        [ig,ig_,_] = jtu.s2mpj_ii('PL',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'PL')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['2']))+','+str(int(v_['xm']))]])
        valA = jtu.append(valA,float(1))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X'+str(int(v_['2']))+','+str(int(v_['xm-']))]])
        valA = jtu.append(valA,float(-1))
        for i in range(int(v_['1']),int(v_['3'])+1):
            for j in range(int(v_['1']),int(v_['2'])+1):
                for k in range(int(v_['1']),int(v_['Lm-'])+1):
                    v_['k+'] = 1+k
                    v_['2k'] = 2*k
                    v_['2k-'] = -1+v_['2k']
                    [ig,ig_,_] = jtu.s2mpj_ii('SU'+str(i)+','+str(j)+','+str(k),ig_)
                    gtype = jtu.arrset(gtype,ig,'<=')
                    cnames = jtu.arrset(cnames,ig,'SU'+str(i)+','+str(j)+','+str(k))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['L'+str(i)+','+str(j)+','+str(int(v_['k+']))]])
                    valA = jtu.append(valA,float(1))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['L'+str(i)+','+str(j)+','+str(k)]])
                    valA = jtu.append(valA,float(-1))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(j)+','+str(int(v_['2k']))]])
                    valA = jtu.append(valA,float(v_['-k1']))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(j)+','+str(int(v_['2k-']))]])
                    valA = jtu.append(valA,float(v_['k1']))
                    [ig,ig_,_] = jtu.s2mpj_ii('SL'+str(i)+','+str(j)+','+str(k),ig_)
                    gtype = jtu.arrset(gtype,ig,'>=')
                    cnames = jtu.arrset(cnames,ig,'SL'+str(i)+','+str(j)+','+str(k))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['L'+str(i)+','+str(j)+','+str(int(v_['k+']))]])
                    valA = jtu.append(valA,float(1))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['L'+str(i)+','+str(j)+','+str(k)]])
                    valA = jtu.append(valA,float(-1))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(j)+','+str(int(v_['2k']))]])
                    valA = jtu.append(valA,float(v_['k1']))
                    irA  = jtu.append(irA,[ig])
                    icA  = jtu.append(icA,[ix_['X'+str(j)+','+str(int(v_['2k-']))]])
                    valA = jtu.append(valA,float(v_['-k1']))
        for i in range(int(v_['1']),int(v_['3'])+1):
            for k in range(int(v_['2']),int(v_['Lm-'])+1):
                [ig,ig_,_] = jtu.s2mpj_ii('cc1',ig_)
                gtype = jtu.arrset(gtype,ig,'==')
                cnames = jtu.arrset(cnames,ig,'cc1')
                irA  = jtu.append(irA,[ig])
                icA  = jtu.append(icA,[ix_['L'+str(i)+','+str(int(v_['2']))+','+str(k)]])
                valA = jtu.append(valA,float(v_['S'+str(int(v_['3']))+','+str(i)]))
        for i in range(int(v_['1']),int(v_['3'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('c2const'+str(i),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'c2const'+str(i))
            irA  = jtu.append(irA,[ig])
            icA   = (
                  jtu.append(icA,[ix_['L'+str(i)+','+str(int(v_['2']))+','+str(int(v_['Lm']))]]))
            valA = jtu.append(valA,float(v_['k10']))
        [ig,ig_,_] = jtu.s2mpj_ii('c3con1',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'c3con1')
        [ig,ig_,_] = jtu.s2mpj_ii('c3con2',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'c3con2')
        [ig,ig_,_] = jtu.s2mpj_ii('c4const',ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'c4const')
        for i in range(int(v_['1']),int(v_['3'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('c5con'+str(i),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'c5con'+str(i))
        for i in range(int(v_['1']),int(v_['2'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('c6cn'+str(i),ig_)
            gtype = jtu.arrset(gtype,ig,'==')
            cnames = jtu.arrset(cnames,ig,'c6cn'+str(i))
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
        v_['Opmr1'] = v_['k3']*v_['S2,1']
        v_['Opmr2'] = v_['k3']*v_['S2,2']
        v_['Opmr3'] = v_['k3']*v_['S2,3']
        for i in range(int(v_['1']),int(v_['3'])+1):
            self.gconst  = (
                  jtu.arrset(self.gconst,ig_['c2const'+str(i)],float(v_['Opmr'+str(i)])))
        self.gconst = jtu.arrset(self.gconst,ig_['c3con1'],float(v_['k4-']))
        self.gconst = jtu.arrset(self.gconst,ig_['c3con2'],float(v_['k13']))
        self.gconst = jtu.arrset(self.gconst,ig_['c5con1'],float(v_['k13']))
        self.gconst = jtu.arrset(self.gconst,ig_['c5con2'],float(v_['-k3']))
        self.gconst = jtu.arrset(self.gconst,ig_['c5con3'],float(v_['k9']))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        for j in range(int(v_['1']),int(v_['2'])+1):
            for k in range(int(v_['2']),int(v_['xm-'])+1):
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(j)+','+str(k)]]), v_['xlo'+str(j)])
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(j)+','+str(k)]]), v_['xup'+str(j)])
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(j)+','+str(int(v_['1']))]]), v_['xlo'+str(j)])
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(j)+','+str(int(v_['1']))]]), v_['xlo'+str(j)])
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['xm']))]]), v_['xup1'])
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['xm']))]]), v_['xup1'])
        self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['X'+str(int(v_['2']))+','+str(int(v_['xm']))]]), v_['k11'])
        self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['X'+str(int(v_['2']))+','+str(int(v_['xm']))]]), v_['k12'])
        for i in range(int(v_['1']),int(v_['3'])+1):
            for j in range(int(v_['1']),int(v_['2'])+1):
                for k in range(int(v_['2']),int(v_['Lm-'])+1):
                    self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['L'+str(i)+','+str(j)+','+str(k)]]), v_['-k2'])
                    self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['L'+str(i)+','+str(j)+','+str(k)]]), v_['k2'])
                self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['L'+str(i)+','+str(j)+','+str(int(v_['1']))]]), (v_['Llo'+)
                     str(j)+','+str(i)])
                self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['L'+str(i)+','+str(j)+','+str(int(v_['1']))]]), (v_['Llo'+)
                     str(j)+','+str(i)])
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['L'+str(i)+','+str(int(v_['1']))+','+str(int(v_['Lm']))]]), ()
                  v_['Lup'+str(int(v_['1']))+','+str(i)])
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['L'+str(i)+','+str(int(v_['1']))+','+str(int(v_['Lm']))]]), ()
                  v_['Lup'+str(int(v_['1']))+','+str(i)])
            self.xlower = jtu.np_like_set(self.xlower, jnp.array([ix_['L'+str(i)+','+str(int(v_['2']))+','+str(int(v_['Lm']))]]), ()
                  v_['-k2'])
            self.xupper = jtu.np_like_set(self.xupper, jnp.array([ix_['L'+str(i)+','+str(int(v_['2']))+','+str(int(v_['Lm']))]]), ()
                  v_['k2'])
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        v_['intlen1'] = v_['xup1']-v_['xlo1']
        v_['Rxm-'] = float(v_['xm-'])
        v_['dx1'] = v_['intlen1']/v_['Rxm-']
        v_['intlen2'] = v_['k11']-v_['xlo2']
        v_['dx2'] = v_['intlen2']/v_['Rxm-']
        for k in range(int(v_['1']),int(v_['xm-'])+1):
            v_['Rk'] = float(k)
            v_['dist1'] = v_['dx1']*v_['Rk']
            v_['strtv1'] = v_['xlo1']+v_['dist1']
            v_['dist2'] = v_['dx2']*v_['Rk']
            v_['strtv2'] = v_['xlo2']+v_['dist2']
            v_['k+'] = 1+k
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['1']))+','+str(int(v_['k+']))]]), ()
                  float(v_['strtv1']))
            self.x0 = jtu.np_like_set(self.x0, jnp.array([ix_['X'+str(int(v_['2']))+','+str(int(v_['k+']))]]), ()
                  float(v_['strtv2']))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'euv1', iet_)
        elftv = jtu.loaset(elftv,it,0,'v1')
        elftv = jtu.loaset(elftv,it,1,'v2')
        elftv = jtu.loaset(elftv,it,2,'v3')
        elftv = jtu.loaset(elftv,it,3,'v4')
        [it,iet_,_] = jtu.s2mpj_ii( 'euv2', iet_)
        elftv = jtu.loaset(elftv,it,0,'v1')
        elftv = jtu.loaset(elftv,it,1,'v2')
        elftv = jtu.loaset(elftv,it,2,'v3')
        [it,iet_,_] = jtu.s2mpj_ii( 'euvw1', iet_)
        elftv = jtu.loaset(elftv,it,0,'v1')
        elftv = jtu.loaset(elftv,it,1,'v2')
        elftv = jtu.loaset(elftv,it,2,'v3')
        elftp = []
        elftp = jtu.loaset(elftp,it,0,'p1')
        [it,iet_,_] = jtu.s2mpj_ii( 'emo', iet_)
        elftv = jtu.loaset(elftv,it,0,'s1')
        elftv = jtu.loaset(elftv,it,1,'s2')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        self.elpar   = []
        for i in range(int(v_['1']),int(v_['3'])+1):
            for j in range(int(v_['1']),int(v_['2'])+1):
                for k in range(int(v_['1']),int(v_['Lm-'])+1):
                    v_['2k'] = 2*k
                    v_['k+'] = 1+k
                    v_['2k-'] = -1+v_['2k']
                    ename = 'e1'+str(i)+','+str(j)+','+str(k)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'euv1')
                    ielftype = jtu.arrset(ielftype,ie,iet_["euv1"])
                    vname = 'X'+str(j)+','+str(int(v_['2k']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v1')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(j)+','+str(int(v_['2k-']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v2')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'L'+str(i)+','+str(j)+','+str(int(v_['k+']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v3')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'L'+str(i)+','+str(j)+','+str(k)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v4')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    ename = 'e3'+str(i)+','+str(j)+','+str(k)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'euvw1')
                    ielftype = jtu.arrset(ielftype,ie,iet_["euvw1"])
                    vname = 'L'+str(i)+','+str(j)+','+str(k)
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v1')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(j)+','+str(int(v_['2k']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v2')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(j)+','+str(int(v_['2k-']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v3')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    posep = jnp.where(elftp[ielftype[ie]]=='p1')[0]
                    self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['1/12']))
                    ename = 'e5'+str(i)+','+str(j)+','+str(k)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'euvw1')
                    ielftype = jtu.arrset(ielftype,ie,iet_["euvw1"])
                    vname = 'L'+str(i)+','+str(j)+','+str(int(v_['k+']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v1')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(j)+','+str(int(v_['2k']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v2')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(j)+','+str(int(v_['2k-']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v3')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    posep = jnp.where(elftp[ielftype[ie]]=='p1')[0]
                    self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['-1/12']))
                for k in range(int(v_['1']),int(v_['Lm-2'])+1):
                    v_['2k'] = 2*k
                    v_['k+'] = 1+k
                    v_['2k+'] = 1+v_['2k']
                    ename = 'e2'+str(i)+','+str(j)+','+str(k)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'euv2')
                    ielftype = jtu.arrset(ielftype,ie,iet_["euv2"])
                    vname = 'X'+str(j)+','+str(int(v_['2k+']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v1')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(j)+','+str(int(v_['2k']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v2')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'L'+str(i)+','+str(j)+','+str(int(v_['k+']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v3')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    ename = 'e4'+str(i)+','+str(j)+','+str(k)
                    [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
                    self.elftype = jtu.arrset(self.elftype,ie,'euvw1')
                    ielftype = jtu.arrset(ielftype,ie,iet_["euvw1"])
                    vname = 'L'+str(i)+','+str(j)+','+str(int(v_['k+']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v1')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(j)+','+str(int(v_['2k+']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v2')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    vname = 'X'+str(j)+','+str(int(v_['2k']))
                    [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
                    posev = jnp.where(elftv[ielftype[ie]]=='v3')[0]
                    self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
                    posep = jnp.where(elftp[ielftype[ie]]=='p1')[0]
                    self.elpar = jtu.loaset(self.elpar,ie,posep[0],float(v_['R0']))
        for i in range(int(v_['1']),int(v_['3'])+1):
            ename = 'factr'+str(i)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'emo')
            ielftype = jtu.arrset(ielftype,ie,iet_["emo"])
            vname = 'X'+str(int(v_['2']))+','+str(int(v_['xm']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='s1')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'L'+str(i)+','+str(int(v_['2']))+','+str(int(v_['Lm']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='s2')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for i in range(int(v_['1']),int(v_['3'])+1):
            ig = ig_['c2const'+str(i)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['factr'+str(i)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        for j in range(int(v_['1']),int(v_['2'])+1):
            for i in range(int(v_['1']),int(v_['3'])+1):
                for k in range(int(v_['1']),int(v_['Lm-'])+1):
                    ig = ig_['c3con'+str(j)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e1'+str(i)+','+str(int(v_['2']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw  = (
                          jtu.loaset(self.grelw,ig,posel,float(v_['S'+str(int(v_['1']))+','+str(i)])))
                for k in range(int(v_['1']),int(v_['Lm-2'])+1):
                    ig = ig_['c3con'+str(j)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e2'+str(i)+','+str(int(v_['2']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw  = (
                          jtu.loaset(self.grelw,ig,posel,float(v_['S'+str(int(v_['1']))+','+str(i)])))
        for i in range(int(v_['1']),int(v_['3'])+1):
            for k in range(int(v_['1']),int(v_['Lm-'])+1):
                ig = ig_['c4const']
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['e1'+str(i)+','+str(int(v_['2']))+','+str(k)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw  = (
                      jtu.loaset(self.grelw,ig,posel,float(v_['S'+str(int(v_['2']))+','+str(i)])))
            for k in range(int(v_['1']),int(v_['Lm-2'])+1):
                ig = ig_['c4const']
                posel = len(self.grelt[ig])
                self.grelt  = (
                      jtu.loaset(self.grelt,ig,posel,ie_['e2'+str(i)+','+str(int(v_['2']))+','+str(k)]))
                nlc = jnp.union1d(nlc,jnp.array([ig]))
                self.grelw  = (
                      jtu.loaset(self.grelw,ig,posel,float(v_['S'+str(int(v_['2']))+','+str(i)])))
        for j in range(int(v_['1']),int(v_['3'])+1):
            for i in range(int(v_['1']),int(v_['3'])+1):
                for k in range(int(v_['1']),int(v_['Lm-'])+1):
                    ig = ig_['c5con'+str(j)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e1'+str(i)+','+str(int(v_['1']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-S'+str(j)+','+str(i)]))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e1'+str(i)+','+str(int(v_['2']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['S'+str(j)+','+str(i)]))
                for k in range(int(v_['1']),int(v_['Lm-2'])+1):
                    ig = ig_['c5con'+str(j)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e2'+str(i)+','+str(int(v_['1']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-S'+str(j)+','+str(i)]))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e2'+str(i)+','+str(int(v_['2']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['S'+str(j)+','+str(i)]))
        for j in range(int(v_['1']),int(v_['2'])+1):
            for i in range(int(v_['1']),int(v_['3'])+1):
                for k in range(int(v_['1']),int(v_['Lm-'])+1):
                    ig = ig_['c6cn'+str(j)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e3'+str(i)+','+str(int(v_['1']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['S'+str(j)+','+str(i)]))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e5'+str(i)+','+str(int(v_['1']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['S'+str(j)+','+str(i)]))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e3'+str(i)+','+str(int(v_['2']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-S'+str(j)+','+str(i)]))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e5'+str(i)+','+str(int(v_['2']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-S'+str(j)+','+str(i)]))
                for k in range(int(v_['1']),int(v_['Lm-2'])+1):
                    ig = ig_['c6cn'+str(j)]
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e4'+str(i)+','+str(int(v_['1']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['2S'+str(j)+','+str(i)]))
                    posel = len(self.grelt[ig])
                    self.grelt  = (
                          jtu.loaset(self.grelt,ig,posel,ie_['e4'+str(i)+','+str(int(v_['2']))+','+str(k)]))
                    nlc = jnp.union1d(nlc,jnp.array([ig]))
                    self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-2S'+str(j)+','+str(i)]))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               18.2179000000
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.array([jnp.arange(self.nle,self.nle+self.neq)]), jnp.zeros((self.neq,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (
              jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CLOR2-MY-36-55"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def euv1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,4))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]-1)
        U_ = jtu.np_like_set(U_, jnp.array([1,2]), U_[1,2]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,3]), U_[1,3]+1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        f_   = 0.5e0*IV_[0]*IV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, 0.5e0*IV_[1])
            g_ = jtu.np_like_set(g_, 1, 0.5e0*IV_[0])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 0.5e0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def euv2(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,3))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,0]), U_[0,0]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]-1)
        U_ = jtu.np_like_set(U_, jnp.array([1,2]), U_[1,2]+1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        f_   = IV_[0]*IV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, IV_[1])
            g_ = jtu.np_like_set(g_, 1, IV_[0])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), 1.0e0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def euvw1(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        d14 = 0.25e0
        f_    = (
              EV_[0]*(EV_[1]-EV_[2])*((d14+self.elpar[iel_][0])*EV_[1]+(d14-self.elpar[iel_][0])*EV_[2]))
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, ()
                  (EV_[1]-EV_[2])*((d14+self.elpar[iel_][0])*EV_[1]+(d14-self.elpar[iel_][0])*EV_[2]))
            g_ = jtu.np_like_set(g_, 1, ()
                  EV_[0]*2.0e0*(EV_[1]*(d14+self.elpar[iel_][0])-self.elpar[iel_][0]*EV_[2]))
            g_ = jtu.np_like_set(g_, 2, ()
                  EV_[0]*2.0e0*(-EV_[2]*(d14-self.elpar[iel_][0])-self.elpar[iel_][0]*EV_[1]))
            if nargout>2:
                H_ = jnp.zeros((3,3))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), ()
                      2.0e0*(EV_[1]*(d14+self.elpar[iel_][0])-self.elpar[iel_][0]*EV_[2]))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
                H_ = jtu.np_like_set(H_, jnp.array([0,2]), ()
                      2.0e0*(-EV_[2]*(d14-self.elpar[iel_][0])-self.elpar[iel_][0]*EV_[1]))
                H_ = jtu.np_like_set(H_, jnp.array([2,0]), H_[0,2])
                H_ = jtu.np_like_set(H_, jnp.array([1,1]), EV_[0]*2.0e0*(d14+self.elpar[iel_][0]))
                H_ = jtu.np_like_set(H_, jnp.array([1,2]), -EV_[0]*2.0e0*self.elpar[iel_][0])
                H_ = jtu.np_like_set(H_, jnp.array([2,1]), H_[1,2])
                H_ = jtu.np_like_set(H_, jnp.array([2,2]), -EV_[0]*2.0e0*(d14-self.elpar[iel_][0]))
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def emo(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = -EV_[1]*EV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 0, -EV_[1])
            g_ = jtu.np_like_set(g_, 1, -EV_[0])
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), -1.0e0)
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), H_[0,1])
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


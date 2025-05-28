import jax.numpy as jnp
import s2jax.sparse_utils as spu
from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class CAMSHAPE:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : CAMSHAPE
#    *********
# 
#    Maximize the area of the valve opening for one rotation of a convex cam 
#    with constraints on the curvature and on the radius of the cam
# 
#    This is problem 4 in the COPS (Version 2) collection of 
#    E. Dolan and J. More'
#    see "Benchmarking Optimization Software with COPS"
#    Argonne National Labs Technical Report ANL/MCS-246 (2000)
# 
#    SIF input: Nick Gould, November 2000
# 
#    classification = "C-CLOR2-AN-V-V"
# 
#    The number of discretization points
# 
#           Alternative values for the SIF file parameters:
# IE N                   100            $-PARAMETER
# IE N                   200            $-PARAMETER
# IE N                   400            $-PARAMETER
# IE N                   800            $-PARAMETER
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'CAMSHAPE'

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
        v_['RV'] = 1.0
        v_['RMAX'] = 2.0
        v_['RMIN'] = 1.0
        v_['RAV'] = v_['RMIN']+v_['RMAX']
        v_['RAV'] = 0.5*v_['RAV']
        v_['PI/4'] = jnp.arctan(1.0)
        v_['PI'] = 4.0*v_['PI/4']
        v_['ALPHA'] = 1.5
        v_['N+1'] = 1+v_['N']
        v_['5(N+1)'] = 5*v_['N+1']
        v_['5(N+1)'] = float(v_['5(N+1)'])
        v_['DTHETA'] = 2.0*v_['PI']
        v_['DTHETA'] = v_['DTHETA']/v_['5(N+1)']
        v_['0'] = 0
        v_['1'] = 1
        v_['2'] = 2
        v_['N-1'] = -1+v_['N']
        v_['RN'] = float(v_['N'])
        v_['PIRV'] = v_['PI']*v_['RV']
        v_['PIRV/N'] = v_['PIRV']/v_['RN']
        v_['-PIRV/N'] = -1.0*v_['PIRV/N']
        v_['CDTHETA'] = jnp.cos(v_['DTHETA'])
        v_['2CDTHETA'] = 2.0*v_['CDTHETA']
        v_['ADTHETA'] = v_['ALPHA']*v_['DTHETA']
        v_['-ADTHETA'] = -1.0*v_['ADTHETA']
        v_['2ADTHETA'] = 2.0*v_['ADTHETA']
        v_['-RMIN'] = -1.0*v_['RMIN']
        v_['-RMAX'] = -1.0*v_['RMAX']
        v_['-2RMAX'] = -2.0*v_['RMAX']
        v_['RMIN2'] = v_['RMIN']*v_['RMIN']
        v_['RMIN2CD'] = v_['RMIN']*v_['2CDTHETA']
        v_['RMAX2CD'] = v_['RMAX']*v_['2CDTHETA']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['N'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('R'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'R'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        for I in range(int(v_['1']),int(v_['N'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('AREA',ig_)
            gtype = jtu.arrset(gtype,ig,'<>')
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['R'+str(I)]])
            valA = jtu.append(valA,float(v_['-PIRV/N']))
        for I in range(int(v_['2']),int(v_['N-1'])+1):
            [ig,ig_,_] = jtu.s2mpj_ii('CO'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'<=')
            cnames = jtu.arrset(cnames,ig,'CO'+str(I))
        [ig,ig_,_] = jtu.s2mpj_ii('E1',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'E1')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['R'+str(int(v_['1']))]])
        valA = jtu.append(valA,float(v_['-RMIN']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['R'+str(int(v_['2']))]])
        valA = jtu.append(valA,float(v_['RMIN2CD']))
        v_['R'] = v_['RMIN2CD']-v_['RMIN']
        [ig,ig_,_] = jtu.s2mpj_ii('E2',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'E2')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['R'+str(int(v_['1']))]])
        valA = jtu.append(valA,float(v_['R']))
        [ig,ig_,_] = jtu.s2mpj_ii('E3',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'E3')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['R'+str(int(v_['N']))]])
        valA = jtu.append(valA,float(v_['-RMAX']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['R'+str(int(v_['N-1']))]])
        valA = jtu.append(valA,float(v_['RMAX2CD']))
        [ig,ig_,_] = jtu.s2mpj_ii('E4',ig_)
        gtype = jtu.arrset(gtype,ig,'<=')
        cnames = jtu.arrset(cnames,ig,'E4')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['R'+str(int(v_['N']))]])
        valA = jtu.append(valA,float(v_['-2RMAX']))
        [ig,ig_,_] = jtu.s2mpj_ii('CU'+str(int(v_['0'])),ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'CU'+str(int(v_['0'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['R'+str(int(v_['1']))]])
        valA = jtu.append(valA,float(1.0))
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            v_['I+1'] = 1+I
            [ig,ig_,_] = jtu.s2mpj_ii('CU'+str(I),ig_)
            gtype = jtu.arrset(gtype,ig,'>=')
            cnames = jtu.arrset(cnames,ig,'CU'+str(I))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['R'+str(int(v_['I+1']))]])
            valA = jtu.append(valA,float(1.0))
            irA  = jtu.append(irA,[ig])
            icA  = jtu.append(icA,[ix_['R'+str(I)]])
            valA = jtu.append(valA,float(-1.0))
        [ig,ig_,_] = jtu.s2mpj_ii('CU'+str(int(v_['N'])),ig_)
        gtype = jtu.arrset(gtype,ig,'>=')
        cnames = jtu.arrset(cnames,ig,'CU'+str(int(v_['N'])))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['R'+str(int(v_['N']))]])
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
        #%%%%%%%%%%%%%%%%%% CONSTANTS %%%%%%%%%%%%%%%%%%%%%
        self.gconst = jnp.zeros((ngrp,1))
        self.gconst = jtu.arrset(self.gconst,ig_['E2'],float(v_['RMIN2']))
        v_['R'] = v_['-ADTHETA']+v_['RMIN']
        self.gconst = jtu.arrset(self.gconst,ig_['CU'+str(int(v_['0']))],float(v_['R']))
        for I in range(int(v_['1']),int(v_['N-1'])+1):
            self.gconst = jtu.arrset(self.gconst,ig_['CU'+str(I)],float(v_['-ADTHETA']))
        v_['R'] = v_['-ADTHETA']-v_['RMAX']
        self.gconst = jtu.arrset(self.gconst,ig_['CU'+str(int(v_['N']))],float(v_['R']))
        #%%%%%%%%%%%%%%%%%%%%  RANGES %%%%%%%%%%%%%%%%%%%%%%
        grange = jnp.full((ngrp,1),None)
        grange = jtu.np_like_set(grange, legrps, jnp.full((self.nle,1),float('inf')))
        grange = jtu.np_like_set(grange, gegrps, jnp.full((self.nge,1),float('inf')))
        for I in range(int(v_['0']),int(v_['N'])+1):
            grange = jtu.arrset(grange,ig_['CU'+str(I)],float(v_['2ADTHETA']))
        #%%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.zeros((self.n,1))
        self.xupper = jnp.full((self.n,1),float('inf'))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['R'+str(I)], v_['RMIN'])
            self.xupper = jtu.np_like_set(self.xupper, ix_['R'+str(I)], v_['RMAX'])
        #%%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.zeros((self.n,1))
        self.y0 = jnp.zeros((self.m,1))
        for I in range(int(v_['1']),int(v_['N'])+1):
            self.x0 = jtu.np_like_set(self.x0, ix_['R'+str(I)], float(v_['RAV']))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eSQR', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        [it,iet_,_] = jtu.s2mpj_ii( 'ePROD', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'Y')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        for I in range(int(v_['2']),int(v_['N-1'])+1):
            v_['I-1'] = -1+I
            v_['I+1'] = 1+I
            ename = 'RA'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
            ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
            vname = 'R'+str(I)
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'R'+str(int(v_['I-1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            ename = 'RB'+str(I)
            [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
            self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
            ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
            vname = 'R'+str(int(v_['I+1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
            vname = 'R'+str(int(v_['I-1']))
            [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
            posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
            self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'RA'+str(int(v_['N']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'ePROD')
        ielftype = jtu.arrset(ielftype,ie,iet_["ePROD"])
        ename = 'RA'+str(int(v_['N']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        vname = 'R'+str(int(v_['N']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'RA'+str(int(v_['N']))
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        vname = 'R'+str(int(v_['N-1']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='Y')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'R2'
        [ie,ie_,_] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eSQR')
        ielftype = jtu.arrset(ielftype,ie,iet_["eSQR"])
        vname = 'R'+str(int(v_['N']))
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,None)
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        for I in range(int(v_['2']),int(v_['N-1'])+1):
            v_['I+1'] = 1+I
            ig = ig_['CO'+str(I)]
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['RA'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
            posel = posel+1
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['RA'+str(int(v_['I+1']))])
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
            posel = len(self.grelt[ig])
            self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['RB'+str(I)])
            nlc = jnp.union1d(nlc,jnp.array([ig]))
            self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['2CDTHETA']))
        ig = ig_['E1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['RA'+str(int(v_['2']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['E3']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['RA'+str(int(v_['N']))])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['E4']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['R2'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['2CDTHETA']))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLUTION             -4.2841D+00   $ (NH=100)
# LO SOLUTION             -4.2785D+00   $ (NH=200)
# LO SOLUTION             -4.2757D+00   $ (NH=400)
# LO SOLUTION             -4.2743D+00   $ (NH=800)
        #%%%%%%%% BUILD THE SPARSE MATRICES %%%%%%%%%%%%%%%
        self.A = BCSR.from_bcoo(BCOO((valA, jnp.array((irA,icA)).T), shape=(ngrp,self.n)))
        #%%%%%%%% DEFAULT FOR MISSING SECTION(S) %%%%%%%%%%
        #%%%%%%%%%%%%% FORM clower AND cupper %%%%%%%%%%%%%
        self.clower = jnp.full((self.m,1),-float('Inf'))
        self.cupper = jnp.full((self.m,1),+float('Inf'))
        self.clower = jtu.np_like_set(self.clower, jnp.arange(self.nle), grange[legrps])
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nle), jnp.zeros((self.nle,1)))
        self.clower = jtu.np_like_set(self.clower, jnp.array([jnp.arange(self.nle+self.neq,self.m)]), jnp.zeros((self.nge,1)))
        self.cupper = jtu.np_like_set(self.cupper, jnp.arange(self.nge), grange[gegrps])
        #%%%% RETURN VALUES FROM THE __INIT__ METHOD %%%%%%
        self.lincons  = (               jnp.where(jnp.isin(self.congrps,jnp.setdiff1d(self.congrps,nlc)))[0])
        self.pbclass   = "C-CLOR2-AN-V-V"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def ePROD(self, nargout,*args):

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
    def eSQR(self, nargout,*args):

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


from jax.experimental.sparse import BCOO, BCSR
import s2jax.jax_utils as jtu
from s2jax.utils import *
class TENBARS3:

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 
# 
#    Problem : TENBARS3
#    *********
#    The "ten bar truss" structural optimization problem,
#    version P3.
# 
#    The problem is to minimize the cross section areas of the bars
#    in the structure
# 
#      /|
#      /|>o------o------o
#      /|  \    /|\    /|
#           \  / | \  / |
#            \/  |  \/  |
#            /\  |  /\  |
#           /  \ | /  \ |
#      /|  /    \|/    \|
#      /|>o------o------o
#      /|
# 
#    submitted to vertical forces of equal magnitude (P0) applied at
#    the two free lower nodes, subject to limits of nodal displacements.
# 
#    Source:
#    K. Svanberg,
#    private communication,  August 1990.
#    See also
#    K. Svanberg,
#    "On local and global minima in structural optimization",
#    in "New directions in optimum structural design" (Atrek, Ragsdell
#    and Zienkiwewicz, eds.), Wiley, 1984.
# 
#    SIF input: Ph. Toint, August 1990.
# 
#    classification = "C-CLOR2-MY-18-8"
# 
#    Constants
# 
# 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#   Translated to Python by S2MPJ version 25 XI 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    name = 'TENBARS3'

    def __init__(self, *args): 
        import jax.numpy as jnp
        from scipy.sparse import csr_matrix
        nargin   = len(args)

        #%%%%%%%%%%%%%%%%%%%  PREAMBLE %%%%%%%%%%%%%%%%%%%%
        v_  = {}
        ix_ = {}
        ig_ = {}
        v_['1'] = 1
        v_['2'] = 2
        v_['8'] = 8
        v_['10'] = 10
        v_['SQ2'] = jnp.sqrt(2.0)
        v_['SQ8'] = jnp.sqrt(8.0)
        v_['1/SQ8'] = 1.0/v_['SQ8']
        v_['-1/SQ8'] = -1.0*v_['1/SQ8']
        v_['C0'] = 2.53106
        v_['-P0'] = -589.884
        v_['C0SQ2'] = v_['C0']*v_['SQ2']
        #%%%%%%%%%%%%%%%%%%%  VARIABLES %%%%%%%%%%%%%%%%%%%%
        self.xnames = jnp.array([])
        self.xscale = jnp.array([])
        intvars   = jnp.array([])
        binvars   = jnp.array([])
        irA          = jnp.array([],dtype=int)
        icA          = jnp.array([],dtype=int)
        valA         = jnp.array([],dtype=float)
        for I in range(int(v_['1']),int(v_['8'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('U'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'U'+str(I))
        for I in range(int(v_['1']),int(v_['10'])+1):
            [iv,ix_,_] = jtu.s2mpj_ii('X'+str(I),ix_)
            self.xnames=jtu.arrset(self.xnames,iv,'X'+str(I))
        #%%%%%%%%%%%%%%%%%%  DATA GROUPS %%%%%%%%%%%%%%%%%%%
        self.gscale  = jnp.array([])
        self.grnames = jnp.array([])
        cnames       = jnp.array([])
        self.cnames  = jnp.array([])
        gtype        = jnp.array([])
        [ig,ig_,_] = jtu.s2mpj_ii('OBJ',ig_)
        gtype = jtu.arrset(gtype,ig,'<>')
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X1']])
        valA = jtu.append(valA,float(v_['C0']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X2']])
        valA = jtu.append(valA,float(v_['C0SQ2']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X3']])
        valA = jtu.append(valA,float(v_['C0SQ2']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X4']])
        valA = jtu.append(valA,float(v_['C0']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X5']])
        valA = jtu.append(valA,float(v_['C0']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X6']])
        valA = jtu.append(valA,float(v_['C0']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X7']])
        valA = jtu.append(valA,float(v_['C0SQ2']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X8']])
        valA = jtu.append(valA,float(v_['C0SQ2']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X9']])
        valA = jtu.append(valA,float(v_['C0']))
        irA  = jtu.append(irA,[ig])
        icA  = jtu.append(icA,[ix_['X10']])
        valA = jtu.append(valA,float(v_['C0']))
        for I in range(int(v_['1']),int(v_['8'])+1):
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
        self.gconst = jtu.arrset(self.gconst,ig_['C4'],float(v_['-P0']))
        self.gconst = jtu.arrset(self.gconst,ig_['C8'],float(v_['-P0']))
        #%%%%%%%%%%%%%%%%%%%  BOUNDS %%%%%%%%%%%%%%%%%%%%%
        self.xlower = jnp.full((self.n,1),-float('Inf'))
        self.xupper = jnp.full((self.n,1),+float('Inf'))
        for I in range(int(v_['1']),int(v_['10'])+1):
            self.xlower = jtu.np_like_set(self.xlower, ix_['X'+str(I)], 0.645)
        self.xlower = jtu.np_like_set(self.xlower, ix_['U4'], -50.8)
        self.xlower = jtu.np_like_set(self.xlower, ix_['U8'], -50.8)
        #%%%%%%%%%%%%%%%%%% START POINT %%%%%%%%%%%%%%%%%%
        self.x0 = jnp.full((self.n,1),float(0.0))
        #%%%%%%%%%%%%%%%%%%%% ELFTYPE %%%%%%%%%%%%%%%%%%%%%
        iet_  = {}
        elftv = []
        [it,iet_,_] = jtu.s2mpj_ii( 'eXU', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'U')
        [it,iet_,_] = jtu.s2mpj_ii( 'eXUPV', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'U')
        elftv = jtu.loaset(elftv,it,2,'V')
        [it,iet_,_] = jtu.s2mpj_ii( 'eXUMV', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'U')
        elftv = jtu.loaset(elftv,it,2,'V')
        [it,iet_,_] = jtu.s2mpj_ii( 'eXBIG', iet_)
        elftv = jtu.loaset(elftv,it,0,'X')
        elftv = jtu.loaset(elftv,it,1,'U')
        elftv = jtu.loaset(elftv,it,2,'V')
        elftv = jtu.loaset(elftv,it,3,'W')
        elftv = jtu.loaset(elftv,it,4,'Z')
        #%%%%%%%%%%%%%%%%%% ELEMENT USES %%%%%%%%%%%%%%%%%%
        ie_ = {}
        self.elftype = jnp.array([])
        ielftype     = jnp.array([])
        self.elvar   = []
        ename = 'EA'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eXU')
        ielftype = jtu.arrset(ielftype,ie,iet_["eXU"])
        vname = 'X1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EB'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eXUPV')
        ielftype = jtu.arrset(ielftype,ie,iet_["eXUPV"])
        vname = 'X3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EC'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eXUMV')
            ielftype = jtu.arrset(ielftype,ie,iet_['eXUMV'])
        vname = 'X6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'ED'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eXBIG')
        ielftype = jtu.arrset(ielftype,ie,iet_["eXBIG"])
        vname = 'X7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U1'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EE'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eXUMV')
            ielftype = jtu.arrset(ielftype,ie,iet_['eXUMV'])
        vname = 'X5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EF'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eXUMV')
            ielftype = jtu.arrset(ielftype,ie,iet_['eXUMV'])
        vname = 'X2'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EG'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eXU')
        ielftype = jtu.arrset(ielftype,ie,iet_["eXU"])
        vname = 'X4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EH'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        self.elftype = jtu.arrset(self.elftype,ie,'eXBIG')
        ielftype = jtu.arrset(ielftype,ie,iet_["eXBIG"])
        vname = 'X8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U4'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U5'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='W')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='Z')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EI'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eXUMV')
            ielftype = jtu.arrset(ielftype,ie,iet_['eXUMV'])
        vname = 'X9'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U3'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U7'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        ename = 'EJ'
        [ie,ie_,newelt] = jtu.s2mpj_ii(ename,ie_)
        if newelt:
            self.elftype = jtu.arrset(self.elftype,ie,'eXUMV')
            ielftype = jtu.arrset(ielftype,ie,iet_['eXUMV'])
        vname = 'X10'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='X')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U6'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='U')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        vname = 'U8'
        [iv,ix_] = jtu.s2mpj_nlx(self,vname,ix_,1,None,None,float(0.0))
        posev = jnp.where(elftv[ielftype[ie]]=='V')[0]
        self.elvar = jtu.loaset(self.elvar,ie,posev[0],iv)
        #%%%%%%%%%%%%%%%%%%% GROUP USES %%%%%%%%%%%%%%%%%%%
        self.grelt   = []
        for ig in jnp.arange(0,ngrp):
            self.grelt.append(jnp.array([]))
        self.grftype = jnp.array([])
        self.grelw   = []
        nlc         = jnp.array([])
        ig = ig_['C1']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EA'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EC'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EB'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/SQ8']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['ED'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/SQ8']))
        ig = ig_['C2']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EB'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/SQ8']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EE'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['ED'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-1/SQ8']))
        ig = ig_['C3']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EF'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/SQ8']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EH'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/SQ8']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EI'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        posel = posel+1
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EG'])
        self.grelw = jtu.loaset(self.grelw,ig,posel, 1.)
        ig = ig_['C4']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EF'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-1/SQ8']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EH'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/SQ8']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EE'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['C5']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EH'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-1/SQ8']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EC'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['C6']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EH'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-1/SQ8']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EJ'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,1.)
        ig = ig_['C7']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['ED'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['-1/SQ8']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EI'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        ig = ig_['C8']
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['ED'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(v_['1/SQ8']))
        posel = len(self.grelt[ig])
        self.grelt = jtu.loaset(self.grelt,ig,posel,ie_['EJ'])
        nlc = jnp.union1d(nlc,jnp.array([ig]))
        self.grelw = jtu.loaset(self.grelw,ig,posel,float(-1.0))
        #%%%%%%%%%%%%%%%%%% OBJECT BOUNDS %%%%%%%%%%%%%%%%%
#    Solution
# LO SOLTN               2247.1290
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
        self.pbclass   = "C-CLOR2-MY-18-8"
        self.objderlvl = 2
        self.conderlvl = [2]

# **********************
#  SET UP THE FUNCTION *
#  AND RANGE ROUTINES  *
# **********************

    #%%%%%%%%%%%%%%% NONLINEAR ELEMENTS %%%%%%%%%%%%%%%

    @staticmethod
    def eXU(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        f_   = EV_[0]*EV_[1]
        if not isinstance( f_, float ):
            f_   = f_.item();
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
    def eXUPV(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,3))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,2]), U_[0,2]+1)
        U_ = jtu.np_like_set(U_, jnp.array([1,0]), U_[1,0]+1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        f_   = IV_[1]*IV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 1, IV_[0])
            g_ = jtu.np_like_set(g_, 0, IV_[1])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eXUMV(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,3))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,2]), U_[0,2]-1)
        U_ = jtu.np_like_set(U_, jnp.array([1,0]), U_[1,0]+1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        f_   = IV_[1]*IV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 1, IV_[0])
            g_ = jtu.np_like_set(g_, 0, IV_[1])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
                H_ = U_.T.dot(H_).dot(U_)
        if nargout == 1:
            return f_
        elif nargout == 2:
            return f_,g_
        elif nargout == 3:
            return f_,g_,H_

    @staticmethod
    def eXBIG(self, nargout,*args):

        import jax.numpy as jnp
        EV_  = args[0]
        iel_ = args[1]
        U_ = jnp.zeros((2,5))
        IV_ = jnp.zeros(2)
        U_ = jtu.np_like_set(U_, jnp.array([0,1]), U_[0,1]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,2]), U_[0,2]+1)
        U_ = jtu.np_like_set(U_, jnp.array([0,3]), U_[0,3]-1)
        U_ = jtu.np_like_set(U_, jnp.array([0,4]), U_[0,4]-1)
        U_ = jtu.np_like_set(U_, jnp.array([1,0]), U_[1,0]+1)
        IV_ = jtu.np_like_set(IV_, 0, U_[0:1,:].dot(EV_))
        IV_ = jtu.np_like_set(IV_, 1, U_[1:2,:].dot(EV_))
        f_   = IV_[1]*IV_[0]
        if not isinstance( f_, float ):
            f_   = f_.item();
        if nargout>1:
            try:
                dim = len(IV_)
            except:
                dim = len(EV_)
            g_ = jnp.zeros(dim)
            g_ = jtu.np_like_set(g_, 1, IV_[0])
            g_ = jtu.np_like_set(g_, 0, IV_[1])
            g_ =  U_.T.dot(g_)
            if nargout>2:
                H_ = jnp.zeros((2,2))
                H_ = jtu.np_like_set(H_, jnp.array([1,0]), 1.0)
                H_ = jtu.np_like_set(H_, jnp.array([0,1]), H_[1,0])
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


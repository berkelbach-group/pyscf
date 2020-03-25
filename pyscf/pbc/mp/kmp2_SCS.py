#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Timothy Berkelbach <tim.berkelbach@gmail.com>
#         James McClain <jdmcclain47@gmail.com>
#


'''
kpoint-adapted and spin-adapted MP2
t2[i,j,a,b] = <ij|ab> / D_ij^ab

t2 and eris are never stored in full, only a partial
eri of size (nkpts,nocc,nocc,nvir,nvir)
'''

import time
import numpy as np
from scipy.linalg import block_diag

from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.mp import mp2
from pyscf.pbc.mp import kmp2
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LARGE_DENOM
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def kernel(mp, mo_energy, mo_coeff, css=None, cos=None,verbose=logger.NOTE, with_t2=WITH_T2):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))

    fao2mo = mp._scf.with_df.ao2mo
    kconserv = mp.khelper.kconserv
    emp2p = 0.
    emp2m = 0.
    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    if css is None: css = mp.css 
    if cos is None: cos = mp.cos 

    print(css, cos) 
    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind="split")

    if with_t2:
        t2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
    else:
        t2 = None

    for ki in range(nkpts):
      for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[ki,ka,kj]
            orbo_i = mo_coeff[ki][:,:nocc]
            orbo_j = mo_coeff[kj][:,:nocc]
            orbv_a = mo_coeff[ka][:,nocc:]
            orbv_b = mo_coeff[kb][:,nocc:]
            oovv_ij[ka] = fao2mo((orbo_i,orbv_a,orbo_j,orbv_b),
                            (mp.kpts[ki],mp.kpts[ka],mp.kpts[kj],mp.kpts[kb]),
                            compact=False).reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3) / nkpts
        for ka in range(nkpts):
            kb = kconserv[ki,ka,kj]

            # Remove zero/padded elements from denominator
            eia = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
            n0_ovp_ia = np.ix_(nonzero_opadding[ki], nonzero_vpadding[ka])
            eia[n0_ovp_ia] = (mo_e_o[ki][:,None] - mo_e_v[ka])[n0_ovp_ia]

            ejb = LARGE_DENOM * np.ones((nocc, nvir), dtype=mo_energy[0].dtype)
            n0_ovp_jb = np.ix_(nonzero_opadding[kj], nonzero_vpadding[kb])
            ejb[n0_ovp_jb] = (mo_e_o[kj][:,None] - mo_e_v[kb])[n0_ovp_jb]

            eijab = lib.direct_sum('ia,jb->ijab',eia,ejb)
            t2_ijab = np.conj(oovv_ij[ka]/eijab)
            if with_t2:
                t2[ki, kj, ka] = t2_ijab
            woovv = oovv_ij[ka]
            woovv_ex=- oovv_ij[kb].transpose(0,1,3,2)
            emp2p += np.einsum('ijab,ijab', t2_ijab, woovv).real
            emp2m += np.einsum('ijab,ijab', t2_ijab, woovv_ex).real
    
    emp2_scs=css*(emp2p+emp2m)+cos*emp2p
    emp2_scs /= nkpts

    return emp2_scs, t2


class KMP2_SCS(kmp2.KMP2):
 
       #def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, css=1./3, cos=6./5,with_t2=WITH_T2):
            #mp.KMP2.__init__(self, mf, frozen, mo_coeff, mo_occ, with_t2)
       def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, css=1./3, cos=6./5):

        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        self.css=css
        self.cos=cos
        print("css",css,"cos",cos)
##################################################
# don't modify the following attributes, they are not input options
        self.kpts = mf.kpts
        self.mo_energy = mf.mo_energy
        self.nkpts = len(self.kpts)
        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self.e_corr = None
        self.t2 = None
        self._keys = set(self.__dict__.keys())



       def kernel(self, mo_energy=None, mo_coeff=None, css=None, cos=None,with_t2=WITH_T2):
           if mo_energy is None:
               mo_energy = self.mo_energy
           if mo_coeff is None:
               mo_coeff = self.mo_coeff
           if mo_energy is None or mo_coeff is None:
               log = logger.Logger(self.stdout, self.verbose)
               log.warn('mo_coeff, mo_energy are not given.\n'
                        'You may need to call mf.kernel() to generate them.')
               raise RuntimeError
 
           mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)
 
           self.e_corr, self.t2 = \
                   kernel(self, mo_energy, mo_coeff, css, cos, verbose=self.verbose, with_t2=with_t2)
           logger.log(self, 'KMP_scs2 energy = %.15g', self.e_corr)
           return self.e_corr, self.t2

 

if __name__ == '__main__':
     from pyscf.pbc import gto, scf, mp
 
     cell = gto.Cell()
     cell.atom='''
     C 0.000000000000   0.000000000000   0.000000000000
     C 1.685068664391   1.685068664391   1.685068664391
     '''
     cell.basis = 'gth-szv'
     cell.pseudo = 'gth-pade'
     cell.a = '''
     0.000000000, 3.370137329, 3.370137329
     3.370137329, 0.000000000, 3.370137329
     3.370137329, 3.370137329, 0.000000000'''
     cell.unit = 'B'
     cell.verbose = 5
     cell.build()
 
     # Running HF and MP2 with 1x1x2 Monkhorst-Pack k-point mesh
     #kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,2]), exxdiv=None)
     kmf = scf.KRHF(cell, kpts=cell.make_kpts([1,1,3]), exxdiv=None)
     ehf = kmf.kernel()
 
     mymp = mp.KMP2(kmf)
     emp2, t2 = mymp.kernel()
     print(emp2 - -0.204721432828996)

     mymp_scs=KMP2_SCS(kmf)
     #mymp_scs.css=0
     #mymp_scs.cos=1.2
     emp2scs,t2=mymp_scs.kernel()
     #print('E(MP2scs) = %.9g' % emp2scs)

     from pyscf.pbc.tools.pbc import super_cell
     from pyscf.mp import mp2_SCS
     #scell = super_cell(cell, [1,1,2])
     scell = super_cell(cell, [1,1,3]) 
     mf = scf.RHF(scell, exxdiv=None)
     mf.kernel()
     mol_mp2scs = mp2_SCS.MP2_SCS(mf)
     emp2_mol, t2 = mol_mp2scs.kernel()
     print("KMP2 SCS:", emp2scs)	
     print("molecular MP2 SCS:", emp2_mol/3) 

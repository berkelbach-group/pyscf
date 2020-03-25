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
import h5py

from pyscf import lib
from pyscf.lib import logger, einsum
from pyscf.mp import mp2
from pyscf.pbc.mp import kmp2
from pyscf.pbc import df
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LARGE_DENOM
from pyscf import __config__

WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)


def kernel(mp, mo_energy, mo_coeff, kappa=None, verbose=logger.NOTE, with_t2=WITH_T2):
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    nkpts = mp.nkpts

    eia = np.zeros((nocc,nvir))
    eijab = np.zeros((nocc,nocc,nvir,nvir))
    eijab_reg = np.zeros((nocc,nocc,nvir,nvir))
    
    fao2mo = mp._scf.with_df.ao2mo
    kconserv = mp.khelper.kconserv
    emp2p = 0.
    emp2m = 0.
    emp2p_reg = 0.
    emp2m_reg = 0.
    oovv_ij = np.zeros((nkpts,nocc,nocc,nvir,nvir), dtype=mo_coeff[0].dtype)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    if kappa is None: kappa = mp.kappa 

    print("kappa", kappa) 
    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = kmp2.padding_k_idx(mp, kind="split")

    if with_t2:
        t2 = np.zeros((nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), dtype=complex)
    else:
        t2 = None

    do_df = mp.with_df and type(mp._scf.with_df) is df.GDF

    if do_df: Lov = _init_mp_df_eris(mp)

    for ki in range(nkpts):
      for kj in range(nkpts):
        for ka in range(nkpts):
            kb = kconserv[ki,ka,kj]
            # (ia|jb)
            if do_df:
                oovv_ij[ka] = (1./nkpts) * einsum("Lia,Ljb->iajb", Lov[ki, ka], Lov[kj, kb]).transpose(0,2,1,3)
            else:
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
            ### making tensor for calculating mp2_reg for various kappa ###
            dim_k=np.shape(kappa)[0]
            keijab=np.kron(kappa,eijab).reshape(nocc,nocc,nvir,dim_k,nvir)
            keijab1=np.zeros((dim_k,nocc,nocc,nvir,nvir))
            for i in range(dim_k):
                keijab1[i,:,:,:,:]=keijab[:,:,:,i,:]
            ## t2_ijab_reg - calculate t2 with the regulerized term ##
            #t2_ijab_reg = np.conj(oovv_ij[ka]*((1 - np.exp(kappa * eijab))**2)/eijab)
            t2_ijab_reg = np.conj(oovv_ij[ka]*((1 - np.exp(keijab1))**2)/eijab)
            if with_t2:
                t2[ki, kj, ka] = t2_ijab
            woovv = oovv_ij[ka]
            #print("woovv",np.shape(woovv))
            woovv_ex= - oovv_ij[kb].transpose(0,1,3,2)
            ## This part calculate the "direct" and "exchange" contribution separatly ###
            ## This is the direct part of the mp2 ##
            emp2p += np.einsum('ijab,ijab', t2_ijab, woovv).real
            ## This is the exchange part of the mp2 ##    
            emp2m += np.einsum('ijab,ijab', t2_ijab, woovv_ex).real
            ## This part is calculating the same terms for regulerized MP2 ##
            emp2p_reg += np.einsum('kijab,ijab', t2_ijab_reg, woovv).real
            emp2m_reg +=np.einsum('kijab,ijab', t2_ijab_reg, woovv_ex).real

    ### This part summing the terms for all the different mp2 methods###
    ### MP2 "coulomb"###
    emp2j = emp2p
    emp2j /= nkpts
    ### MP2 exchange ###
    emp2_ex = emp2m
    emp2_ex /= nkpts
    ### MP2 ###
    emp2=2*emp2p+emp2m
    emp2 /= nkpts
    ### Regulerized MP2 ###
    emp2_reg=2*emp2p_reg+emp2m_reg
    emp2_reg /= nkpts
   
    return emp2, emp2_reg, emp2j, emp2_ex, t2


def _init_mp_df_eris(mp):
    """Add 3-center electron repulsion integrals, i.e. (L|pq), in `eris`,
    where `L` denotes DF auxiliary basis functions and `p` and `q` canonical 
    crystalline orbitals. Note that `p` and `q` contain kpt indices `kp` and `kq`, 
    and the third kpt index `kL` is determined by the conservation of momentum.  
    
    Arguments:
        mp {Kmp} -- A Kmp instance
        eris {_mp_ERIS} -- A _mp_ERIS instance to which we want to add 3c ints
        
    Returns:
        _mp_ERIS -- A _mp_ERIS instance with 3c ints
    """
    from pyscf.pbc.df import df
    from pyscf.ao2mo import _ao2mo
    from pyscf.pbc.lib.kpts_helper import gamma_point

    log = logger.Logger(mp.stdout, mp.verbose)

    if mp._scf.with_df._cderi is None:
        mp._scf.with_df.build()

    cell = mp._scf.cell
    if cell.dimension == 2:
        # 2D ERIs are not positive definite. The 3-index tensors are stored in
        # two part. One corresponds to the positive part and one corresponds
        # to the negative part. The negative part is not considered in the
        # DF-driven CCSD implementation.
        raise NotImplementedError

    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    nao = cell.nao_nr()

    mo_coeff = mp._scf.mo_coeff
    kpts = mp.kpts
    nkpts = len(kpts)
    if gamma_point(kpts):
        dtype = np.double
    else:
        dtype = np.complex128
    dtype = np.result_type(dtype, *mo_coeff)
    Lov = np.empty((nkpts, nkpts), dtype=object)

    cput0 = (time.clock(), time.time())

    bra_start = 0
    bra_end = nocc
    ket_start = nmo+nocc
    ket_end = ket_start + nvir
    with h5py.File(mp._scf.with_df._cderi, 'r') as f:
        kptij_lst = f['j3c-kptij'].value 
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                kpti_kptj = np.array((kpti, kptj))
                Lpq_ao = np.asarray(df._getitem(f, 'j3c', kpti_kptj, kptij_lst))

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')
                if dtype == np.double:
                    out = _ao2mo.nr_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), aosym='s2')
                else:
                    #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    out = _ao2mo.r_e2(Lpq_ao, mo, (bra_start, bra_end, ket_start, ket_end), tao, ao_loc)
                Lov[ki, kj] = out.reshape(-1, nocc, nvir)

    log.timer_debug1("transforming DF-MP2 integrals", *cput0)

    return Lov


class KMP2_reg_sep(kmp2.KMP2):
 
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, kappa=[1.45,2]):

        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.frozen = frozen
        self.with_df = True
        self.kappa = kappa
        print("kappa", kappa)
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
        self.emp2_reg = None
        self.emp2j = None
        self.emp2_ex = None
        self._keys = set(self.__dict__.keys())



    def kernel(self, mo_energy=None, mo_coeff=None, kappa=None, with_t2=WITH_T2):
        cpu0 = (time.clock(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        if mo_energy is None:
            mo_energy = self.mo_energy
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_energy is None or mo_coeff is None:
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need to call mf.kernel() to generate them.')
            raise RuntimeError
 
        mo_coeff, mo_energy = kmp2._add_padding(self, mo_coeff, mo_energy)

        self.e_corr, self.emp2_reg, self.emp2j, self.emp2_ex, self.t2 = \
                kernel(self, mo_energy, mo_coeff,kappa,verbose=self.verbose, with_t2=with_t2)
        logger.log(self, 'KMP2 energy             = %.15g', self.e_corr)
        for i in range(np.shape(self.emp2_reg)[0]):
            logger.log(self, 'Regularized KMP2 kappa = %.3g',self.kappa[i])
            logger.log(self, 'Regularized KMP2 energy = %.15g',self.emp2_reg[i])
        logger.log(self, 'KMP2-J energy         = %.15g', self.emp2j)
        logger.log(self, 'KMP2-Ex energy         = %.15g', self.emp2_ex)

        log.timer("KMP2", *cpu0)
        return self.e_corr, self.emp2_reg, self.emp2j, self.emp2_ex, self.t2

 

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
     
     import time
     t0 = time.time()
     mymp_reg_s=KMP2_reg_sep(kmf)
     emp2, kmp2_reg, emp2j, emp2_ex, t2=mymp_reg_s.kernel()
     t1 = time.time()
     total_time= t1-t0

     #from pyscf.pbc.tools.pbc import super_cell
     #from pyscf.mp import mp2_SCS
     #scell = super_cell(cell, [1,1,2])
     #scell = super_cell(cell, [1,1,3]) 
     #mf = scf.RHF(scell, exxdiv=None)
     #mf.kernel()
     #mol_mp2scs = mp2_SCS.MP2_SCS(mf)
     #emp2_mol, t2 = mol_mp2scs.kernel()
     #print("KMP2 SCS:", emp2_scs, "kmp2:", emp2, "kmp2_sos", emp2_sos, "kmp2_reg", kmp2_reg  )	
     #print("molecular MP2 SCS:", emp2_mol/3) 

#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Sylvia Bintrim <sjb2225@columbia.edu>
#

'''
G0W0 approximation based on iterative (matrix-vector) diagonalization
'''

import time
from functools import reduce
import numpy
import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import scf
from pyscf import dft 
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
from pyscf import __config__


einsum = lib.einsum


def kernel(gw, nroots=1, koopmans=True, guess=None, eris=None, **kwargs):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(gw.stdout, gw.verbose)
    if gw.verbose >= logger.WARN:
        gw.check_sanity()
    gw.dump_flags()

    if eris is None:
        eris = gw.ao2mo()

    matvec, diag = gw.gen_matvec(eris)

    size = gw.vector_size()
    nroots = min(nroots, size)
    if guess is not None:
        user_guess = True
        for g in guess:
            assert g.size == size
    else:
        user_guess = False
        guess = gw.get_init_guess(nroots, koopmans, diag)

    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    # GHF or customized RHF/UHF may be of complex type
    real_system = (gw._scf.mo_coeff[0].dtype == np.double)

    eig = lib.davidson_nosym1
    if user_guess or koopmans:
        assert len(guess) == nroots
        def eig_close_to_init_guess(w, v, nroots, envs):
            x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
            s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
            snorm = np.einsum('pi,pi->i', s.conj(), s)
            idx = np.argsort(-snorm)[:nroots]
            return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)
        conv, es, vs = eig(matvec, guess, precond, pick=eig_close_to_init_guess,
                           tol=gw.conv_tol, max_cycle=gw.max_cycle,
                           max_space=gw.max_space, nroots=nroots, verbose=log)
    else:
        def pickeig(w, v, nroots, envs):
            real_idx = np.where(abs(w.imag) < 1e-3)[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
        conv, es, vs = eig(matvec, guess, precond, pick=pickeig,
                           tol=gw.conv_tol, max_cycle=gw.max_cycle,
                           max_space=gw.max_space, nroots=nroots, verbose=log)

    if gw.verbose >= logger.INFO:
        for n, en, vn, convn in zip(range(nroots), es, vs, conv):
            r1ip, r1ea, r2ip, r2ea = gw.vector_to_amplitudes(vn)
            if isinstance(r1ip, np.ndarray):
                qp_weight = np.linalg.norm(r1ip)**2 + np.linalg.norm(r1ea)**2
            else: # for UGW 
                r1ip = np.hstack([x.ravel() for x in r1ip])
                r1ea = np.hstack([x.ravel() for x in r1ea])
                qp_weight = np.linalg.norm(r1ip)**2 + np.linalg.norm(r1ea)**2
            logger.info(gw, 'GW root %d E = %.16g  qpwt = %.6g  conv = %s',
                        n, en, qp_weight, convn)
        log.timer('GW', *cput0)
    if nroots == 1:
        return conv[0], es[0].real, vs[0]
    else:
        return conv, es.real, vs


def vector_to_amplitudes(vector, nmo, nocc):
    nvir = nmo - nocc
    r1ip = vector[:nocc].copy()
    r1ea = vector[nocc:nmo].copy()
    r2ip = vector[nmo:nmo+nocc**2*nvir].copy().reshape(nocc,nocc,nvir)
    r2ea = vector[nmo+nocc**2*nvir:].copy().reshape(nocc,nvir,nvir)
    return r1ip, r1ea, r2ip, r2ea


def amplitudes_to_vector(r1ip, r1ea, r2ip, r2ea):
    vector = np.hstack((r1ip, r1ea, r2ip.ravel(),r2ea.ravel()))
    return vector


def matvec(gw, vector, eris=None):
    if eris is None: eris = gw.ao2mo()
    nocc = gw.nocc
    nmo = gw.nmo

    r1ip, r1ea, r2ip, r2ea = vector_to_amplitudes(vector, nmo, nocc)

    mo_e_occ = eris.mo_energy[:nocc]
    mo_e_vir = eris.mo_energy[nocc:]
    vk_occ = eris.vk[:nocc,:nocc]
    vk_vir = eris.vk[nocc:,nocc:]
    vxc_occ = eris.vxc[:nocc,:nocc]
    vxc_vir = eris.vxc[nocc:,nocc:]

    DIAG = True

    Hr1ip  = einsum('i,i->i', mo_e_occ, r1ip) 
    if DIAG:
        Hr1ip -= einsum('ii,i->i', vk_occ, r1ip) 
        Hr1ip -= einsum('ii,i->i', vxc_occ, r1ip) 
    else:
        Hr1ip -= einsum('ij,j->i', vk_occ, r1ip) 
        Hr1ip -= einsum('ij,j->i', vxc_occ, r1ip) 
    Hr1ip += einsum('klic,klc->i', eris.ooov, r2ip)
    Hr1ip += einsum('kicd,kcd->i', eris.oovv.conj(), r2ea)

    Hr1ea = einsum('a,a->a', mo_e_vir, r1ea)
    if DIAG:
        Hr1ea -= einsum('aa,a->a', vk_vir, r1ea)
        Hr1ea -= einsum('aa,a->a', vxc_vir, r1ea)
    else:
        Hr1ea -= einsum('ab,b->a', vk_vir, r1ea)
        Hr1ea -= einsum('ab,b->a', vxc_vir, r1ea)
    Hr1ea += einsum('klac,klc->a', eris.oovv, r2ip)
    Hr1ea += einsum('kacd,kcd->a', eris.ovvv.conj(), r2ea)
    
    Hr2ip = einsum('ijka,k->ija', eris.ooov.conj(), r1ip)
    Hr2ip += einsum('ijba,b->ija', eris.oovv.conj(), r1ea)
    Hr2ip += einsum('i,ija->ija', mo_e_occ, r2ip)
    Hr2ip += einsum('j,ija->ija', mo_e_occ, r2ip)
    Hr2ip -= einsum('a,ija->ija', mo_e_vir, r2ip)
    Hr2ip -= einsum('jcal,ilc->ija', eris.ovvo, r2ip)
    
    Hr2ea = einsum('icab,c->iab', eris.ovvv, r1ea)
    Hr2ea += einsum('ijab,j->iab', eris.oovv, r1ip)
    Hr2ea += einsum('a,iab->iab', mo_e_vir, r2ea)
    Hr2ea += einsum('b,iab->iab', mo_e_vir, r2ea)
    Hr2ea -= einsum('i,iab->iab', mo_e_occ, r2ea)
    Hr2ea += einsum('icak,kcb->iab', eris.ovvo, r2ea)

    vector = amplitudes_to_vector(Hr1ip, Hr1ea, Hr2ip, Hr2ea)
    return vector


class GW(lib.StreamObject):
    '''Non-relativistic generalized G0W0
    '''
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        assert(isinstance(mf, scf.ghf.GHF) or isinstance(mf, dft.gks.GKS))
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_space = getattr(__config__, 'eom_rccsd_EOM_max_space', 20)
        self.max_cycle = getattr(__config__, 'eom_rccsd_EOM_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', 1e-7)

        self.frozen = frozen
##################################################
# don't modify the following attributes, they are not input options
        self.converged = False
        self.e = None
        self.v = None
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        #log = logger.new_logger(self, verbose)
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        #logger.info('nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not None:
            logger.info('frozen orbitals %s', self.frozen)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self

    def kernel(self, nroots=1, eris=None):
        if eris is None: 
            eris = self.ao2mo(self.mo_coeff)
        self.converged, self.e, self.v = kernel(self, nroots=nroots, eris=eris) 
        return self.e, self.v

    matvec = matvec

    def gen_matvec(self, eris=None):
        if eris is None:
            eris = self.ao2mo()
        diag = self.get_diag()
        matvec = lambda xs: [self.matvec(x, eris) for x in xs]
        return matvec, diag

    def vector_to_amplitudes(self, vector, nmo=None, nocc=None):
        if nmo is None: nmo = self.nmo
        if nocc is None: nocc = self.nocc
        return vector_to_amplitudes(vector, nmo, nocc)

    def amplitudes_to_vector(self, r1ip, r1ea, r2ip, r2ea):
        return amplitudes_to_vector(r1ip, r1ea, r2ip, r2ea)

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nvir + nocc**2*nvir + nocc*nvir**2

    def get_diag(self, eris=None):
        if eris is None:
            eris = self.ao2mo()
        nocc = self.nocc
        nvir = self.nmo - nocc
        mo_e_occ = eris.mo_energy[:nocc]
        mo_e_vir = eris.mo_energy[nocc:]
        Hr1ip = mo_e_occ + np.diag(eris.vk - eris.vxc)[:nocc]
        Hr1ea = mo_e_vir + np.diag(eris.vk - eris.vxc)[nocc:]
        Hr2ip = np.zeros((nocc,nocc,nvir))
        Hr2ea = np.zeros((nocc,nvir,nvir))
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    Hr2ip[i,j,a] = mo_e_occ[i] + mo_e_occ[j] - mo_e_vir[a]
        for i in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    Hr2ea[i,a,b] = mo_e_vir[a] + mo_e_vir[b] - mo_e_occ[i]

        return self.amplitudes_to_vector(Hr1ip, Hr1ea, Hr2ip, Hr2ea)

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        nmo = self.nmo
        mem_incore = nmo**4*2 * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            raise NotImplementedError 


class GWIP(GW):

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in range(nroots):
                g = np.zeros(int(size), dtype)
                g[self.nocc-n-1] = 1.0
                guess.append(g)
        else:
            d1ip, d1ea, d2ip, d2ea = self.vector_to_amplitudes(diag)
            idx = self.amplitudes_to_vector(-d1ip, -d1ea+1e9, -d2ip, -d2ea+1e9).argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype)
                g[i] = 1.0
                guess.append(g)
        return guess


class GWEA(GW):

    def get_init_guess(self, nroots=1, koopmans=True, diag=None):
        size = self.vector_size()
        dtype = getattr(diag, 'dtype', np.double)
        nroots = min(nroots, size)
        guess = []
        if koopmans:
            for n in range(nroots):
                g = np.zeros(int(size), dtype)
                g[self.nocc+n] = 1.0
                guess.append(g)
        else:
            d1ip, d1ea, d2ip, d2ea = self.vector_to_amplitudes(diag)
            idx = self.amplitudes_to_vector(d1ip+1e9, d1ea, d2ip+1e9, d2ea).argsort()[:nroots]
            for i in idx:
                g = np.zeros(int(size), dtype)
                g[i] = 1.0
                guess.append(g)
        return guess


class _PhysicistsERIs:
    '''<pq|rs> not antisymmetrized
    
    This is gccsd _PhysicistsERIs without vvvv and without antisym'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.e_hf = None
        self.orbspin = None

        self.oooo = None
        self.ooov = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None

    def _common_init_(self, mygw, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mygw.mo_coeff
        mo_idx = mygw.get_frozen_mask()
        if getattr(mo_coeff, 'orbspin', None) is not None:
            self.orbspin = mo_coeff.orbspin[mo_idx]
            mo_coeff = lib.tag_array(mo_coeff[:,mo_idx], orbspin=self.orbspin)
        else:
            orbspin = scf.ghf.guess_orbspin(mo_coeff)
            mo_coeff = mo_coeff[:,mo_idx]
            if not np.any(orbspin == -1):
                self.orbspin = orbspin[mo_idx]
                mo_coeff = lib.tag_array(mo_coeff, orbspin=self.orbspin)
        self.mo_coeff = mo_coeff

        self.mo_energy = mygw._scf.mo_energy
        dm = mygw._scf.make_rdm1(mygw.mo_coeff, mygw.mo_occ)
        vj, vk = mygw._scf.get_jk(mygw.mol, dm) 
        vj = reduce(np.dot, (mo_coeff.conj().T, vj, mo_coeff))
        self.vk = reduce(np.dot, (mo_coeff.conj().T, vk, mo_coeff))
        vxc = mygw._scf.get_veff(mygw.mol, dm)
        self.vxc = reduce(np.dot, (mo_coeff.conj().T, vxc, mo_coeff)) - vj
        # Note: Recomputed fock matrix since SCF may not be fully converged.
        #dm = mygw._scf.make_rdm1(mygw.mo_coeff, mygw.mo_occ)
        #vhf = mygw._scf.get_veff(mygw.mol, dm)
        #fockao = mygw._scf.get_fock(vhf=vhf, dm=dm)
        #self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        #self.e_hf = mygw._scf.energy_tot(dm=dm, vhf=vhf)
        self.nocc = mygw.nocc
        self.mol = mygw.mol

        mo_e = self.mo_energy
        gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
        if gap < 1e-5:
            logger.warn(mygw, 'HOMO-LUMO gap %s too small for GCCSD', gap)
        return self


def _make_eris_incore(mygw, mo_coeff=None, ao2mofn=None):
    eris = _PhysicistsERIs()
    eris._common_init_(mygw, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape

    if callable(ao2mofn):
        eri = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        assert(eris.mo_coeff.dtype == np.double)
        mo_a = eris.mo_coeff[:nao//2]
        mo_b = eris.mo_coeff[nao//2:]
        orbspin = eris.orbspin
        if orbspin is None:
            eri  = ao2mo.kernel(mygw._scf._eri, mo_a)
            eri += ao2mo.kernel(mygw._scf._eri, mo_b)
            eri1 = ao2mo.kernel(mygw._scf._eri, (mo_a,mo_a,mo_b,mo_b))
            eri += eri1
            eri += eri1.T
        else:
            mo = mo_a + mo_b
            eri = ao2mo.kernel(mygw._scf._eri, mo)
            if eri.size == nmo**4:  # if mygw._scf._eri is a complex array
                sym_forbid = (orbspin[:,None] != orbspin).ravel()
            else:  # 4-fold symmetry
                sym_forbid = (orbspin[:,None] != orbspin)[np.tril_indices(nmo)]
            eri[sym_forbid,:] = 0
            eri[:,sym_forbid] = 0

        if eri.dtype == np.double:
            eri = ao2mo.restore(1, eri, nmo)

    eri = eri.reshape(nmo,nmo,nmo,nmo)
    eri = eri.transpose(0,2,1,3)
    #eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)

    eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
    eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvo = eri[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
    return eris


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 0
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '321g'
    mol.build()

    mf = scf.GKS(mol)
    mf.xc = 'hf'
    mf.kernel()

    gw = GWIP(mf)
    eip,v = gw.kernel(nroots=6)

    gw = GWEA(mf)
    eea,v = gw.kernel(nroots=6)

    print("EIP =", eip)
    print("EEA =", eea)

    print(eip[-1] - -0.40986000004366957)
    print(eip[-3] - -0.48524655758347784)
    print(eip[-5] - -0.6699279585527715)
    print(eea[0] - 0.2544676075076162)
    print(eea[2] - 0.34901293602250677)
    print(eea[4] - 1.1427379448996156)

    mf.xc = 'lda'
    mf.kernel()

    gw = GWIP(mf)
    eip,v = gw.kernel(nroots=2)

    gw = GWEA(mf)
    eea,v = gw.kernel(nroots=2)

    print("EIP =", eip)
    print("EEA =", eea)

    print(eip[-1] - -0.34807392)
    print(eea[0] - 0.24821382)

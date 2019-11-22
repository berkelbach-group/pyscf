#!/usr/bin/env python
import numpy
from numpy import array
from pyscf import gto as mol_gto
from pyscf import scf as mol_scf
from pyscf.mp import mp2 as mol_mp2
from pyscf.cc import rccsd as mol_rccsd
from pyscf.cc import eom_rccsd as mol_eom_rccsd
from pyscf.pbc import gto, scf, cc#, ci
from pyscf.pbc.scf import addons
# from pyscf.pbc.cc import eom_kccsd_rhf as eom_krccsd
# from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
from pyscf.pbc.tools.pbc import super_cell
from pyscf.cc.rccsd import RCCSD
from pyscf.pbc.cc.kccsd import KGCCSD
from pyscf.pbc.cc.kccsd_rhf import KRCCSD
from pyscf.pbc.cc.kccsd_uhf import KUCCSD
from pyscf.pbc.tools.pbc import super_cell

cell = gto.Cell()
cell.verbose = 0
cell.unit = 'B'
cell.max_memory = 8000  # MB

#
# SiC
#
cell.atom = [['Si', array([0., 0., 0.])], ['C', array([2.05507701, 2.05507701, 2.05507701])]]
cell.a = array([[0.        , 4.11015403, 4.11015403],
    [4.11015403, 0.        , 4.11015403],
    [4.11015403, 4.11015403, 0.        ]])


cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.mesh = [15]*3

cell.build()

nmp = [2,1,1]

# treating supercell at gamma point
supcell = super_cell(cell,nmp)

mf = scf.RHF(supcell, exxdiv=None)
mf.conv_tol = 1e-12
mf.max_cycle = 100
mf.verbose=0
erhf = mf.kernel()
print("erhf = {}".format(erhf/ numpy.prod(nmp)))

cc2 = True

mycc = mol_rccsd.RCCSD(mf)
mycc.conv_tol = 1e-12
mycc.max_cycle = 100
mycc.verbose=0
mycc.cc2 = cc2
# mycc.cc2 = False
ercc, t1, t2 = mycc.kernel()
print("ercc = {}".format(ercc/numpy.prod(nmp)))

# mf = scf.RHF(cell, exxdiv=None)
# mf.conv_tol = 1e-12
# mf.max_cycle = 100
# mf.verbose=0
# erhf = mf.kernel()
# print("erhf = {}".format(erhf))

# cc2 = True

# mycc = mol_rccsd.RCCSD(mf)
# mycc.conv_tol = 1e-12
# mycc.max_cycle = 100
# mycc.verbose=0
# mycc.cc2 = cc2
# # mycc.cc2 = False
# ercc, t1, t2 = mycc.kernel()
# print("ercc = {}".format(ercc))

# gmf  = scf.UHF(supcell,exxdiv=None)
# ehf  = gmf.kernel()
# gcc  = cc.UCCSD(gmf)
# ecc, t1, t2 = gcc.kernel()
# print('UHF energy (supercell) %f \n' % (float(ehf)/numpy.prod(nmp)+4.343308413289))
# print('UCCSD correlation energy (supercell) %f \n' % (float(ecc)/numpy.prod(nmp)+0.009470753047083676))

# treating mesh of k points

kpts  = cell.make_kpts(nmp)
kpts -= kpts[0]

kmf0   = scf.KRHF(cell,kpts,exxdiv=None)
kmf0.conv_tol = 1e-12
kmf0.max_cycle = 100
kmf0.verbose=0
ehf   = kmf0.kernel()
print("ekmf = {}".format(ehf))

# kcc   = KRCCSD(kmf,cc2=cc2)
# kcc.max_cycle=100
# kcc.conv_tol=1e-12
# ecc, t1, t2 = kcc.kernel()
# print("ekcc = {}".format(ecc))

kmf = scf.addons.convert_to_ghf(kmf0)
mycc = KGCCSD(kmf, cc2=cc2)

mycc.max_cycle=100
mycc.conv_tol=1e-12
ecc, t1, t2 = mycc.kernel()
print("E(KGCCSD) = {}".format(ecc))

kmf = scf.addons.convert_to_uhf(kmf0)
mycc = KUCCSD(kmf, cc2=cc2)
mycc.max_cycle=100
mycc.conv_tol=1e-12
ecc, t1, t2 = mycc.kernel()
print("E(KUCCSD) = {}".format(ecc))

mycc = KRCCSD(kmf0, cc2=cc2)
mycc.max_cycle=100
mycc.conv_tol=1e-12
ecc, t1, t2 = mycc.kernel()
print("E(KRCCSD) = {}".format(ecc))
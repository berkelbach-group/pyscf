#!/usr/bin/env python

def main():
    import numpy as np
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
    
    cell = gto.Cell()
    cell.verbose = 0
    cell.unit = 'B'
    cell.max_memory = 8000  # MB

    #
    # Diamond
    #
    # cell.atom = '''
    # C 0.000000000000   0.000000000000   0.000000000000
    # C 1.685068664391   1.685068664391   1.685068664391
    # '''
    # cell.a = '''
    # 0.000000000, 3.370137329, 3.370137329
    # 3.370137329, 0.000000000, 3.370137329
    # 3.370137329, 3.370137329, 0.000000000'''

    #
    # Silicon
    #
    # cell.atom = [['Si', array([0., 0., 0.])], ['Si', array([2.56577546, 2.56577546, 2.56577546])]]
    # cell.a = array([[0.        , 5.13155092, 5.13155092],
    #     [5.13155092, 0.        , 5.13155092],
    #     [5.13155092, 5.13155092, 0.        ]])

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

    # Gamma-point RHF
    mf = scf.RHF(cell, exxdiv=None)
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.verbose=5
    erhf = mf.kernel()
    
    mymp = mol_mp2.RMP2(mf)
    emp2 = mymp.kernel()

    cc2 = True

    mycc = mol_rccsd.RCCSD(mf)
    mycc.conv_tol = 1e-12
    mycc.max_cycle = 100
    mycc.verbose=5
    mycc.cc2 = cc2
    ercc, t1, t2 = mycc.kernel()

    kmf0 = scf.KRHF(cell, kpts=cell.make_kpts([1,1,1]), exxdiv=None)
    kmf0.conv_tol=1e-12
    kmf0.max_cycle=100
    kmf0.kernel()

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
# E(RCCSD) = -6.588789421647054  E_corr = -0.2605418857453103
# E(KGCCSD) = -0.26054185588608375
# E(KUCCSD) = -0.2605418559205005
# E(KRCCSD) = -0.2605418559232318

# E(RCCSD) = -6.586743471901487  E_corr = -0.2609293950667302 Reference number
# E(GCCSD) = -6.586743420843472  E_corr = -0.2609293440087155


if __name__ == "__main__":
    main()
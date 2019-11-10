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
    # from pyscf.pbc.cc import eom_kccsd_rhf as eom_krccsd
    # from pyscf.pbc.cc import eom_kccsd_ghf as eom_kgccsd
    from pyscf.pbc.tools.pbc import super_cell
    from pyscf.cc.rccsd import RCCSD
    
    cell = gto.Cell()
    cell.verbose = 7
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

    cell.build()

    # Gamma-point RHF
    mf = scf.RHF(cell, exxdiv='ewald')
    erhf = mf.kernel()
    
    # Molecular RHF
    # mf = mol_scf.RHF(cell)
    # erhf = mf.kernel()

    mymp = mol_mp2.RMP2(mf)
    emp2 = mymp.kernel()

    mycc = mol_rccsd.RCCSD(mf)
    mycc.cc2 = True
    # E(RCCSD) = -8.618718453661803  E_corr = -0.06191155910005863

    ercc, t1, t2 = mycc.kernel()

if __name__ == "__main__":

    main()
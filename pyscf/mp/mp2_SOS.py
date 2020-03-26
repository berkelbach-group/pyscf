import time
from functools import reduce
import copy
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import mp
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf import __config__
from pyscf.mp import mp2
from pyscf.mp import mp2_SCS

###SOS is writen as a sunclass pf mp2-scs, css=0, cos=1.2 taking only the opposite spin component
class MP2_SOS(mp2_SCS.MP2_SCS):
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, css=0, cos=1.2):
          mp2.MP2.__init__(self,mf, frozen, mo_coeff, mo_occ)
          self.css = css
          self.cos = cos



if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    mol = gto.Mole()
    mol.verbose = 5
    #mol.output = 'H2_mp2.log'
    #mol.output = 'H2O_mp2_scs.log'
    mol.atom= 'H 0 0 -0.6969; H 0 0 0.6969'
    #mol.atom= 'O 0 0 0.2253; H 0 1.4424 -0.9012; H 0 -1.4424 -0.9012'
    
    mol.basis = 'ccpvtz'
    mol.build()
    m = scf.RHF(mol)
    print('E(HF) = %g' % m.kernel())
    #mp2n=mp.MP2(m, frozen=[0])
    mp2n=mp.MP2(m)
    print('E(MP2) = %.9g' % mp2n.kernel()[0])
    mp2_SOS = MP2_SOS(m)
    #mp2_SOS.css = 10.0
    #mp2_SOS.cos = 2
    #print(isinstance(mp2_SOS,MP2_SCS))
    emp2scs, t2 = mp2_SOS.kernel()
    print('E(MP2p) = %.9g' % emp2scs)



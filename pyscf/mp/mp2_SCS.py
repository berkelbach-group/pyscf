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


WITH_T2 = getattr(__config__, 'mp_mp2_with_t2', True)

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, css=None, cos=None,
           verbose=logger.NOTE):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert(mp.frozen == 0 or mp.frozen is None)

    if eris is None: 
        eris = mp.ao2mo(mo_coeff)
    
    if mo_energy is None:
        mo_energy = eris.mo_energy

    if css is None: css = mp.css 
    if cos is None: cos = mp.cos 

    print(css, cos)
    
    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    print("mo_energy",mo_energy,numpy.size(mo_energy))
    print("eia",eia,numpy.shape(eia))
    if with_t2:
        t2 = numpy.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    emp2p = 0
    emp2m = 0
    for i in range(nocc):
        gi = numpy.asarray(eris.ovov[i*nvir:(i+1)*nvir])   #Importing the 4iindex integral ovov
        gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)   # Transforming it to a oovv shape
        t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])  # Summing over the orbitals energies deviding the <ij||ab>
        emp2p += numpy.einsum('jab,jab', t2i, gi)         #summing over the coulomb like term 
        emp2m -= numpy.einsum('jab,jba', t2i, gi)         #summing over the exchange like term
        if with_t2:
            t2[i] = t2i
    #print("nocc",nocc,"nvir",nvir)
    #print("t2",numpy.shape(t2))
    emp2_scs=css*(emp2p+emp2m)+cos*emp2p
    return emp2_scs, t2

class MP2_SCS(mp2.MP2):

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, css=1./3, cos=6./5):
        mp2.MP2.__init__(self,mf, frozen, mo_coeff, mo_occ)
        self.css = css
        self.cos = cos 
    
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, css=None, cos=None):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2, css, cos)



if __name__ == '__main__':
     from pyscf import scf
     from pyscf import gto
     mol = gto.Mole()
     mol.verbose = 5
     #mol.output = 'H2_mp2.log'
     mol.output = 'H2O_mp2_scs.log'
     #mol.atom= 'H 0 0 -0.6969; H 0 0 0.6969' 
     mol.atom= 'O 0 0 0.2253; H 0 1.4424 -0.9012; H 0 -1.4424 -0.9012'
     mol.basis = 'ccpvtz'
     mol.build()
     m = scf.RHF(mol)
     print('E(HF) = %g' % m.kernel())
     mp2n=mp.MP2(m, frozen=[0])
     print('E(MP2) = %.9g' % mp2n.kernel()[0])
     #css=2
     #cos=1
     mp2_SCS = MP2_SCS(m, frozen=[0])
     #mp2_SCS.css = 10.0        #The user can change the css, cos
     #mp2_SCS.cos = 2          
     print(isinstance(mp2_SCS,MP2_SCS))
     emp2scs, t2 = mp2_SCS.kernel()
     print('E(MP2p) = %.9g' % mp2_SCS.kernel()[0])

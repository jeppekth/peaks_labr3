import numpy as np
import matplotlib.pyplot as plt

from NIST_isotope_mass import NIST_isotope_mass

from numpy.typing import NDArray

def lab_energy_to_cm(lab_energy: NDArray[np.float64], proj_mass: float, target_mass: float) -> NDArray[np.float64]:
    factor: float = target_mass / (proj_mass + target_mass)
    return lab_energy * factor

def cm_energy_to_lab(cm_energy: NDArray[np.float64], proj_mass: float, target_mass: float) -> NDArray[np.float64]:
    factor: float = (proj_mass + target_mass) / target_mass
    return cm_energy * factor

def eject_lab_energy(lab_energy: NDArray[np.float64], lab_angle: float, proj_mass: float, target_mass: float, eject_mass: float, recoil_mass: float, recoil_ex: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    q_value: float = proj_mass + target_mass - eject_mass - recoil_mass - recoil_ex

    enum1: NDArray[np.float64] = np.sqrt( (proj_mass * eject_mass) * lab_energy) * np.cos(lab_angle)
    enum2: NDArray[np.float64] = np.sqrt( (proj_mass *eject_mass * np.power(np.cos(lab_angle), 2)) * lab_energy \
                                         + (recoil_mass + eject_mass) * (recoil_mass * q_value + (recoil_mass - proj_mass) * lab_energy) )
    
    result_plus: NDArray[np.float64] = np.power((enum1 + enum2) / (recoil_mass + eject_mass), 2)
    result_minus: NDArray[np.float64] = np.power((enum1 - enum2) / (recoil_mass + eject_mass), 2)
    
    return (result_plus, result_minus)

def main():
    nist_isotope_mass: NIST_isotope_mass = NIST_isotope_mass("data/nist_isotope_mass.txt")
    
    neutron_mass: float = 939.5654133e3 # keV
    H1_mass: float = nist_isotope_mass.get_isotope_mass("H", 1)
    He4_mass: float = nist_isotope_mass.get_isotope_mass("He", 4)
    C12_mass: float = nist_isotope_mass.get_isotope_mass("C", 12)
    C13_mass: float = nist_isotope_mass.get_isotope_mass("C", 13)
    O16_mass: float = nist_isotope_mass.get_isotope_mass("O", 16)

    C12_a_g_O16_q_value: float = C12_mass + He4_mass - O16_mass
    C13_a_n_O16_q_value: float = C13_mass + He4_mass - neutron_mass - O16_mass

    print(f"12C(a,g)16O Q-value: {C12_a_g_O16_q_value : .3f}")
    print(f"13C(a,n)16O Q-value: {C13_a_n_O16_q_value : .3f}")
    

    return

if __name__ == "__main__":
    main()
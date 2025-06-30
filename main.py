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
    As76_mass: float = nist_isotope_mass.get_isotope_mass("As", 76)
    As78_mass: float = nist_isotope_mass.get_isotope_mass("As", 78)
    Se79_mass: float = nist_isotope_mass.get_isotope_mass("Se", 79)
    Se81_mass: float = nist_isotope_mass.get_isotope_mass("Se", 81)
    Br79_mass: float = nist_isotope_mass.get_isotope_mass("Br", 79)
    Br81_mass: float = nist_isotope_mass.get_isotope_mass("Br", 81)

    C12_a_g_O16_q_value: float = C12_mass + He4_mass - O16_mass
    C13_a_n_O16_q_value: float = C13_mass + He4_mass - neutron_mass - O16_mass
    Br79_n_p_Se79_q_value: float = Br79_mass + neutron_mass - H1_mass - Se79_mass
    Br81_n_p_Se81_q_value: float = Br81_mass + neutron_mass - H1_mass - Se81_mass
    Br79_n_a_As76_q_value: float = Br79_mass + neutron_mass - He4_mass - As76_mass
    Br81_n_a_As78_q_value: float = Br81_mass + neutron_mass - He4_mass - As78_mass


    print(f"12C(a,g)16O Q-value: {C12_a_g_O16_q_value : .3f} keV")
    print(f"13C(a,n)16O Q-value: {C13_a_n_O16_q_value : .3f} keV")
    print(f"79Br(n,p)79Se Q-value: {Br79_n_p_Se79_q_value : .3f} keV")
    print(f"81Br(n,p)81Se Q-value {Br81_n_p_Se81_q_value : .3f} keV")
    print(f"79Br(n,a)76As Q-value: {Br79_n_a_As76_q_value : .3f} keV")
    print(f"81Br(n,a)78As Q-value: {Br81_n_a_As78_q_value : .3f} keV")


    return

if __name__ == "__main__":
    main()
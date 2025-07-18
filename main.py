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
    lab_angle = np.deg2rad(lab_angle)
    q_value: float = proj_mass + target_mass - eject_mass - recoil_mass - recoil_ex

    enum1: NDArray[np.float64] = np.sqrt( (proj_mass * eject_mass) * lab_energy) * np.cos(lab_angle)
    enum2: NDArray[np.float64] = np.sqrt( (proj_mass * eject_mass * np.power(np.cos(lab_angle), 2)) * lab_energy \
                                         + (recoil_mass + eject_mass) * (recoil_mass * q_value + (recoil_mass - proj_mass) * lab_energy) )
    
    result_plus: NDArray[np.float64] = np.power((enum1 + enum2) / (recoil_mass + eject_mass), 2)
    result_minus: NDArray[np.float64] = np.power((enum1 - enum2) / (recoil_mass + eject_mass), 2)
    
    return (result_plus, result_minus)

def gamma_energy(lab_energy: NDArray[np.float64], proj_mass: float, target_mass: float, recoil_mass: float, recoil_ex: float) -> NDArray[np.float64]:
    q_value: float = proj_mass + target_mass - recoil_mass - recoil_ex

    momentum2: NDArray[np.float64] = (2 * proj_mass ) * lab_energy

    recoil_energy: NDArray[np.float64] = momentum2 / (2 * (recoil_mass + recoil_ex))
    
    return lab_energy + q_value - recoil_energy

def has_enough_energy(lab_energy: NDArray[np.float64], lab_angle: float, proj_mass: float, target_mass: float, eject_mass: float, recoil_mass: float, recoil_ex: float) -> list[bool]:
    lab_angle = np.deg2rad(lab_angle)
    q_value: float = proj_mass + target_mass - eject_mass - recoil_mass - recoil_ex

    result: list[bool] = []
    
    for i in range(len(lab_energy)):
        if q_value + lab_energy[i] <= 0:
            result.append(False)
            continue

        enum2: float = (proj_mass * eject_mass * np.power(np.cos(lab_angle), 2)) * lab_energy[i] \
                                         + (recoil_mass + eject_mass) * (recoil_mass * q_value + (recoil_mass - proj_mass) * lab_energy[i])
        
        if enum2 < 0:
            result.append(False)
            continue

        result.append(True)

    return  result

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

    lab_energy: NDArray[np.float64] = np.linspace(4e3, 9e3, 1000, dtype=np.float64)

    C12_a_g_O16_gs: NDArray[np.float64] = gamma_energy(lab_energy, He4_mass, C12_mass, O16_mass, 0)
    C12_a_g_O16_6130: NDArray[np.float64] = gamma_energy(lab_energy, He4_mass, C12_mass, O16_mass, 6130)
    C12_a_g_O16_6917: NDArray[np.float64] = gamma_energy(lab_energy, He4_mass, C12_mass, O16_mass, 6917)
    C12_a_g_O16_7117: NDArray[np.float64] = gamma_energy(lab_energy, He4_mass, C12_mass, O16_mass, 7117)

    plt.rc("font", family=["Helvetica", "Arial"])
    plt.rc("text", usetex=True)
    plt.rc("axes", labelsize=14, titlesize=14)
    plt.rc("xtick", labelsize=14, top=True, direction="in")
    plt.rc("ytick", labelsize=14, right=True, direction="in")
    plt.rc("legend", fontsize=14)

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{12}$C$(\\alpha,\\gamma)^{16}$O (labels for final state)")
    ax.set_ylabel("$E_{\\gamma}$ [keV]")
    ax.set_xlabel("$E_{\\alpha}$ [keV]")
    ax.plot(lab_energy, C12_a_g_O16_gs, linestyle="-", color="black", label="$^{16}$O$(\\mathrm{g.s.})$")
    ax.plot(lab_energy, C12_a_g_O16_6130, linestyle="--", color="red", label="$^{16}$O$(6130)$")
    ax.plot(lab_energy, C12_a_g_O16_6917, linestyle="-.", color="blue", label="$^{16}$O$(6917)$")
    ax.plot(lab_energy, C12_a_g_O16_7117, linestyle=":", color="green", label="$^{16}$O$(7117)$")
    ax.legend()

    fig.savefig("figs/C12_a_g_O16.png", bbox_inches="tight", dpi=400)

    fig_45deg, ax_45deg = plt.subplots(1, 1, figsize=(9,7))
    ax_45deg.set_title("$\\gamma$ energy and neutron induced ejectile energy from ${13}$C$(\\alpha,n)^{16}$O at 45 degree detector angle")
    ax_45deg.set_ylabel("$\\gamma$ or ejectile energy [keV]")
    ax_45deg.set_xlabel("Incoming $E_\\alpha$ [keV]")
    ax_45deg.plot(lab_energy, C12_a_g_O16_gs, linestyle="-", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(\\mathrm{g.s.})$")
    ax_45deg.plot(lab_energy, C12_a_g_O16_6130, linestyle="--", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(6130)$")
    ax_45deg.plot(lab_energy, C12_a_g_O16_6917, linestyle="-.", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(6917)$")
    ax_45deg.plot(lab_energy, C12_a_g_O16_7117, linestyle=":", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(7117)$")

    fig_90deg, ax_90deg = plt.subplots(1, 1, figsize=(9,7))
    ax_90deg.set_title("$\\gamma$ energy and neutron induced ejectile energy from ${13}$C$(\\alpha,n)^{16}$O at 90 degree detector angle")
    ax_90deg.set_ylabel("$\\gamma$ or ejectile energy [keV]")
    ax_90deg.set_xlabel("Incoming $E_\\alpha$ [keV]")
    ax_90deg.plot(lab_energy, C12_a_g_O16_gs, linestyle="-", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(\\mathrm{g.s.})$")
    ax_90deg.plot(lab_energy, C12_a_g_O16_6130, linestyle="--", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(6130)$")
    ax_90deg.plot(lab_energy, C12_a_g_O16_6917, linestyle="-.", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(6917)$")
    ax_90deg.plot(lab_energy, C12_a_g_O16_7117, linestyle=":", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(7117)$")

    fig_135deg, ax_135deg = plt.subplots(1, 1, figsize=(9,7))
    ax_135deg.set_title("$\\gamma$ energy and neutron induced ejectile energy from ${13}$C$(\\alpha,n)^{16}$O at 135 degree detector angle")
    ax_135deg.set_ylabel("$\\gamma$ or ejectile energy [keV]")
    ax_135deg.set_xlabel("Incoming $E_\\alpha$ [keV]")
    ax_135deg.plot(lab_energy, C12_a_g_O16_gs, linestyle="-", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(\\mathrm{g.s.})$")
    ax_135deg.plot(lab_energy, C12_a_g_O16_6130, linestyle="--", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(6130)$")
    ax_135deg.plot(lab_energy, C12_a_g_O16_6917, linestyle="-.", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(6917)$")
    ax_135deg.plot(lab_energy, C12_a_g_O16_7117, linestyle=":", color="red", label="$^{12}$C$(\\alpha,\\gamma)^{16}$O$(7117)$")

    ################
    ################
    # C13_a_n_O16_gs

    C13_a_n_O16_gs_45deg: NDArray[np.float64] = eject_lab_energy(lab_energy, 45, He4_mass, C13_mass, neutron_mass, O16_mass, 0)[0]
    C13_a_n_O16_gs_90deg: NDArray[np.float64] = eject_lab_energy(lab_energy, 90, He4_mass, C13_mass, neutron_mass, O16_mass, 0)[0]
    C13_a_n_O16_gs_135deg: NDArray[np.float64] = eject_lab_energy(lab_energy, 135, He4_mass, C13_mass, neutron_mass, O16_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(\\mathrm{g.s.})$")
    ax.set_ylabel("$E_{n}$ [keV]")
    ax.set_xlabel("$E_{\\alpha}$ [keV]")
    ax.plot(lab_energy, C13_a_n_O16_gs_45deg, linestyle="-", color="black", label="$\\theta_\\mathrm{LAB}=45\\,$deg")
    ax.plot(lab_energy, C13_a_n_O16_gs_90deg, linestyle="--", color="red", label="$\\theta_\\mathrm{LAB}=90\\,$deg")
    ax.plot(lab_energy, C13_a_n_O16_gs_135deg, linestyle="-.", color="blue", label="$\\theta_\\mathrm{LAB}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_g_O16_gs.png", bbox_inches="tight", dpi=400)

    Br79_n_p_Se79_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_45deg, 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)[0]
    Br79_n_p_Se79_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_90deg, 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)[0]
    Br79_n_p_Se79_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_135deg, 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(\\mathrm{g.s.})$ to $^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $\\theta_p=0\\,$deg")
    ax.set_ylabel("$E_{p}$ [keV]")
    ax.set_xlabel("$E_{\\alpha}$ [keV]")
    ax.plot(lab_energy, Br79_n_p_Se79_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy, Br79_n_p_Se79_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy, Br79_n_p_Se79_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_n_O16_gs_Br79_n_p_Se79.png", bbox_inches="tight", dpi=400)

    Br81_n_p_Se81_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_45deg, 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)[0]
    Br81_n_p_Se81_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_90deg, 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)[0]
    Br81_n_p_Se81_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_135deg, 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(\\mathrm{g.s.})$ to $^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $\\theta_p=0\\,$deg")
    ax.set_ylabel("$E_{p}$ [keV]")
    ax.set_xlabel("$E_{\\alpha}$ [keV]")
    ax.plot(lab_energy, Br81_n_p_Se81_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy, Br81_n_p_Se81_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy, Br81_n_p_Se81_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_n_O16_gs_Br81_n_p_Se81.png", bbox_inches="tight", dpi=400)
    

    Br79_n_a_As76_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_45deg, 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)[0]
    Br79_n_a_As76_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_90deg, 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)[0]
    Br79_n_a_As76_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_135deg, 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(\\mathrm{g.s.})$ to $^{79}$Br$(n,a)^{76}$As$(\\mathrm{g.s.})$ $\\theta_\\alpha=0\\,$deg")
    ax.set_ylabel("$E_{\\alpha}$ [keV]")
    ax.set_xlabel("Incoming $E_{\\alpha}$ [keV]")
    ax.plot(lab_energy, Br79_n_a_As76_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy, Br79_n_a_As76_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy, Br79_n_a_As76_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_n_O16_gs_Br79_n_a_As76.png", bbox_inches="tight", dpi=400)

    Br81_n_a_As78_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_45deg, 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)[0]
    Br81_n_a_As78_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_90deg, 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)[0]
    Br81_n_a_As78_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_gs_135deg, 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(\\mathrm{g.s.})$ to $^{81}$Br$(n,a)^{78}$As$(\\mathrm{g.s.})$ $\\theta_\\alpha=0\\,$deg")
    ax.set_ylabel("$E_{\\alpha}$ [keV]")
    ax.set_xlabel("Incoming $E_{\\alpha}$ [keV]")
    ax.plot(lab_energy, Br81_n_a_As78_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy, Br81_n_a_As78_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy, Br81_n_a_As78_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_n_O16_gs_Br81_n_a_As78.png", bbox_inches="tight", dpi=400)

    ax_45deg.plot(lab_energy, Br79_n_p_Se79_gs_45deg, linestyle="-", color="black", label="$^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")
    ax_45deg.plot(lab_energy, Br81_n_p_Se81_gs_45deg, linestyle="-", color="blue", label="$^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")
    ax_45deg.plot(lab_energy, Br79_n_a_As76_gs_45deg, linestyle="-", color="green", label="$^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")
    ax_45deg.plot(lab_energy, Br81_n_a_As78_gs_45deg, linestyle="-", color="cyan", label="$^{81}$Br$(n,\\alpha)^{78}$As$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")

    ax_90deg.plot(lab_energy, Br79_n_p_Se79_gs_90deg, linestyle="-", color="black", label="$^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")
    ax_90deg.plot(lab_energy, Br81_n_p_Se81_gs_90deg, linestyle="-", color="blue", label="$^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")
    ax_90deg.plot(lab_energy, Br79_n_a_As76_gs_90deg, linestyle="-", color="green", label="$^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")
    ax_90deg.plot(lab_energy, Br81_n_a_As78_gs_90deg, linestyle="-", color="cyan", label="$^{81}$Br$(n,\\alpha)^{78}$As$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")

    ax_135deg.plot(lab_energy, Br79_n_p_Se79_gs_135deg, linestyle="-", color="black", label="$^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")
    ax_135deg.plot(lab_energy, Br81_n_p_Se81_gs_135deg, linestyle="-", color="blue", label="$^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")
    ax_135deg.plot(lab_energy, Br79_n_a_As76_gs_135deg, linestyle="-", color="green", label="$^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")
    ax_135deg.plot(lab_energy, Br81_n_a_As78_gs_135deg, linestyle="-", color="cyan", label="$^{81}$Br$(n,\\alpha)^{78}$As$(\\mathrm{g.s.})$ $^{16}$O$(\\mathrm{g.s.})$")

    ################
    ################
    # C13_a_n_O16_6130

    C13_a_n_O16_6130_45deg_valid: list[bool] = has_enough_energy(lab_energy, 45, He4_mass, C13_mass, neutron_mass, O16_mass, 6130)
    C13_a_n_O16_6130_90deg_valid: list[bool] = has_enough_energy(lab_energy, 90, He4_mass, C13_mass, neutron_mass, O16_mass, 6130)
    C13_a_n_O16_6130_135deg_valid: list[bool] = has_enough_energy(lab_energy, 135, He4_mass, C13_mass, neutron_mass, O16_mass, 6130)

    C13_a_n_O16_6130_45deg: NDArray[np.float64] = eject_lab_energy(lab_energy[C13_a_n_O16_6130_45deg_valid], 45, He4_mass, C13_mass, neutron_mass, O16_mass, 6130)[0]
    C13_a_n_O16_6130_90deg: NDArray[np.float64] = eject_lab_energy(lab_energy[C13_a_n_O16_6130_90deg_valid], 90, He4_mass, C13_mass, neutron_mass, O16_mass, 6130)[0]
    C13_a_n_O16_6130_135deg: NDArray[np.float64] = eject_lab_energy(lab_energy[C13_a_n_O16_6130_135deg_valid], 135, He4_mass, C13_mass, neutron_mass, O16_mass, 6130)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(6130)$")
    ax.set_ylabel("$E_{n}$ [keV]")
    ax.set_xlabel("$E_{\\alpha}$ [keV]")
    ax.plot(lab_energy[C13_a_n_O16_6130_45deg_valid], C13_a_n_O16_6130_45deg, linestyle="-", color="black", label="$\\theta_\\mathrm{LAB}=45\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6130_90deg_valid], C13_a_n_O16_6130_90deg, linestyle="--", color="red", label="$\\theta_\\mathrm{LAB}=90\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6130_135deg_valid], C13_a_n_O16_6130_135deg, linestyle="-.", color="blue", label="$\\theta_\\mathrm{LAB}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_g_O16_6130.png", bbox_inches="tight", dpi=400)

    Br79_n_p_Se79_gs_45deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_45deg, 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)
    Br79_n_p_Se79_gs_90deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_90deg, 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)
    Br79_n_p_Se79_gs_135deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_135deg, 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)

    Br79_n_p_Se79_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_45deg[Br79_n_p_Se79_gs_45deg_valid], 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)[0]
    Br79_n_p_Se79_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_90deg[Br79_n_p_Se79_gs_90deg_valid], 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)[0]
    Br79_n_p_Se79_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_135deg[Br79_n_p_Se79_gs_135deg_valid], 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(6130)$ to $^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $\\theta_p=0\\,$deg")
    ax.set_ylabel("$E_{p}$ [keV]")
    ax.set_xlabel("$E_{\\alpha}$ [keV]")
    ax.plot(lab_energy[C13_a_n_O16_6130_45deg_valid][Br79_n_p_Se79_gs_45deg_valid], Br79_n_p_Se79_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6130_90deg_valid][Br79_n_p_Se79_gs_90deg_valid], Br79_n_p_Se79_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6130_135deg_valid][Br79_n_p_Se79_gs_135deg_valid], Br79_n_p_Se79_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_n_O16_6130_Br79_n_p_Se79.png", bbox_inches="tight", dpi=400)

    Br81_n_p_Se81_gs_45deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_45deg, 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)
    Br81_n_p_Se81_gs_90deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_90deg, 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)
    Br81_n_p_Se81_gs_135deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_135deg, 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)

    Br81_n_p_Se81_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_45deg[Br81_n_p_Se81_gs_45deg_valid], 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)[0]
    Br81_n_p_Se81_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_90deg[Br81_n_p_Se81_gs_90deg_valid], 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)[0]
    Br81_n_p_Se81_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_135deg[Br81_n_p_Se81_gs_135deg_valid], 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(6130)$ to $^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $\\theta_p=0\\,$deg")
    ax.set_ylabel("$E_{p}$ [keV]")
    ax.set_xlabel("$E_{\\alpha}$ [keV]")
    ax.plot(lab_energy[C13_a_n_O16_6130_45deg_valid][Br81_n_p_Se81_gs_45deg_valid], Br81_n_p_Se81_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6130_90deg_valid][Br81_n_p_Se81_gs_90deg_valid], Br81_n_p_Se81_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6130_135deg_valid][Br81_n_p_Se81_gs_135deg_valid], Br81_n_p_Se81_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_n_O16_6130_Br81_n_p_Se81.png", bbox_inches="tight", dpi=400)

    Br79_n_a_As76_gs_45deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_45deg, 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)
    Br79_n_a_As76_gs_90deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_90deg, 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)
    Br79_n_a_As76_gs_135deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_135deg, 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)

    Br79_n_a_As76_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_45deg[Br79_n_a_As76_gs_45deg_valid], 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)[0]
    Br79_n_a_As76_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_90deg[Br79_n_a_As76_gs_90deg_valid], 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)[0]
    Br79_n_a_As76_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_135deg[Br79_n_a_As76_gs_135deg_valid], 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(6130)$ to $^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $\\theta_\\alpha=0\\,$deg")
    ax.set_ylabel("$E_{\\alpha}$ [keV]")
    ax.set_xlabel("Incoming $E_{\\alpha}$ [keV]")
    ax.plot(lab_energy[C13_a_n_O16_6130_45deg_valid][Br79_n_a_As76_gs_45deg_valid], Br79_n_a_As76_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6130_90deg_valid][Br79_n_a_As76_gs_90deg_valid], Br79_n_a_As76_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6130_135deg_valid][Br79_n_a_As76_gs_135deg_valid], Br79_n_a_As76_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_n_O16_6130_Br79_n_a_As76.png", bbox_inches="tight", dpi=400)

    Br81_n_a_As78_gs_45deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_45deg, 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)
    Br81_n_a_As78_gs_90deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_90deg, 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)
    Br81_n_a_As78_gs_135deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6130_135deg, 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)

    Br81_n_a_As78_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_45deg[Br81_n_a_As78_gs_45deg_valid], 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)[0]
    Br81_n_a_As78_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_90deg[Br81_n_a_As78_gs_90deg_valid], 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)[0]
    Br81_n_a_As78_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6130_135deg[Br81_n_a_As78_gs_135deg_valid], 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(6130)$ to $^{81}$Br$(n,\\alpha)^{81}$Se$(\\mathrm{g.s.})$ $\\theta_\\alpha=0\\,$deg")
    ax.set_ylabel("$E_{\\alpha}$ [keV]")
    ax.set_xlabel("Incoming $E_{\\alpha}$ [keV]")
    ax.plot(lab_energy[C13_a_n_O16_6130_45deg_valid][Br81_n_a_As78_gs_45deg_valid], Br81_n_a_As78_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6130_90deg_valid][Br81_n_a_As78_gs_90deg_valid], Br81_n_a_As78_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6130_135deg_valid][Br81_n_a_As78_gs_135deg_valid], Br81_n_a_As78_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend() 

    fig.savefig("figs/C13_a_n_O16_6130_Br81_n_a_As78.png", bbox_inches="tight", dpi=400)

    ax_45deg.plot(lab_energy[C13_a_n_O16_6130_45deg_valid][Br79_n_p_Se79_gs_45deg_valid], Br79_n_p_Se79_gs_45deg, linestyle="--", color="black", label="$^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")
    ax_45deg.plot(lab_energy[C13_a_n_O16_6130_45deg_valid][Br81_n_p_Se81_gs_45deg_valid], Br81_n_p_Se81_gs_45deg, linestyle="--", color="blue", label="$^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")
    ax_45deg.plot(lab_energy[C13_a_n_O16_6130_45deg_valid][Br79_n_a_As76_gs_45deg_valid], Br79_n_a_As76_gs_45deg, linestyle="--", color="green", label="$^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")
    ax_45deg.plot(lab_energy[C13_a_n_O16_6130_45deg_valid][Br81_n_a_As78_gs_45deg_valid], Br81_n_a_As78_gs_45deg, linestyle="--", color="cyan", label="$^{81}$Br$(n,\\alpha)^{78}$As$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")

    ax_90deg.plot(lab_energy[C13_a_n_O16_6130_90deg_valid][Br79_n_p_Se79_gs_90deg_valid], Br79_n_p_Se79_gs_90deg, linestyle="--", color="black", label="$^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")
    ax_90deg.plot(lab_energy[C13_a_n_O16_6130_90deg_valid][Br81_n_p_Se81_gs_90deg_valid], Br81_n_p_Se81_gs_90deg, linestyle="--", color="blue", label="$^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")
    ax_90deg.plot(lab_energy[C13_a_n_O16_6130_90deg_valid][Br79_n_a_As76_gs_90deg_valid], Br79_n_a_As76_gs_90deg, linestyle="--", color="green", label="$^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")
    ax_90deg.plot(lab_energy[C13_a_n_O16_6130_90deg_valid][Br81_n_a_As78_gs_90deg_valid], Br81_n_a_As78_gs_90deg, linestyle="--", color="cyan", label="$^{81}$Br$(n,\\alpha)^{78}$As$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")

    ax_135deg.plot(lab_energy[C13_a_n_O16_6130_135deg_valid][Br79_n_p_Se79_gs_135deg_valid], Br79_n_p_Se79_gs_135deg, linestyle="--", color="black", label="$^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")
    ax_135deg.plot(lab_energy[C13_a_n_O16_6130_135deg_valid][Br81_n_p_Se81_gs_135deg_valid], Br81_n_p_Se81_gs_135deg, linestyle="--", color="blue", label="$^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")
    ax_135deg.plot(lab_energy[C13_a_n_O16_6130_135deg_valid][Br79_n_a_As76_gs_135deg_valid], Br79_n_a_As76_gs_135deg, linestyle="--", color="green", label="$^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")
    ax_135deg.plot(lab_energy[C13_a_n_O16_6130_135deg_valid][Br81_n_a_As78_gs_135deg_valid], Br81_n_a_As78_gs_135deg, linestyle="--", color="cyan", label="$^{81}$Br$(n,\\alpha)^{78}$As$(\\mathrm{g.s.})$ $^{16}$O$(6130)$")



    ################
    ################
    # C13_a_n_O16_6917

    C13_a_n_O16_6917_45deg_valid: list[bool] = has_enough_energy(lab_energy, 45, He4_mass, C13_mass, neutron_mass, O16_mass, 6917)
    C13_a_n_O16_6917_90deg_valid: list[bool] = has_enough_energy(lab_energy, 90, He4_mass, C13_mass, neutron_mass, O16_mass, 6917)
    C13_a_n_O16_6917_135deg_valid: list[bool] = has_enough_energy(lab_energy, 135, He4_mass, C13_mass, neutron_mass, O16_mass, 6917)

    C13_a_n_O16_6917_45deg: NDArray[np.float64] = eject_lab_energy(lab_energy[C13_a_n_O16_6917_45deg_valid], 45, He4_mass, C13_mass, neutron_mass, O16_mass, 6917)[0]
    C13_a_n_O16_6917_90deg: NDArray[np.float64] = eject_lab_energy(lab_energy[C13_a_n_O16_6917_90deg_valid], 90, He4_mass, C13_mass, neutron_mass, O16_mass, 6917)[0]
    C13_a_n_O16_6917_135deg: NDArray[np.float64] = eject_lab_energy(lab_energy[C13_a_n_O16_6917_135deg_valid], 135, He4_mass, C13_mass, neutron_mass, O16_mass, 6917)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(6917)$")
    ax.set_ylabel("$E_{n}$ [keV]")
    ax.set_xlabel("$E_{\\alpha}$ [keV]")
    ax.plot(lab_energy[C13_a_n_O16_6917_45deg_valid], C13_a_n_O16_6917_45deg, linestyle="-", color="black", label="$\\theta_\\mathrm{LAB}=45\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6917_90deg_valid], C13_a_n_O16_6917_90deg, linestyle="--", color="red", label="$\\theta_\\mathrm{LAB}=90\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6917_135deg_valid], C13_a_n_O16_6917_135deg, linestyle="-.", color="blue", label="$\\theta_\\mathrm{LAB}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_g_O16_6917.png", bbox_inches="tight", dpi=400)

    Br79_n_p_Se79_gs_45deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_45deg, 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)
    Br79_n_p_Se79_gs_90deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_90deg, 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)
    Br79_n_p_Se79_gs_135deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_135deg, 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)

    Br79_n_p_Se79_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_45deg[Br79_n_p_Se79_gs_45deg_valid], 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)[0]
    Br79_n_p_Se79_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_90deg[Br79_n_p_Se79_gs_90deg_valid], 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)[0]
    Br79_n_p_Se79_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_135deg[Br79_n_p_Se79_gs_135deg_valid], 0, neutron_mass, Br79_mass, H1_mass, Se79_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(6917)$ to $^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $\\theta_p=0\\,$deg")
    ax.set_ylabel("$E_{p}$ [keV]")
    ax.set_xlabel("$E_{\\alpha}$ [keV]")
    ax.plot(lab_energy[C13_a_n_O16_6917_45deg_valid][Br79_n_p_Se79_gs_45deg_valid], Br79_n_p_Se79_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6917_90deg_valid][Br79_n_p_Se79_gs_90deg_valid], Br79_n_p_Se79_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6917_135deg_valid][Br79_n_p_Se79_gs_135deg_valid], Br79_n_p_Se79_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_n_O16_6917_Br79_n_p_Se79.png", bbox_inches="tight", dpi=400)

    Br81_n_p_Se81_gs_45deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_45deg, 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)
    Br81_n_p_Se81_gs_90deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_90deg, 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)
    Br81_n_p_Se81_gs_135deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_135deg, 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)

    Br81_n_p_Se81_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_45deg[Br81_n_p_Se81_gs_45deg_valid], 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)[0]
    Br81_n_p_Se81_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_90deg[Br81_n_p_Se81_gs_90deg_valid], 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)[0]
    Br81_n_p_Se81_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_135deg[Br81_n_p_Se81_gs_135deg_valid], 0, neutron_mass, Br81_mass, H1_mass, Se81_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(6917)$ to $^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $\\theta_p=0\\,$deg")
    ax.set_ylabel("$E_{p}$ [keV]")
    ax.set_xlabel("$E_{\\alpha}$ [keV]")
    ax.plot(lab_energy[C13_a_n_O16_6917_45deg_valid][Br81_n_p_Se81_gs_45deg_valid], Br81_n_p_Se81_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6917_90deg_valid][Br81_n_p_Se81_gs_90deg_valid], Br81_n_p_Se81_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6917_135deg_valid][Br81_n_p_Se81_gs_135deg_valid], Br81_n_p_Se81_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_n_O16_6917_Br81_n_p_Se81.png", bbox_inches="tight", dpi=400)

    Br79_n_a_As76_gs_45deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_45deg, 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)
    Br79_n_a_As76_gs_90deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_90deg, 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)
    Br79_n_a_As76_gs_135deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_135deg, 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)

    Br79_n_a_As76_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_45deg[Br79_n_a_As76_gs_45deg_valid], 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)[0]
    Br79_n_a_As76_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_90deg[Br79_n_a_As76_gs_90deg_valid], 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)[0]
    Br79_n_a_As76_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_135deg[Br79_n_a_As76_gs_135deg_valid], 0, neutron_mass, Br79_mass, He4_mass, As76_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(6917)$ to $^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $\\theta_\\alpha=0\\,$deg")
    ax.set_ylabel("$E_{\\alpha}$ [keV]")
    ax.set_xlabel("Incoming $E_{\\alpha}$ [keV]")
    ax.plot(lab_energy[C13_a_n_O16_6917_45deg_valid][Br79_n_a_As76_gs_45deg_valid], Br79_n_a_As76_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6917_90deg_valid][Br79_n_a_As76_gs_90deg_valid], Br79_n_a_As76_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6917_135deg_valid][Br79_n_a_As76_gs_135deg_valid], Br79_n_a_As76_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend()

    fig.savefig("figs/C13_a_n_O16_6917_Br79_n_a_As76.png", bbox_inches="tight", dpi=400)

    Br81_n_a_As78_gs_45deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_45deg, 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)
    Br81_n_a_As78_gs_90deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_90deg, 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)
    Br81_n_a_As78_gs_135deg_valid: list[bool] = has_enough_energy(C13_a_n_O16_6917_135deg, 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)

    Br81_n_a_As78_gs_45deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_45deg[Br81_n_a_As78_gs_45deg_valid], 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)[0]
    Br81_n_a_As78_gs_90deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_90deg[Br81_n_a_As78_gs_90deg_valid], 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)[0]
    Br81_n_a_As78_gs_135deg: NDArray[np.float64] = eject_lab_energy(C13_a_n_O16_6917_135deg[Br81_n_a_As78_gs_135deg_valid], 0, neutron_mass, Br81_mass, He4_mass, As78_mass, 0)[0]

    fig, ax = plt.subplots(1, 1, figsize=(9,7))
    ax.set_title("$^{13}$C$(\\alpha,n)^{16}$O$(6917)$ to $^{81}$Br$(n,\\alpha)^{81}$Se$(\\mathrm{g.s.})$ $\\theta_\\alpha=0\\,$deg")
    ax.set_ylabel("$E_{\\alpha}$ [keV]")
    ax.set_xlabel("Incoming $E_{\\alpha}$ [keV]")
    ax.plot(lab_energy[C13_a_n_O16_6917_45deg_valid][Br81_n_a_As78_gs_45deg_valid], Br81_n_a_As78_gs_45deg, linestyle="-", color="black", label="$\\theta_{n,\\mathrm{LAB}}=45\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6917_90deg_valid][Br81_n_a_As78_gs_90deg_valid], Br81_n_a_As78_gs_90deg, linestyle="--", color="red", label="$\\theta_{n,\\mathrm{LAB}}=90\\,$deg")
    ax.plot(lab_energy[C13_a_n_O16_6917_135deg_valid][Br81_n_a_As78_gs_135deg_valid], Br81_n_a_As78_gs_135deg, linestyle="-.", color="blue", label="$\\theta_{n,\\mathrm{LAB}}=135\\,$deg")
    ax.legend() 

    fig.savefig("figs/C13_a_n_O16_6917_Br81_n_a_As78.png", bbox_inches="tight", dpi=400)

    ax_45deg.plot(lab_energy[C13_a_n_O16_6917_45deg_valid][Br79_n_p_Se79_gs_45deg_valid], Br79_n_p_Se79_gs_45deg, linestyle="-.", color="black", label="$^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")
    ax_45deg.plot(lab_energy[C13_a_n_O16_6917_45deg_valid][Br81_n_p_Se81_gs_45deg_valid], Br81_n_p_Se81_gs_45deg, linestyle="-.", color="blue", label="$^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")
    ax_45deg.plot(lab_energy[C13_a_n_O16_6917_45deg_valid][Br79_n_a_As76_gs_45deg_valid], Br79_n_a_As76_gs_45deg, linestyle="-.", color="green", label="$^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")
    ax_45deg.plot(lab_energy[C13_a_n_O16_6917_45deg_valid][Br81_n_a_As78_gs_45deg_valid], Br81_n_a_As78_gs_45deg, linestyle="-.", color="cyan", label="$^{81}$Br$(n,\\alpha)^{78}$As$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")

    ax_90deg.plot(lab_energy[C13_a_n_O16_6917_90deg_valid][Br79_n_p_Se79_gs_90deg_valid], Br79_n_p_Se79_gs_90deg, linestyle="-.", color="black", label="$^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")
    ax_90deg.plot(lab_energy[C13_a_n_O16_6917_90deg_valid][Br81_n_p_Se81_gs_90deg_valid], Br81_n_p_Se81_gs_90deg, linestyle="-.", color="blue", label="$^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")
    ax_90deg.plot(lab_energy[C13_a_n_O16_6917_90deg_valid][Br79_n_a_As76_gs_90deg_valid], Br79_n_a_As76_gs_90deg, linestyle="-.", color="green", label="$^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")
    ax_90deg.plot(lab_energy[C13_a_n_O16_6917_90deg_valid][Br81_n_a_As78_gs_90deg_valid], Br81_n_a_As78_gs_90deg, linestyle="-.", color="cyan", label="$^{81}$Br$(n,\\alpha)^{78}$As$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")

    ax_135deg.plot(lab_energy[C13_a_n_O16_6917_135deg_valid][Br79_n_p_Se79_gs_135deg_valid], Br79_n_p_Se79_gs_135deg, linestyle="-.", color="black", label="$^{79}$Br$(n,p)^{79}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")
    ax_135deg.plot(lab_energy[C13_a_n_O16_6917_135deg_valid][Br81_n_p_Se81_gs_135deg_valid], Br81_n_p_Se81_gs_135deg, linestyle="-.", color="blue", label="$^{81}$Br$(n,p)^{81}$Se$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")
    ax_135deg.plot(lab_energy[C13_a_n_O16_6917_135deg_valid][Br79_n_a_As76_gs_135deg_valid], Br79_n_a_As76_gs_135deg, linestyle="-.", color="green", label="$^{79}$Br$(n,\\alpha)^{76}$As$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")
    ax_135deg.plot(lab_energy[C13_a_n_O16_6917_135deg_valid][Br81_n_a_As78_gs_135deg_valid], Br81_n_a_As78_gs_135deg, linestyle="-.", color="cyan", label="$^{81}$Br$(n,\\alpha)^{78}$As$(\\mathrm{g.s.})$ $^{16}$O$(6917)$")

    ax_45deg.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax_90deg.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax_135deg.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig_45deg.savefig("figs/peaks_45deg.png", bbox_inches="tight", dpi=400)
    fig_90deg.savefig("figs/peaks_90deg.png", bbox_inches="tight", dpi=400)
    fig_135deg.savefig("figs/peaks_135deg.png", bbox_inches="tight", dpi=400)

    return

if __name__ == "__main__":
    main()
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

current_dir = os.getcwd()
file_dir = os.path.join(current_dir, "figures")

mub = 0.05788  # meV/T

def hamiltonian(bfield, theta, deltaSO, deltaKK, deltaSV, gs, gv, delta_orb,bpara_off=0.0, bortho_off=0.0):
    bpara = np.cos(theta) * bfield + bpara_off
    bortho = np.sin(theta) * bfield + bortho_off
    ezv = 0.5 * mub * gv * bortho
    esz = 0.5 * mub * gs * bortho
    ham = np.array([[0.5 * deltaSO + ezv + esz + delta_orb, 0.5 * gs * mub * bpara, deltaKK, deltaSV],  # K up
                    [0.5 * gs * mub * bpara, -0.5 * deltaSO + ezv - esz + delta_orb, deltaSV, deltaKK],  # K down
                    [deltaKK, deltaSV, -0.5 * deltaSO - ezv + esz + delta_orb, 0.5 * gs * mub * bpara],  # K' up
                    [deltaSV, deltaKK, 0.5 * gs * mub * bpara, 0.5 * deltaSO - ezv - esz + delta_orb]])  # K' down
    return ham


def calc_Bfield_dispersion(bfields, theta, deltaSO, deltaKK, deltaSV, gs, gv, delta_orb,bpara_off=0.0, bortho_off=0.0):
    eigen_energies = []
    eigen_vectors = []
    eigen_energies_h = []
    eigen_vectors_h = []
    diff_to_Kup = []
    for bfield in bfields:
        H = hamiltonian(bfield, theta, deltaSO, deltaKK, deltaSV, gs, gv, delta_orb, bpara_off, bortho_off)
        H_h = -hamiltonian(bfield, theta, deltaSO, deltaKK, deltaSV, gs, gv, 0, bpara_off, bortho_off)
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals_h, eigvecs_h = np.linalg.eig(H_h)
        eigen_energies.append(eigvals)
        eigen_vectors.append(eigvecs)
        eigen_energies_h.append(eigvals_h)
        eigen_vectors_h.append(eigvecs_h)
        diff_to_Kup.append(np.sqrt((eigvals[:, np.newaxis] - np.flip(eigvals_h))**2))
        #diff_to_Kup.append(np.abs((eigvals - eigvals[0])))
    return np.array(eigen_energies), np.array(eigen_vectors), np.array(eigen_energies_h), np.array(eigen_vectors_h), np.array(diff_to_Kup)


def main(coloring="band"):
    # Energies in meV
    deltaSO = -60 * 10 ** -3
    deltaKK = 0.0 * 10 ** -3
    deltaSV = 0.0 * 10 ** -3
    delta_orb = 1.6
    gs = 2.0
    gv = 14
    theta = np.deg2rad(0)
    bfields = np.arange(-.1, 1.5, 0.001)
    eigen_energies, eigen_vectors, eigen_energies_h, eigen_vectors_h, diff = calc_Bfield_dispersion(
        bfields, theta, deltaSO, deltaKK, deltaSV, gs, gv, delta_orb, bpara_off=0.0, bortho_off=0.5
    )

    plt.figure()

    if coloring == "band":
        # Color by band index
        for band_idx, band in enumerate(eigen_energies.T):
            plt.scatter(bfields, band, s=1.0, label=f'Band {band_idx}', cmap='viridis')
        for band_idx, band in enumerate(eigen_energies_h.T):
            plt.scatter(bfields, band, s=1.0, label=f'Band {band_idx}', cmap='viridis')

    elif coloring == "elho":
        # Color by band index
        for band_idx, band in enumerate(eigen_energies.T):
            plt.scatter(bfields, band, s=1.0, label=f'Band {band_idx}', color='mediumblue')
        for band_idx, band in enumerate(eigen_energies_h.T):
            plt.scatter(bfields, band, s=1.0, label=f'Band {band_idx}', color='darkred')


    elif coloring == "spin":
        # Color by spin projection (0, 2 for spin up, 1, 3 for spin down)
        norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalize to [0, 1]
        for band_idx, band in enumerate(eigen_energies.T):
            spin_projection = abs(np.abs(eigen_vectors[:, band_idx, 0]) ** 2 + np.abs(
                eigen_vectors[:, band_idx, 2]) ** 2)   # Spin up projection
            plt.scatter(bfields, band, c=spin_projection, s=1.0, cmap='coolwarm', norm=norm,
                        label=f'Spin projection band {band_idx}')

    elif coloring == "valley":
        # Color by valley projection (0, 1 for K valley, 2, 3 for K' valley)
        norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalize to [0, 1]
        for band_idx, band in enumerate(eigen_energies.T):
            valley_projection = np.abs(eigen_vectors[:, band_idx, 0]) ** 2 + np.abs(
                eigen_vectors[:, band_idx, 1]) ** 2  # K valley projection
            plt.scatter(bfields, band, c=valley_projection, s=1.0, cmap='plasma', norm=norm,
                        label=f'Valley projection band {band_idx}')

    #plt.colorbar(label='Projection')
    plt.xlabel('B-field (T)')
    plt.ylabel('Energy (meV)')
    #plt.legend(loc='upper right')
    plt.show()

    plt.figure()
    """
    for idx, dif in enumerate(diff):
        plt.scatter(bfields[idx], dif[0, 0], s=1.0, label=f'GS-GS')
        plt.scatter([bfields[idx]] * 2, [dif[1, 0], dif[0, 1]], s=1.0, label=f'$\nu$')
        plt.scatter([bfields[idx]] * 2, [dif[2, 0], dif[0, 2]], s=1.0, label=f'$\alpha$')
        plt.scatter([bfields[idx]] * 2, [dif[3, 1], dif[1, 3]], s=1.0, label=f'$\beta$')
        plt.scatter([bfields[idx]] * 4, [dif[3, 0], dif[0, 3], dif[2, 1], dif[1, 2]], s=1.0, label=f'$\gamma$')
        plt.scatter([bfields[idx]] * 3, [dif[1, 1], dif[2, 2], dif[3, 3]], s=1.0, label=f'Spin-Valley')
        plt.scatter([bfields[idx]] * 2, [dif[3, 2], dif[2, 3]], s=1.0, label=f'Valley')
    """
    y_gs_gs = diff[:, 0, 0]  # GS-GS
    y_nu = np.vstack([diff[:, 1, 0], diff[:, 0, 1]]).T  # ν
    y_alpha = np.vstack([diff[:, 2, 0], diff[:, 0, 2]]).T  # α
    y_beta = np.vstack([diff[:, 3, 1], diff[:, 1, 3]]).T  # β
    #y_gamma = np.vstack([diff[:, 3, 0], diff[:, 0, 3], diff[:, 2, 1], diff[:, 1, 2]]).T  # γ
    y_spin_valley = np.vstack([diff[:, 1, 1], diff[:, 2, 2], diff[:, 3, 3]]).T  # Spin-Valley
    y_valley = np.vstack([diff[:, 3, 2], diff[:, 2, 3]]).T  # Valley

    # Now plot each array against bfields
    plt.figure(figsize=(12, 6))
    plt.scatter(bfields, y_gs_gs, s=1.0, label='GS-GS')
    plt.scatter(np.repeat(bfields, 2), y_nu.flatten(), s=1.0, label='$\\nu$')
    #plt.scatter(np.repeat(bfields, 2), y_alpha.flatten(), s=1.0, label='$\\alpha$')
    plt.scatter(np.repeat(bfields, 2), y_beta.flatten(), s=1.0, label='$\\beta$')
    #plt.scatter(np.repeat(bfields, 4), y_gamma.flatten(), s=1.0, label='$\\gamma$')
    plt.scatter(np.repeat(bfields, 3), y_spin_valley.flatten(), s=1.0, label='Spin-Valley')
    plt.scatter(np.repeat(bfields, 2), y_valley.flatten(), s=1.0, label='Valley')
    plt.xlabel('B-field (T)')
    plt.ylabel('$\Delta E$ (meV)')
    plt.legend(loc='upper right', markerscale=2.5)
    plt.savefig(os.path.join(file_dir, r'energy_diagram_500mT.svg'))


if __name__ == "__main__":
    # You can change "coloring" to "spin" or "valley" to change the coloring mode
    main(coloring="elho")  # Options: "band", "spin", "valley"

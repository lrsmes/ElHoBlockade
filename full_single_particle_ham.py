import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

mub = 0.05788  # meV/T

def hamiltonian(bfield, theta, deltaSO, deltaKK, deltaSV, gs, gv, bpara_off=0.0, bortho_off=0.0):
    bpara = np.cos(theta) * bfield + bpara_off
    bortho = np.sin(theta) * bfield + bortho_off
    ezv = 0.5 * mub * gv * bortho
    esz = 0.5 * mub * gs * bortho
    ham = np.array([[0.5 * deltaSO + ezv + esz, 0.5 * gs * mub * bpara, deltaKK, deltaSV],  # K up
                    [0.5 * gs * mub * bpara, -0.5 * deltaSO + ezv - esz, deltaSV, deltaKK],  # K down
                    [deltaKK, deltaSV, -0.5 * deltaSO - ezv + esz, 0.5 * gs * mub * bpara],  # K' up
                    [deltaSV, deltaKK, 0.5 * gs * mub * bpara, 0.5 * deltaSO - ezv - esz]])  # K' down
    return ham


def calc_Bfield_dispersion(bfields, theta, deltaSO, deltaKK, deltaSV, gs, gv, bpara_off=0.0, bortho_off=0.0):
    eigen_energies = []
    eigen_vectors = []
    for bfield in bfields:
        H = hamiltonian(bfield, theta, deltaSO, deltaKK, deltaSV, gs, gv, bpara_off, bortho_off)
        eigvals, eigvecs = np.linalg.eigh(H)
        eigen_energies.append(eigvals)
        eigen_vectors.append(eigvecs)
    return np.array(eigen_energies), np.array(eigen_vectors)


def main(coloring="band"):
    # Energies in meV
    deltaSO = -60 * 10 ** -3
    deltaKK = 5.0 * 10 ** -3
    deltaSV = 5.0 * 10 ** -3
    gs = 2.0
    gv = 15.0
    theta = np.deg2rad(0)
    bfields = np.arange(-.1, 3.4, 0.001)
    eigen_energies, eigen_vectors = calc_Bfield_dispersion(bfields, theta, deltaSO, deltaKK, deltaSV, gs, gv,
                                                           bpara_off=0.0, bortho_off=0.5)

    plt.figure()

    if coloring == "band":
        # Color by band index
        for band_idx, band in enumerate(eigen_energies.T):
            plt.scatter(bfields, band, s=1.0, label=f'Band {band_idx}', cmap='viridis')

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

    plt.colorbar(label='Projection')
    plt.xlabel('B-field (T)')
    plt.ylabel('Energy (meV)')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    # You can change "coloring" to "spin" or "valley" to change the coloring mode
    main(coloring="band")  # Options: "band", "spin", "valley"

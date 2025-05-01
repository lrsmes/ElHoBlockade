import numpy as np
from scipy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

mub = 0.05788  # meV/T

def track_eigs_over_bfields(Bs, theta, deltaSO, deltaKK, deltaSV, gs, gv, bpara_off=0.0, bortho_off=0.0, el_hl_fac=1.0):
    W = np.zeros((len(Bs), 4), dtype=complex)
    V = np.zeros((len(Bs), 4, 4), dtype=complex)

    H0 = el_hl_fac *  hamiltonian(Bs[0], theta, deltaSO, deltaKK, deltaSV, gs, gv, bpara_off, bortho_off)
    w_prev, V_prev = np.linalg.eig(H0)
    W[0] = w_prev
    V[0] = V_prev

    for i, B in enumerate(Bs[1:], start=1):
        H = hamiltonian(B, theta, deltaSO, deltaKK, deltaSV, gs, gv)
        w_cur, V_cur = np.linalg.eig(H)
        W[i] = w_cur
        V[i] = V_cur



    return W, V

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

def calc_Bfield_dispersion(bfields, theta, deltaSO, deltaKK, deltaSV, gs, gv, bpara_off=0.0, bortho_off=0.0, hole_fac=1.0):
    eigen_energies = []
    eigen_vectors = []
    prev_eigenvectors = None
    prev_eigenvalues = None

    for bfield in bfields:
        H = hole_fac * hamiltonian(bfield, theta, deltaSO, deltaKK, deltaSV, gs, gv, bpara_off, bortho_off)
        eigvals, eigvecs = la.eigh(H)

        if prev_eigenvalues is not None:
            # Match current eigenvalues to the previous step based on minimum distance
            mapping = []
            used_indices = set()
            for prev_idx, prev_val in enumerate(prev_eigenvalues):
                min_distance = float('inf')
                best_match = None
                for curr_idx, curr_val in enumerate(eigvals):
                    if curr_idx in used_indices:
                        continue
                    distance = abs(curr_val - prev_val)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = curr_idx
                mapping.append(best_match)
                used_indices.add(best_match)

            # Reorder eigenvalues and eigenvectors according to the mapping
            eigvals = eigvals[mapping]
            eigvecs = eigvecs[:, mapping]

            # Check and fix eigenvector sign continuity
            for i in range(len(eigvals)):
                dot_product = np.dot(prev_eigenvectors[:, i], eigvecs[:, i])
                if dot_product < 0:  # Flip sign to maintain continuity
                    eigvecs[:, i] *= -1

        prev_eigenvalues = eigvals
        prev_eigenvectors = eigvecs

        eigen_energies.append(eigvals)
        eigen_vectors.append(eigvecs)

    return np.array(eigen_energies), np.array(eigen_vectors)

def main(coloring="band"):
    # Energies in meV
    deltaSO = -60 * 10 ** -3
    deltaKK = 0.1 * 10 ** -3
    deltaSV = 0.000 * 10 ** -3
    gs = 2.0
    gv = 15.0
    theta = np.deg2rad(90.0)
    bfields = np.arange(.01, .1, 0.0005)
    print(len(bfields))
    eigen_energies, eigen_vectors = calc_Bfield_dispersion(bfields, theta, deltaSO, deltaKK, deltaSV, gs, gv,
                                                           bpara_off=.0, bortho_off=0.000, hole_fac=-1) # correct for valley or spin projections

    eigen_energies_2, eigen_vectors_2 = calc_Bfield_dispersion(bfields, theta, deltaSO, deltaKK, deltaSV, gs, gv,
                                                           bpara_off=.0, bortho_off=0.000, hole_fac=1) # correct for valley or spin projections

    eigen_energies, eigen_vectors = track_eigs_over_bfields(bfields, theta, deltaSO, deltaKK, deltaSV, gs, gv,
                                                           bpara_off=.0, bortho_off=.0, el_hl_fac=1) # correct tracking eigenvalues

    eigen_energies_2, eigen_vectors_2 =  track_eigs_over_bfields(bfields, theta, deltaSO+0*10**-3, -1 * deltaKK + 5.* 10**-3, deltaSV, gs, gv,
                                                           bpara_off=.0, bortho_off=.0, el_hl_fac=-1)  # correct tracking eigenvalues

    plt.figure()

    if coloring == "band":
        # Color by band index
        for band_idx, band in enumerate(eigen_energies.T):
            plt.scatter(bfields, band, s=1.0, label=f'Band {band_idx} electron', cmap='viridis')
        for band_idx, band in enumerate(eigen_energies_2.T):
            plt.scatter(bfields, band-0.5 , s=1.0, label=f'Band {band_idx} hole', cmap='viridis')


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
    # define transitions via energy differences
    gs_gs = eigen_energies.T[3] - eigen_energies_2.T[2]
    alpha = eigen_energies.T[3] - eigen_energies_2.T[1]
    nu1 = eigen_energies.T[3] - eigen_energies_2.T[0]
    nu2 = eigen_energies.T[2] - eigen_energies_2.T[1]
    nu1_p = eigen_energies.T[0] - eigen_energies_2.T[3]
    nu2_p = eigen_energies.T[1] - eigen_energies_2.T[2]
    beta = eigen_energies.T[2] - eigen_energies_2.T[0]
    gamma = eigen_energies.T[3] - eigen_energies_2.T[3]
    plt.figure()
    plt.scatter(bfields, alpha , s=1.0, label=f'alpha')
    plt.scatter(bfields, beta , s=1.0, label=f'beta')
    plt.scatter(bfields, nu1 , s=1.0, label=f'nu1')
    plt.scatter(bfields, nu2, s=1.0, label=f'nu2')
    plt.scatter(bfields, nu1_p, s=1.0, label=f'nu1p')
    plt.scatter(bfields, nu2_p, s=1.0, label=f'nu2p')
    plt.scatter(bfields, gamma, s=1.0, label=f'gamma')
    plt.xlabel('B-field (T)')
    plt.ylabel('Energy (meV)')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    # You can change "coloring" to "spin" or "valley" to change the coloring mode
    main(coloring="valley")  # Options: "band", "spin", "valley"

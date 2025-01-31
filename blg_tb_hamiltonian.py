import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.cm import coolwarm, viridis, bwr
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def rotation_matrix_2d(alpha):
    return np.array([
        [np.cos(alpha), -1 * np.sin(alpha)],
        [np.sin(alpha), np.cos(alpha)]
    ])


def calculate_dos(eigen_energies, energy_range, bin_size=0.01):
    # approximate calculation of DOS along the line cut
    energy_bins = np.arange(energy_range[0], energy_range[1], bin_size)
    dos = np.zeros_like(energy_bins)
    eigen_energies = np.hstack(eigen_energies)
    # Calculate DOS
    for energy in eigen_energies:
        index = int((energy - energy_range[0]) / bin_size)
        if 0 <= index < len(dos):
            dos[index] += 1
    dos /= (bin_size * len(eigen_energies))  # Normalize by total number of states and bin width

    return energy_bins, dos


def interpolate(start, end, num_points):
    # Generate a linear interpolation between two points in 2D space
    return np.linspace(start, end, num_points, endpoint=False)


def high_symmetry_path_hex(a, total_points):
    gamma = np.array([0, 0])
    m = 2 * np.pi / a * np.array([1 / 2, np.sqrt(3) / 3])
    k = 2 * np.pi / a * np.array([2 / 3, 0])
    k_prime = rotation_matrix_2d(np.deg2rad(60)).dot(k) # not correct, has to be re-examined

    # Calculate the lengths of each segment
    segments = [
        (gamma, k),
        (k, m),
        (m, k_prime),
        (k_prime, gamma)
    ]
    segment_lengths = [np.linalg.norm(end - start) for start, end in segments]
    total_length = sum(segment_lengths)
    print(total_length/total_points)
    # Calculate the number of points for each segment
    points_per_segment = [int(total_points * (length / total_length)) for length in segment_lengths]

    # Ensure the total number of points is correct
    points_per_segment[-1] += total_points - sum(points_per_segment)

    # Generate the k-path
    gamma_k = interpolate(gamma, k, points_per_segment[0])
    k_m = interpolate(k, m, points_per_segment[1])
    m_k_prime = interpolate(m, k_prime, points_per_segment[2])
    k_prime_gamma = interpolate(k_prime, gamma, points_per_segment[3])

    k_path = np.concatenate((gamma_k, k_m, m_k_prime, k_prime_gamma), axis=0)

    return k_path


def structure_function(k, a):
    return np.exp(1j * a / np.sqrt(3) * k[1]) * (1 + 2 * np.exp(-1j * a * k[1] / np.sqrt(3)) * np.cos(0.5 * a * k[0]))


def tb_hamiltonian_blg(k, a, delta, V, gamma_0, gamma_1, gamma_3, gamma_4):
    sf = structure_function(k, a)
    h_intra_layer_1 = np.array([[delta / 2, gamma_0 * sf],
                              [np.conjugate(sf) * gamma_0, -delta /2]])
    h_intra_layer_2 = np.array([[-delta / 2, gamma_0 * sf],
                              [np.conjugate(sf) * gamma_0, delta / 2]])
    h_inter_layer = np.array([[gamma_4 * np.conjugate(sf), gamma_1],
                              [sf * gamma_3, gamma_4 * np.conjugate(sf)]])
    H_without_spin = np.block([
        [h_intra_layer_1 + np.eye(2) * V,  h_inter_layer],
        [(h_inter_layer.conj()).T, h_intra_layer_2 - np.eye(2) * V]
    ])
    return np.kron(H_without_spin, np.eye(2))


def soc_hamiltonian_blg(lamda_i1=0, lamda_i2=0, lamda_0=0, lamda_0_prime=0, lamda_4=0, layer_delta=0):
    h_soc_intra_layer_1 = np.array([
        [(lamda_i2 + layer_delta), 0, 0, 0],
        [0, -1 * (lamda_i2 + layer_delta), 1j * lamda_0, 0],
        [0, -1j * lamda_0, -1 * (lamda_i1 + layer_delta), 0],
        [0, 0, 0, (lamda_i1 + layer_delta)]
    ])
    h_soc_intra_layer_2 = -1 * np.array([
        [(lamda_i1), 0, 0, 0],
        [0, -1 * (lamda_i1), - 1j * lamda_0_prime, 0],
        [0, 1j * lamda_0_prime, -1 * (lamda_i2), 0],
        [0, 0, 0, (lamda_i2)]
    ])
    h_soc_inter_layer = np.array([
        [0, 0, 0, 0],
        [-1j * lamda_4, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, lamda_4]
    ])

    H_soc = np.block([
        [h_soc_intra_layer_1 , (h_soc_inter_layer.conj()).T],
        [h_soc_inter_layer, h_soc_intra_layer_2]
    ])
    return H_soc


def total_hamiltonian(k, a, delta, V, gamma_0, gamma_1, gamma_3, gamma_4,
                      lamda_i1=0, lamda_i2=0, lamda_0=0, lamda_4=0, lamda_0_prime=0, layer_delta=0):
    return (tb_hamiltonian_blg(k, a, delta, V, gamma_0, gamma_1, gamma_3, gamma_4) +
            soc_hamiltonian_blg(lamda_i1, lamda_i2, lamda_0,lamda_0_prime,lamda_4, layer_delta))


def calc_eigen_energies_and_vectors(k_path, a, delta, V, gamma_0, gamma_1, gamma_3, gamma_4,
                      lamda_i1=0, lamda_i2=0, lamda_0=0, lamda_4=0, lamda_0_prime=0, layer_delta=0):
    eigen_energies = []
    eigen_vectors = []
    for k in k_path:
        H = total_hamiltonian(k, a, delta, V, gamma_0, gamma_1, gamma_3, gamma_4,
                      lamda_i1, lamda_i2, lamda_0, lamda_4, lamda_0_prime, layer_delta)
        eigvals, eigvecs = np.linalg.eigh(H)
        eigen_energies.append(eigvals)
        eigen_vectors.append(eigvecs)
    return np.array(eigen_energies) * 1000, np.array(eigen_vectors)


def calc_eigen_energies(k_path, a, delta, V, gamma_0, gamma_1, gamma_3, gamma_4):
    # legacy for calculating TB wo SOC
    eigen_energies = []
    for k in k_path:
        H = tb_hamiltonian_blg(k, a, delta, V, gamma_0, gamma_1, gamma_3, gamma_4)
        eigvals = np.linalg.eigvalsh(H)
        eigen_energies.append(eigvals)
    return np.array(eigen_energies)


def main():
    #plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
    #plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'  # Load 'amsmath' package
    #plt.rcParams["font.family"] = "Arial"
    # TB parameters for 1V/nm in eV from Konschuh et. al. 2012: "Theory of spin-orbit coupling in bilayer graphene"
    cm = 1/2.54
    V = 0.005 / 2 # interlayer potential
    gamma_4 = -0.165
    gamma_3 = 0.25
    gamma_1 = 0.339
    gamma_0 = 2.6
    delta = -0.000
    energy_lim = [-0.01, 0.01]
    # SOC parameters (currently only valid for K)
    lamda_i1 = -0.5 * 10 ** -4
    lamda_i2 = -0.6 * 10 ** -4
    lamda_0 = 5 * 10 ** -5
    lamda_0_prime = 5 * 10 ** -5
    lamda_4 = -10 * 10 ** -5
    layer_delta = -0.0011 # delta spin-orbit coupling


    # Initialize
    n_points = 10001
    a = 2.36 # Angstrom
    k_path = high_symmetry_path_hex(a, n_points)

    indices = {
        'M': 3100,
        'K': 3175,
        'Gamma': 3250
    }

    # Calculate eigenenergies
    eigen_energies, eigen_vectors = calc_eigen_energies_and_vectors(k_path, a, delta, V, gamma_0, gamma_1, gamma_3,
                                                                    gamma_4, lamda_i1, lamda_i2, lamda_0, lamda_0_prime,
                                                                    lamda_4, layer_delta)
    np.save('eigen_energies_0.004.npy', eigen_energies)
    np.save('eigen_vec_0.004.npy', eigen_vectors)
    # Approx. Calculation of DOS (should give the right qualitative picture for low energy)
    energy_bins, dos = calculate_dos(eigen_energies / 1000, energy_lim, bin_size=0.00004)
    dos = gaussian_filter1d(dos, 7)

    # Plotting
    norm = Normalize(vmin=0, vmax=np.max(np.abs(eigen_vectors) ** 2))
    cmap = plt.get_cmap('bwr')  # Colormap

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Band structure plot
    for band_idx, band in enumerate(eigen_energies.T):
        # Color by the square of selected eigenvector components in the basis
        # {A1 Up, A1 Down, B1 Up, B1 Down, A2 Up, A2 Down, B2 Up, B2 Down}
        # e.g selecting indices 0, 1, 2, 3 gives layer polarization, 0, 2, 4, 6 spin polarization
        # and  0, 1, 4, 5 lattice site ploarization
        color_values = ((np.abs(eigen_vectors[:, band_idx, 0]) ** 2 + np.abs(eigen_vectors[:, band_idx, 2]) ** 2
                        + np.abs(eigen_vectors[:, band_idx, 4]) ** 2 + np.abs(eigen_vectors[:, band_idx, 6]) ** 2) /
                        ((np.abs(eigen_vectors[:, band_idx, 0]) ** 2 + np.abs(eigen_vectors[:, band_idx, 1]) ** 2
                        + np.abs(eigen_vectors[:, band_idx, 2]) ** 2 + np.abs(eigen_vectors[:, band_idx, 3]) ** 2) + (np.abs(eigen_vectors[:, band_idx, 4]) ** 2 + np.abs(eigen_vectors[:, band_idx, 5]) ** 2
                        + np.abs(eigen_vectors[:, band_idx, 6]) ** 2 + np.abs(eigen_vectors[:, band_idx, 7]) ** 2)))

        print(np.max(color_values), np.min(color_values))
        colors = bwr((color_values))  # Normalize and map to colors
        ax1.scatter(np.arange(band.size), band, color=colors, s=8)  # Use scatter for individual coloring

    ax1.set_title('Band Structure of Bilayer Graphene')
    ax1.set_xlim(3100, 3200)  # according to the most interesting range (currently centered around K)
    ax1.axvline(3175 + 36)
    ax1.axvline(3175 - 36)
    ax1.set_ylim(energy_lim[0] * 1000, energy_lim[1] * 1000)
    ax1.set_xlabel('')
    ax1.set_ylabel('Energy (meV)', fontsize=24)
    high_sym_labels = [r'$\Gamma$', 'K', 'M']
    high_sym_positions = [indices[key] for key in ['Gamma', 'K', 'M']]
    ax1.set_xticks(high_sym_positions)
    ax1.set_xticklabels(high_sym_labels)
    ax1.yaxis.set_major_locator(MaxNLocator(5))
    ax1.tick_params(axis='x', which='major', labelsize=24)
    ax1.tick_params(axis='y', which='major', labelsize=24)
    # Density of States plot
    ax2.plot(dos, energy_bins, color='r')
    ax2.set_title('Density of States')
    ax2.set_xlabel('DOS (states/eV/unit cell)')
    ax2.set_ylabel('Energy (eV)')
    ax2.set_ylim(energy_lim[0], energy_lim[1])

    # Create the scalar mappable and colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Required for ScalarMappable,<xq although not used directly
    cbar = plt.colorbar(sm, ax=ax1)  # Add colorbar to the band structure plot
    cbar.set_label('Color value (normalized)', fontsize=20)  # Label for the colorbar
    cbar.ax.tick_params(labelsize=16)  # Adjust colorbar tick labels size

    # Adjust figure size and layout
    fig.set_size_inches(7.5, 10.0)
    plt.tight_layout()
    plt.savefig('band_plot_' + str(V) + '.pdf')
    plt.show()




if __name__ == "__main__":
    main()
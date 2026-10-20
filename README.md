# `qbran`: Quantum BRanch ANalysis

**QBrAn** (**Q**uantum **Br**anch **An**alysis) is a high-performance Python library with minimal dependencies (NumPy + SciPy) for simulating driven superconducting circuits, identifying multi-photon resonances associated with Measurement-Induced State Transitions (MIST) and Drive-induced Unwanted State Transitions (DUST), and tracking parametric processes in circuit QED.

The library implements Floquet theory for time-periodic quantum systems governed by Hamiltonians of the form $\hat{H}(t) = \hat{H}_0 + \hat{V}\cos(\omega_d t)$ using two complementary numerical approaches:
1. **Extended Sambe–Shirley Space**: Builds a time-independent Floquet Hamiltonian $\hat{\mathcal{H}}_F$ in Fourier harmonic space using photon replicas of the bare Hilbert space, solved via sparse diagonalization of the block-tridiagonal Hermitian matrix.
2. **Propagator Time Integration**: Employs a custom 4th-order Trotter–Suzuki decomposition with symplectic Yoshida operator splitting to integrate $\hat{H}(t)$ over one drive period, obtaining the operator $\hat{U}(T, 0)$. Diagonalization of $\hat{U}(T, 0)$ yields the Floquet modes and quasienergies. This framework builds upon D. K. Weiss's [floquet](https://github.com/dkweiss31/floquet) library, with custom solver and caching providing a performance speedup of up to 17×.

---

## Installation

The library has minimal dependencies (`numpy`, `scipy`, `h5py`, `tqdm`) and does not require heavyweight quantum simulation packages like `qutip` or `scqubits`, while providing an intuitive, familiar syntax.

```bash
pip install git+https://github.com/AlessandroMiotto/qbran.git
```

---

## Quickstart

```python
import numpy as np
import qbran as br

# 1. Define circuit components
# Parameters from Dai et al., Phys. Rev. X 16, 011011 (2026)
# H_t + H_r = 4 * EC * (̂n - n_g)² - EJ * cos(φ) + ω_r * ̂a†̂a
tmon = br.Transmon(EJ=16.40, EC=0.1695, ng=0.0, truncated_dim=13)
res  = br.Resonator(E_osc=9.029, truncated_dim=3)

# 2. Specify circuit couplings
# H_0 = H_t + H_r + g * (̂n - n_g)( ̂a† + ̂a )
circuit = br.HilbertSpace([tmon, res])
circuit.add_interaction(tmon, res, g=0.153)

# 3. Specify drive frequencies and amplitudes
# H_d(t) = Ω_d * cos(ω_d t) * ̂n
freq_drive = np.linspace(10.2, 11.4, 100)
amps_drive = np.linspace(   0,  5.2, 100)

# 4. Perform branch analysis using propagator time integration
opt = br.Options(n_workers=4)
analysis = circuit.branch_analysis(kind="propagator", options=opt)
results = analysis.run(drive_freq=freq_drive, drive_amp=amps_drive)

# 5. Access results
quasienergies   = results.quasienergies
avg_excitations = results.avg_excitation
```

Additional tutorial notebooks are available in [`examples/`](./examples):
* `01_driven_transmon.ipynb`: Semiclassical driven transmon and Floquet branch analysis (M. F. Dumas et al. example).
* `02_hybridization.ipynb`: Multi-mode circuits, hybridization, 2D scar maps (W. Dai et al. example).
* `03_parametric_sweep.ipynb`: Multi-dimensional parameter sweeps (W. Dai et al. example).
* `04_circuit_fit.ipynb`: Fitting experimental parameters (Single Microwave Photon Detector example).

---

## References

* M. F. Dumas, B. Groleau-Paré, A. McDonald, M. H. Muñoz-Arias, C. Lledó, B. D'Anjou, and A. Blais, *"Measurement-induced transmon ionization"*, [Phys. Rev. X **14**, 041023 (2024)](https://doi.org/10.1103/PhysRevX.14.041023) (arXiv:2402.06615).
* X. Xiao, J. Venkatraman, R. G. Cortiñas, S. Chowdhury, and M. H. Devoret, *"A diagrammatic method to compute the effective Hamiltonian of driven nonlinear oscillators"*, [PRX Quantum **4**, 040337 (2023)](https://doi.org/10.1103/PRXQuantum.4.040337) (arXiv:2304.13656).
* W. Dai, S. Hazra, D. K. Weiss, P. D. Kurilovich, T. Connolly, H. K. Babla, S. Singh, V. R. Joshi, A. Z. Ding, P. D. Parakh, J. Venkatraman, X. Xiao, L. Frunzio, and M. H. Devoret, *"Characterization of Drive-Induced Unwanted State Transitions in Superconducting Circuits"*, [Phys. Rev. X **16**, 011011 (2026)](https://doi.org/10.1103/zdpg-mhpc) (arXiv:2506.24070).
* D. K. Weiss, *"Floquet: Identifying nonlinear resonances in quantum systems due to parametric drives"*, GitHub repository: [dkweiss31/floquet](https://github.com/dkweiss31/floquet).
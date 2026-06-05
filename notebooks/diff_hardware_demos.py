import os
import matplotlib.pyplot as plt
import noisiq as nq
from noisiq.noise import fill_idle_with_identities
from noisiq.backends.many_shot_runner import ManyShotRunner
from noisiq.visualization import plot_hardware_comparison

def build_ghz_circuit(n: int) -> nq.Circuit:
    """Standard GHZ prep: H on q0, cascaded CNOTs q0->q1->...->q_{n-1}."""
    c = nq.Circuit(n_qubits=n, name=f"ghz_{n}q")
    c.h(0, t=0)
    for i in range(n - 1):
        c.cnot(i, i + 1, t=i + 1)
    return c

def main():
    # Ensure output directories exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("notebooks/outputs", exist_ok=True)

    # 1. Build a 10-qubit GHZ circuit
    n_qubits = 10
    circuit = build_ghz_circuit(n_qubits)
    print(f"Created {n_qubits}-qubit GHZ circuit.")

    # 2. Load hardware profiles
    quantinuum = nq.noise.get_hardware("quantinuum_h2")
    ionq = nq.noise.get_hardware("ionq_forte")
    print(f"Loaded hardware profiles: {quantinuum.system} and {ionq.system}.")

    # 3. Fill idle slots with identities using each platform's gate times
    circuit_quantinuum = fill_idle_with_identities(circuit, quantinuum.gate_times)
    circuit_ionq = fill_idle_with_identities(circuit, ionq.gate_times)

    # 4. Generate noise models (using the newly added to_pauli_noise_model helper)
    noise_quantinuum = quantinuum.to_pauli_noise_model(circuit_quantinuum)
    noise_ionq = ionq.to_pauli_noise_model(circuit_ionq)

    # 5. Run simulations
    n_shots = 2000
    runner = ManyShotRunner()

    print(f"Running {n_shots} shots simulation on Quantinuum H2-1 noise model...")
    result_quantinuum = runner.run(circuit_quantinuum, n_shots=n_shots, noise_config=noise_quantinuum, seed=42)

    print(f"Running {n_shots} shots simulation on IonQ Forte noise model...")
    result_ionq = runner.run(circuit_ionq, n_shots=n_shots, noise_config=noise_ionq, seed=42)

    print(f"\n--- Results for {n_qubits}-qubit GHZ ---")
    print(f"Quantinuum H2-1 zero-error fraction: {result_quantinuum.zero_error_fraction:.4f}")
    print(f"IonQ Forte zero-error fraction:      {result_ionq.zero_error_fraction:.4f}")

    # 6. Generate hardware comparison plots
    print("\nGenerating hardware comparison plots...")
    fig_quantinuum = plot_hardware_comparison(
        result_quantinuum, circuit_quantinuum, quantinuum,
        noise_config=noise_quantinuum,
        heat_scale="absolute_log",
        title=f"",
    )
    fig_quantinuum.savefig("outputs/ghz_quantinuum_comparison.png", bbox_inches="tight")
    fig_quantinuum.savefig("notebooks/outputs/ghz_quantinuum_comparison.png", bbox_inches="tight")
    print("Saved: outputs/ghz_quantinuum_comparison.png and notebooks/outputs/ghz_quantinuum_comparison.png")

    fig_ionq = plot_hardware_comparison(
        result_ionq, circuit_ionq, ionq,
        noise_config=noise_ionq,
        heat_scale="absolute_log",
        title=f"",
    )
    fig_ionq.savefig("outputs/ghz_ionq_comparison.png", bbox_inches="tight")
    fig_ionq.savefig("notebooks/outputs/ghz_ionq_comparison.png", bbox_inches="tight")
    print("Saved: outputs/ghz_ionq_comparison.png and notebooks/outputs/ghz_ionq_comparison.png")

    plt.close("all")
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()

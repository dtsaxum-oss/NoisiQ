"""
Stim-based Tableau simulation for Clifford circuits with step-by-step noise tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Mapping

import numpy as np
import stim

from noisiq.ir import Circuit, Operation, Measurement, ConditionalOp
from noisiq.noise import PauliError
from noisiq.noise.correlated_errors import CorrelatedPauliError
from noisiq.noise.kraus_channels import KrausChannel, CombinedChannel
from noisiq.results import SimulationResult


class NonCliffordError(Exception):
    """Raised when a non-Clifford gate is encountered in a Clifford-only backend."""
    pass

@dataclass
class ErrorEvent:
    """Record of a single error occurrence."""
    gate_index: int
    gate_name: str
    qubit: int
    pauli: str
    time_step: int
    
    def __repr__(self) -> str:
        return (
            f"ErrorEvent(t={self.time_step}, gate={self.gate_name}, "
            f"qubit={self.qubit}, pauli={self.pauli})"
        )


@dataclass
class StepResult:
    """Results from a single step (layer) of simulation."""
    time_step: int
    operation: Any  # Operation, Measurement, or ConditionalOp
    errors: List[ErrorEvent] = field(default_factory=list)
    tableau: Optional[stim.Tableau] = None


@dataclass
class StimTableauResult:
    """Results from a full simulation run."""
    steps: List[StepResult]
    n_qubits: int
    seed: Optional[int]
    final_tableau: stim.Tableau
    shot_cbits: Optional[Dict[int, int]] = None  # cbit_index -> outcome (first shot only)
    final_pauli_frames: List[str] = field(default_factory=list)
    # One n-qubit Pauli string per shot, e.g. "IXZY".
    # Exact output-frame net error Pauli strings via ideal/noisy tableau comparison.
    # Empty for MCM circuits because branch-dependent measurements/conditionals do
    # not have a single static ideal tableau suitable for output-frame extraction.
    sampled_error_products: List[str] = field(default_factory=list)
    # Debug-only one n-qubit Pauli string per shot for MCM circuits: the product
    # of sampled Pauli events in the input/event frame, with no gate conjugation.
    # Do not use this field for stabilizer/process fidelity metrics.

    def __repr__(self) -> str:
        return (
            f"StimTableauResult(n_qubits={self.n_qubits}, "
            f"steps={len(self.steps)}, seed={self.seed})"
        )


def _extract_pauli_from_tableau(tableau: stim.Tableau, n_qubits: int) -> str:
    """Extract the net Pauli string from a tableau that represents a Pauli Clifford.

    For a Pauli P on qubit j, conjugation rules give:
      P anticommutes with X_j iff P ∈ {Y, Z}  → sign of x_output(j) is -1
      P anticommutes with Z_j iff P ∈ {X, Y}  → sign of z_output(j) is -1

    Combined sign table:
      x_neg=F, z_neg=F → I
      x_neg=F, z_neg=T → X
      x_neg=T, z_neg=T → Y
      x_neg=T, z_neg=F → Z
    """
    chars = []
    for j in range(n_qubits):
        x_neg = (tableau.x_output(j).sign == -1)
        z_neg = (tableau.z_output(j).sign == -1)
        if not x_neg and not z_neg:
            chars.append('I')
        elif not x_neg and z_neg:
            chars.append('X')
        elif x_neg and z_neg:
            chars.append('Y')
        else:
            chars.append('Z')
    return ''.join(chars)


def _pauli_from_error_events(errors: List[ErrorEvent], n_qubits: int) -> str:
    """Accumulate the XOR product of sampled Pauli errors into a Pauli string.

    Used as a fallback for MCM circuits where a static ideal tableau cannot be
    precomputed. Returns errors in the input frame (no gate conjugation).
    """
    x_bits = [False] * n_qubits
    z_bits = [False] * n_qubits
    for err in errors:
        q = err.qubit
        if err.pauli == 'X':
            x_bits[q] ^= True
        elif err.pauli == 'Y':
            x_bits[q] ^= True
            z_bits[q] ^= True
        elif err.pauli == 'Z':
            z_bits[q] ^= True
    chars = []
    for q in range(n_qubits):
        if not x_bits[q] and not z_bits[q]:
            chars.append('I')
        elif x_bits[q] and not z_bits[q]:
            chars.append('X')
        elif x_bits[q] and z_bits[q]:
            chars.append('Y')
        else:
            chars.append('Z')
    return ''.join(chars)


def _is_pauli_compatible_channel(channel: object) -> bool:
    """Return True when `channel` can be sampled as Pauli-frame noise.

    StimTableauBackend can apply stochastic Pauli channels directly.  A
    CombinedChannel is compatible only when every inner channel is itself a
    PauliError or CorrelatedPauliError (recursively).  Non-Pauli Kraus channels,
    coherent rotations, and amplitude-damping/dephasing channels must route to a
    statevector/density-matrix backend instead.
    """
    if isinstance(channel, (PauliError, CorrelatedPauliError)):
        return True
    if isinstance(channel, CombinedChannel):
        return all(_is_pauli_compatible_channel(inner) for inner in channel.channels)
    return False


def _apply_pauli_to_sim(sim: stim.TableauSimulator, pauli: str, qubit: int) -> None:
    """Apply a single-qubit Pauli to a stim.TableauSimulator."""
    if pauli == 'X':
        sim.x(qubit)
    elif pauli == 'Y':
        sim.y(qubit)
    elif pauli == 'Z':
        sim.z(qubit)
    elif pauli != 'I':
        raise ValueError(f"Invalid Pauli character {pauli!r}; expected one of IXYZ.")


def _sample_and_apply_pauli_channel(
    sim: stim.TableauSimulator,
    channel: object,
    op_qubits: tuple,
    op_idx: int,
    gate_name: str,
    rng: np.random.Generator,
) -> List[ErrorEvent]:
    """Sample a Pauli-compatible channel, apply it, and return error events.

    PauliError acts independently on every qubit targeted by the parent gate.
    CorrelatedPauliError samples one joint Pauli string and applies each
    non-identity character to the corresponding gate qubit.  CombinedChannel is
    applied sequentially, preserving its channel ordering.
    """
    events: List[ErrorEvent] = []

    if isinstance(channel, CombinedChannel):
        for inner in channel.channels:
            events.extend(
                _sample_and_apply_pauli_channel(
                    sim, inner, op_qubits, op_idx, gate_name, rng,
                )
            )
        return events

    if isinstance(channel, CorrelatedPauliError):
        if len(op_qubits) < channel.num_qubits:
            raise ValueError(
                f"CorrelatedPauliError with {channel.num_qubits} Pauli characters "
                f"cannot be applied to gate {gate_name!r} on {len(op_qubits)} qubit(s) "
                f"at op index {op_idx}."
            )
        target_qubits = tuple(op_qubits[:channel.num_qubits])
        sampled_pauli = channel.sample(rng)
        if len(sampled_pauli) != channel.num_qubits:
            raise ValueError(
                f"CorrelatedPauliError.sample() returned {sampled_pauli!r}, whose "
                f"length does not match channel.num_qubits={channel.num_qubits}."
            )
        for pauli_char, qubit in zip(sampled_pauli, target_qubits):
            if pauli_char != 'I':
                _apply_pauli_to_sim(sim, pauli_char, qubit)
                events.append(ErrorEvent(
                    gate_index=op_idx,
                    gate_name=gate_name,
                    qubit=qubit,
                    pauli=pauli_char,
                    time_step=op_idx,
                ))
        return events

    if isinstance(channel, PauliError):
        for qubit in op_qubits:
            sampled_pauli = channel.sample(rng)
            if sampled_pauli != 'I':
                _apply_pauli_to_sim(sim, sampled_pauli, qubit)
                events.append(ErrorEvent(
                    gate_index=op_idx,
                    gate_name=gate_name,
                    qubit=qubit,
                    pauli=sampled_pauli,
                    time_step=op_idx,
                ))
        return events

    raise TypeError(
        "StimTableauBackend only supports Pauli-compatible noise channels "
        f"(PauliError, CorrelatedPauliError, or Pauli-only CombinedChannel); "
        f"got {type(channel).__name__}."
    )


@dataclass
class PairedShotResult:
    """Result from a single paired ideal/noisy shot (used by run_paired_shot).

    All fields come from the same shot, so error counts, fidelity metrics,
    measurement outcomes, and post-selection cbits are perfectly aligned.
    """
    errors: List[ErrorEvent]         # sampled Pauli error events for this shot
    noisy_cbits: Dict[int, int]      # cbit_index → noisy measurement outcome
    ideal_cbits: Dict[int, int]      # cbit_index → ideal measurement outcome
    measurements: Dict[str, int]     # cbit_name  → noisy outcome (for aggregation)
    sampled_error_product: str       # input-frame XOR of all applied Paulis
    final_pauli_frame: str           # output-frame error Pauli string (Option A)
    branch_diverged: bool            # True if noisy branch diverged (Option B)
    zero_error: bool                 # True when no errors were sampled this shot


@dataclass
class PairedSimResult:
    """Per-shot fidelity result from the paired ideal/noisy simulation (Phase 4B).

    Two options for characterising MCM fidelity:

    Option A — ``final_pauli_frames``: per-shot output-frame error Pauli.
        Extracted from the shot-dependent final tableau comparison
        T_ideal_inv.then(T_noisy).  Empty string when n_qubits == 0.
        This is correct even when the ideal circuit branches differ shot to shot,
        because we maintain a per-shot ideal tableau rather than a single static one.

    Option B — ``branch_diverged``: True when the noisy classical branch diverged
        from the ideal branch on at least one *deterministic* measurement.
        "Deterministic" means peek_z returned ±1 before the measurement was taken,
        i.e. the ideal state already had a definite Z-basis outcome. A random ideal
        measurement that gets postselected onto the noisy outcome is *not* flagged as
        divergence — there is no 'correct' deterministic answer to disagree with.

    Headline default: Option B (branch_success_rate).
    Option A (stabilizer_fidelity) is available for asking "did we land in the right
    stabilizer state" independently of the classical path taken.
    """

    n_shots: int
    n_qubits: int
    final_pauli_frames: List[str]   # Option A: length n_shots
    branch_diverged: List[bool]     # Option B: length n_shots

    @property
    def branch_success_rate(self) -> float:
        """Fraction of shots in which the classical branch never diverged (Option B)."""
        if self.n_shots == 0:
            return 0.0
        return 1.0 - sum(self.branch_diverged) / self.n_shots

    @property
    def stabilizer_fidelity(self) -> float:
        """Fraction of shots with an all-identity output-frame Pauli (Option A)."""
        identity = 'I' * self.n_qubits
        if self.n_shots == 0:
            return 0.0
        return sum(1 for pf in self.final_pauli_frames if pf == identity) / self.n_shots

    def branch_success_rate_report(
        self,
        *,
        backend: str = "StimTableauBackend",
        noise_model: str = "unknown",
    ) -> "MetricReport":
        """Return branch_success_rate wrapped in a MetricReport (Option B headline).

        Uses MetricKind.BRANCH_SUCCESS_RATE: the fraction of shots in which
        the noisy classical branch (which conditional corrections fired) exactly
        matched the ideal branch on every deterministic measurement.
        """
        import math
        from ..results.metrics import MetricKind, MetricReport

        p = self.branch_success_rate
        se = math.sqrt(p * (1.0 - p) / self.n_shots) if self.n_shots > 0 else 0.0
        return MetricReport(
            name="Branch success rate",
            kind=MetricKind.BRANCH_SUCCESS_RATE,
            value=p,
            uncertainty=se,
            n_shots=self.n_shots,
            backend=backend,
            noise_model=noise_model,
            method="Paired ideal/noisy simulation (Phase 4B)",
            definition=(
                "Fraction of shots in which the noisy classical branch exactly "
                "matched the ideal branch. Flagged only when the ideal measurement "
                "outcome was deterministic and the noisy outcome differed."
            ),
            limitations=(
                "Random ideal measurements (qubit in superposition) are never "
                "flagged as divergence — the ideal sim is postselected onto the "
                "noisy outcome. Only meaningful for circuits with mid-circuit "
                "measurements and feed-forward corrections.",
            ),
        )

    def stabilizer_fidelity_report(
        self,
        *,
        backend: str = "StimTableauBackend",
        noise_model: str = "unknown",
    ) -> "MetricReport":
        """Return stabilizer_fidelity wrapped in a MetricReport (Option A).

        Uses MetricKind.STABILIZER_STATE_FIDELITY: the fraction of shots in which
        the output-frame error Pauli is all-identity (noisy final state lies in the
        same stabilizer group as the ideal final state).

        Note: this does NOT flag wrong classical branches that happen to land in the
        correct stabilizer state.  Use branch_success_rate_report() for feed-forward
        protocol fidelity.
        """
        import math
        from ..results.metrics import MetricKind, MetricReport

        p = self.stabilizer_fidelity
        se = math.sqrt(p * (1.0 - p) / self.n_shots) if self.n_shots > 0 else 0.0
        return MetricReport(
            name="Stabilizer state fidelity (MCM)",
            kind=MetricKind.STABILIZER_STATE_FIDELITY,
            value=p,
            uncertainty=se,
            n_shots=self.n_shots,
            backend=backend,
            noise_model=noise_model,
            method="Paired ideal/noisy tableau comparison (Phase 4B)",
            definition=(
                "Fraction of shots in which the noisy final stabilizer state matches "
                "the ideal final stabilizer state (output-frame Pauli = I...I)."
            ),
            limitations=(
                "A circuit that takes the wrong correction path but coincidentally "
                "lands in the correct stabilizer state still counts as a success "
                "under this metric. Use branch_success_rate_report() for protocols "
                "where the classical path correctness matters.",
            ),
        )


class StimTableauBackend:
    """
    Backend using stim.TableauSimulator for step-by-step simulation.
    """
    
    def run(
        self,
        circuit: Circuit,
        noise_model: Union[
            KrausChannel,
            PauliError,
            CorrelatedPauliError,
            CombinedChannel,
            Mapping[int, Union[KrausChannel, PauliError, CorrelatedPauliError, CombinedChannel]],
            None,
        ] = None,
        n_shots: int = 100,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run n_shots of the circuit with step-by-step noise tracking.
        Returns a SimulationResult with StimTableauResult inside its meta.
        """
        circuit.validate()
        
        # Handle different noise_model formats
        noise_config = {}
        if isinstance(noise_model, dict):
            noise_config = noise_model
        elif noise_model is not None:
            for i in range(len(circuit.operations)):
                noise_config[i] = noise_model
                
        # Type check.  This backend can handle stochastic Pauli noise, including
        # joint/multi-qubit CorrelatedPauliError.  Non-Pauli Kraus/coherent
        # channels must route to TrajectoryBackend because they cannot be stored
        # as a discrete Pauli frame.
        for k, v in noise_config.items():
            if not _is_pauli_compatible_channel(v):
                raise TypeError(
                    "StimTableauBackend only supports Pauli-compatible noise models "
                    f"(got {type(v).__name__} at op index {k}). "
                    "Use TrajectoryBackend for non-Pauli channels or convert the "
                    "noise model to a Pauli/twirled representation first."
                )
        
        rng = np.random.default_rng(seed=seed)

        # Pre-compute the ideal (no-noise) final tableau for output-frame Pauli extraction.
        # This is only possible for circuits without mid-circuit measurements. MCM circuits
        # keep input-frame sampled_error_products for debugging, but do not emit
        # final_pauli_frames because those are reserved for true output-frame Paulis.
        has_mcm = any(
            isinstance(op, (Measurement, ConditionalOp))
            for op in circuit.operations
        )
        T_ideal_inv: Optional[stim.Tableau] = None
        if not has_mcm:
            ideal_sim = stim.TableauSimulator()
            # Force all n_qubits into the simulator's tableau before applying gates.
            # stim adds qubits lazily, so circuits whose gates skip some qubits (e.g.
            # identity-only circuits) would leave the ideal tableau undersized.
            # XX = I on each qubit is a safe no-op initialiser.
            for q in range(circuit.n_qubits):
                ideal_sim.x(q)
                ideal_sim.x(q)
            for _, op in sorted(enumerate(circuit.operations),
                                key=lambda kv: (kv[1].t, kv[0])):
                self._apply_gate_to_sim(ideal_sim, op)
            T_ideal_inv = ideal_sim.current_inverse_tableau()  # already the inverse of the forward tableau

        counts = {}
        all_steps: List[StepResult] = []
        final_tableau = None
        first_shot_cbits: Optional[Dict[int, int]] = None
        # cbit_name -> per-shot outcomes (one per shot, last outcome wins when a
        # cbit is written multiple times in one shot)
        all_measurements: Dict[str, List[int]] = {}
        all_final_pauli_frames: List[str] = []
        all_sampled_error_products: List[str] = []

        for shot in range(n_shots):
            sim = stim.TableauSimulator()
            # Guarantee the tableau is always n_qubits wide, even for circuits where
            # some qubits are never explicitly targeted (e.g. identity-only circuits).
            for q in range(circuit.n_qubits):
                sim.x(q)
                sim.x(q)
            steps: List[StepResult] = []
            shot_cbits: Dict[int, int] = {}   # cbit_index -> outcome (this shot)
            last_shot_meas: Dict[str, int] = {}  # cbit_name -> last outcome (this shot)
            shot_all_errors: List[ErrorEvent] = []  # used for MCM fallback only

            for op_idx, op in sorted(enumerate(circuit.operations),
                                     key=lambda kv: (kv[1].t, kv[0])):

                # --- MCM dispatch ---
                if isinstance(op, Measurement):
                    outcome = int(sim.measure(op.qubit))
                    shot_cbits[op.cbit.index] = outcome
                    last_shot_meas[op.cbit.name] = outcome
                    if op.reset and outcome == 1:
                        sim.x(op.qubit)
                    if shot == 0:
                        steps.append(StepResult(
                            time_step=op_idx,
                            operation=op,
                            errors=[],
                            tableau=sim.current_inverse_tableau().inverse()
                        ))
                    continue

                if isinstance(op, ConditionalOp):
                    if shot_cbits.get(op.condition.index, 0) != op.value:
                        continue  # Condition not met: skip gate entirely
                    op = op.inner  # Condition met: unwrap and fall through

                # --- Standard gate path ---
                # 1. Apply the ideal gate
                self._apply_gate_to_sim(sim, op)

                # 2. Sample and apply noise
                step_errors: List[ErrorEvent] = []
                if op_idx in noise_config:
                    model = noise_config[op_idx]
                    step_errors.extend(
                        _sample_and_apply_pauli_channel(
                            sim,
                            model,
                            tuple(op.qubits),
                            op_idx,
                            op.gate.name,
                            rng,
                        )
                    )

                # 3. Accumulate errors for MCM fallback; record steps for shot 0
                if has_mcm:
                    shot_all_errors.extend(step_errors)
                if shot == 0:
                    steps.append(StepResult(
                        time_step=op_idx,
                        operation=op,
                        errors=step_errors,
                        tableau=sim.current_inverse_tableau().inverse()
                    ))

            # Capture Pauli frame BEFORE final Z-basis readout (which modifies the simulator).
            if T_ideal_inv is not None:
                T_noisy = sim.current_inverse_tableau().inverse()
                T_error = T_ideal_inv.then(T_noisy)
                all_final_pauli_frames.append(
                    _extract_pauli_from_tableau(T_error, circuit.n_qubits)
                )
            else:
                all_sampled_error_products.append(
                    _pauli_from_error_events(shot_all_errors, circuit.n_qubits)
                )

            if shot == 0:
                all_steps = steps
                final_tableau = sim.current_inverse_tableau().inverse()
                first_shot_cbits = shot_cbits if shot_cbits else None

            # Accumulate per-cbit outcomes (last outcome this shot per cbit)
            for cbit_name, outcome in last_shot_meas.items():
                if cbit_name not in all_measurements:
                    all_measurements[cbit_name] = []
                all_measurements[cbit_name].append(outcome)

            # Measure all qubits for the final bitstring tally
            measurement = []
            for q in range(circuit.n_qubits):
                measurement.append(str(int(sim.measure(q))))
            bitstring = "".join(measurement)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        stim_res = StimTableauResult(
            steps=all_steps,
            n_qubits=circuit.n_qubits,
            seed=seed,
            final_tableau=final_tableau,
            shot_cbits=first_shot_cbits,
            final_pauli_frames=all_final_pauli_frames,
            sampled_error_products=all_sampled_error_products,
        )

        return SimulationResult(
            final_state=None,
            counts=counts,
            meta={"stim_result": stim_res},
            measurements=all_measurements if all_measurements else None,
        )

    def run_single_shot(
        self,
        circuit: Circuit,
        noise_config: Optional[Dict[int, Union[PauliError, CorrelatedPauliError, CombinedChannel]]] = None,
        seed: Optional[int] = None,
    ) -> StimTableauResult:
        """
        Legacy method for single shot.
        """
        res = self.run(circuit, noise_model=noise_config, n_shots=1, seed=seed)
        return res.meta["stim_result"]

    def run_paired(
        self,
        circuit: Circuit,
        noise_config: Optional[Dict[int, Union[PauliError, CorrelatedPauliError, CombinedChannel]]] = None,
        n_shots: int = 100,
        seed: Optional[int] = None,
    ) -> PairedSimResult:
        """Run the paired ideal/noisy simulation for physically-correct MCM fidelity.

        For each shot two coupled stim.TableauSimulators advance together — one ideal
        (noiseless) and one noisy — so that at every Measurement they can be compared
        in a consistent frame.

        Measurement dispatch (§2c of the implementation plan):
          1. Measure the noisy sim to sample the physical outcome b_noisy.
          2. Peek the ideal sim (peek_z) to determine whether its outcome is deterministic.
             • Deterministic (peek_z ≠ 0): call ideal_sim.measure() — safe because the
               qubit is already in an eigenstate, no coin-flip occurs.  Record divergence
               if b_noisy ≠ b_ideal.
             • Random (peek_z == 0): call ideal_sim.postselect_z(desired_value=b_noisy)
               to couple the two evolutions.  NOT flagged as divergence — there is no
               correct deterministic answer to disagree with.
          3. Write outcomes into per-branch classical dicts.
          4. If reset=True: apply X to each sim whose *own* outcome was 1.

        ConditionalOp dispatch (§2b):
          Evaluate the condition independently from ideal_cbits vs noisy_cbits.
          Noise (if configured) is applied to the *noisy sim only* and only when the
          noisy branch actually executes the gate.

        Standard gate dispatch (§2a):
          Apply the gate to both sims; apply noise to the noisy sim only.

        After each shot:
          Option A — compare final tableaus: T_error = T_ideal_inv.then(T_noisy_fwd),
              extract the output-frame Pauli string.
          Option B — branch_diverged already tracked per measurement.

        Parameters
        ----------
        circuit      : Circuit to simulate.
        noise_config : Mapping from operation index to noise channel (Pauli-compatible).
                       Keyed by position in circuit.operations (same convention as run()).
        n_shots      : Number of independent shots.
        seed         : Top-level seed for reproducibility.

        Returns
        -------
        PairedSimResult with final_pauli_frames (Option A) and branch_diverged (Option B).
        """
        if n_shots < 1:
            raise ValueError(f"n_shots must be >= 1, got {n_shots}")

        circuit.validate()

        noise_cfg: Dict[int, Union[PauliError, CorrelatedPauliError, CombinedChannel]] = (
            noise_config if noise_config is not None else {}
        )
        for k, v in noise_cfg.items():
            if not _is_pauli_compatible_channel(v):
                raise TypeError(
                    "run_paired only supports Pauli-compatible noise channels "
                    f"(got {type(v).__name__} at op index {k})."
                )

        n_qubits = circuit.n_qubits
        rng = np.random.default_rng(seed)
        shot_seeds = rng.integers(0, 2**31, size=n_shots)

        # Sort ops once — (t, original_index) order so same-layer gates stay stable.
        sorted_ops: List[tuple] = sorted(
            enumerate(circuit.operations),
            key=lambda kv: (kv[1].t, kv[0]),
        )

        all_pauli_frames: List[str] = []
        all_diverged: List[bool] = []

        for shot_seed in shot_seeds:
            shot_rng = np.random.default_rng(int(shot_seed))
            ideal_sim = stim.TableauSimulator()
            # Seed the noisy simulator so that measure() on superposition states
            # (e.g. after H) is deterministic from shot_seed.  The ideal simulator
            # is unseeded: its measure() calls only occur on eigenstates, which are
            # deterministic regardless of the PRNG state.
            noisy_sim = stim.TableauSimulator(seed=int(shot_seed))
            # Expand both simulators to n_qubits before any gates so the tableau
            # is always the right size (stim is lazy about qubit allocation).
            for q in range(n_qubits):
                ideal_sim.x(q); ideal_sim.x(q)
                noisy_sim.x(q); noisy_sim.x(q)

            ideal_cbits: Dict[int, int] = {}   # cbit_index → outcome in ideal branch
            noisy_cbits: Dict[int, int] = {}   # cbit_index → outcome in noisy branch
            shot_diverged = False

            for op_idx, op in sorted_ops:

                # ── §2c  Measurement ─────────────────────────────────────────
                if isinstance(op, Measurement):
                    b_noisy = bool(noisy_sim.measure(op.qubit))

                    peek = ideal_sim.peek_z(op.qubit)
                    if peek == 0:
                        # Random ideal outcome — postselect onto noisy outcome.
                        # Must NOT call ideal_sim.measure() here: that would
                        # independently coin-flip and risk diverging for no physical
                        # reason.  postselect_z collapses the entangled state consistently.
                        ideal_sim.postselect_z(op.qubit, desired_value=b_noisy)
                        b_ideal = b_noisy
                        # Not flagged as divergence (no deterministic 'correct' answer).
                    else:
                        # Deterministic ideal outcome — safe to measure; no coin-flip.
                        # Must NOT call postselect_z here: if b_noisy ≠ b_ideal the
                        # desired_value would be impossible and raises ValueError.
                        b_ideal = bool(ideal_sim.measure(op.qubit))
                        if b_noisy != b_ideal:
                            shot_diverged = True

                    noisy_cbits[op.cbit.index] = int(b_noisy)
                    ideal_cbits[op.cbit.index] = int(b_ideal)

                    # Reset: each sim is reset to |0⟩ based on its *own* outcome.
                    if op.reset:
                        if b_ideal:
                            ideal_sim.x(op.qubit)
                        if b_noisy:
                            noisy_sim.x(op.qubit)
                    continue

                # ── §2b  ConditionalOp ───────────────────────────────────────
                if isinstance(op, ConditionalOp):
                    inner = op.inner
                    ideal_fires = ideal_cbits.get(op.condition.index, 0) == op.value
                    noisy_fires = noisy_cbits.get(op.condition.index, 0) == op.value

                    if ideal_fires:
                        self._apply_gate_to_sim(ideal_sim, inner)
                        # Ideal sim is always noiseless — no noise applied here.

                    if noisy_fires:
                        self._apply_gate_to_sim(noisy_sim, inner)
                        # Noise attaches to the physical gate: only applied when the
                        # noisy branch actually executes the gate.
                        if op_idx in noise_cfg:
                            _sample_and_apply_pauli_channel(
                                noisy_sim,
                                noise_cfg[op_idx],
                                tuple(inner.qubits),
                                op_idx,
                                inner.gate.name,
                                shot_rng,
                            )
                    continue

                # ── §2a  Standard Clifford gate ──────────────────────────────
                self._apply_gate_to_sim(ideal_sim, op)
                self._apply_gate_to_sim(noisy_sim, op)
                if op_idx in noise_cfg:
                    _sample_and_apply_pauli_channel(
                        noisy_sim,
                        noise_cfg[op_idx],
                        tuple(op.qubits),
                        op_idx,
                        op.gate.name,
                        shot_rng,
                    )

            # ── §3  Output-frame Pauli (Option A) ────────────────────────────
            T_ideal_inv = ideal_sim.current_inverse_tableau()
            T_noisy_fwd = noisy_sim.current_inverse_tableau().inverse()
            T_error = T_ideal_inv.then(T_noisy_fwd)
            pauli_frame = _extract_pauli_from_tableau(T_error, n_qubits)

            all_pauli_frames.append(pauli_frame)
            all_diverged.append(shot_diverged)

        return PairedSimResult(
            n_shots=n_shots,
            n_qubits=n_qubits,
            final_pauli_frames=all_pauli_frames,
            branch_diverged=all_diverged,
        )

    def run_paired_shot(
        self,
        circuit: Circuit,
        noise_config: Optional[Dict[int, Union[PauliError, CorrelatedPauliError, CombinedChannel]]] = None,
        seed: Optional[int] = None,
    ) -> PairedShotResult:
        """Run one paired ideal/noisy shot and return a PairedShotResult.

        The seed is used directly for both the Stim noisy simulator and the
        Pauli-channel numpy RNG — no child-seed derivation occurs here.
        ManyShotRunner is responsible for generating per-shot seeds and passing
        them in, so that every field in the returned result is reproducible from
        that single seed value.

        Returns a PairedShotResult whose errors, cbits, measurements, fidelity
        fields, and zero_error flag all originate from the same shot execution.
        """
        noise_cfg: Dict[int, Union[PauliError, CorrelatedPauliError, CombinedChannel]] = (
            noise_config if noise_config is not None else {}
        )
        n_qubits = circuit.n_qubits

        # cbit_index → human-readable name for measurement recording.
        cbit_index_to_name: Dict[int, str] = {
            reg.offset + i: f"{reg.name}[{i}]"
            for reg in circuit.classical_registers
            for i in range(reg.size)
        }

        sorted_ops: List[tuple] = sorted(
            enumerate(circuit.operations),
            key=lambda kv: (kv[1].t, kv[0]),
        )

        # Seed the Stim noisy simulator so that measure() on superposition
        # states is deterministic from this shot_seed.  The ideal simulator's
        # measure() calls only occur on eigenstates (deterministic regardless),
        # so its seed does not affect reproducibility.
        shot_rng = np.random.default_rng(int(seed) if seed is not None else None)
        noisy_sim = stim.TableauSimulator(seed=int(seed) if seed is not None else None)
        ideal_sim = stim.TableauSimulator()

        for q in range(n_qubits):
            ideal_sim.x(q); ideal_sim.x(q)
            noisy_sim.x(q); noisy_sim.x(q)

        ideal_cbits: Dict[int, int] = {}
        noisy_cbits: Dict[int, int] = {}
        errors: List[ErrorEvent] = []
        shot_diverged = False

        for op_idx, op in sorted_ops:

            # ── §2c  Measurement ─────────────────────────────────────────────
            if isinstance(op, Measurement):
                b_noisy = bool(noisy_sim.measure(op.qubit))
                peek = ideal_sim.peek_z(op.qubit)
                if peek == 0:
                    ideal_sim.postselect_z(op.qubit, desired_value=b_noisy)
                    b_ideal = b_noisy
                else:
                    b_ideal = bool(ideal_sim.measure(op.qubit))
                    if b_noisy != b_ideal:
                        shot_diverged = True
                noisy_cbits[op.cbit.index] = int(b_noisy)
                ideal_cbits[op.cbit.index] = int(b_ideal)
                if op.reset:
                    if b_ideal:
                        ideal_sim.x(op.qubit)
                    if b_noisy:
                        noisy_sim.x(op.qubit)
                continue

            # ── §2b  ConditionalOp ───────────────────────────────────────────
            if isinstance(op, ConditionalOp):
                inner = op.inner
                ideal_fires = ideal_cbits.get(op.condition.index, 0) == op.value
                noisy_fires = noisy_cbits.get(op.condition.index, 0) == op.value
                if ideal_fires:
                    self._apply_gate_to_sim(ideal_sim, inner)
                if noisy_fires:
                    self._apply_gate_to_sim(noisy_sim, inner)
                    if op_idx in noise_cfg:
                        errors.extend(_sample_and_apply_pauli_channel(
                            noisy_sim, noise_cfg[op_idx], tuple(inner.qubits),
                            op_idx, inner.gate.name, shot_rng,
                        ))
                continue

            # ── §2a  Standard Clifford gate ──────────────────────────────────
            self._apply_gate_to_sim(ideal_sim, op)
            self._apply_gate_to_sim(noisy_sim, op)
            if op_idx in noise_cfg:
                errors.extend(_sample_and_apply_pauli_channel(
                    noisy_sim, noise_cfg[op_idx], tuple(op.qubits),
                    op_idx, op.gate.name, shot_rng,
                ))

        # ── §3  Output-frame Pauli (Option A) ────────────────────────────────
        T_ideal_inv = ideal_sim.current_inverse_tableau()
        T_noisy_fwd = noisy_sim.current_inverse_tableau().inverse()
        T_error = T_ideal_inv.then(T_noisy_fwd)
        final_pauli_frame = _extract_pauli_from_tableau(T_error, n_qubits)

        return PairedShotResult(
            errors=errors,
            noisy_cbits=noisy_cbits,
            ideal_cbits=ideal_cbits,
            measurements={
                cbit_index_to_name[idx]: outcome
                for idx, outcome in noisy_cbits.items()
                if idx in cbit_index_to_name
            },
            sampled_error_product=_pauli_from_error_events(errors, n_qubits),
            final_pauli_frame=final_pauli_frame,
            branch_diverged=shot_diverged,
            zero_error=len(errors) == 0,
        )

    def _apply_gate_to_sim(self, sim: stim.TableauSimulator, op: Operation):
        """Map NoisiQ gates to stim instructions."""
        name = op.gate.name.upper()
        if name == 'H':
            sim.h(*op.qubits)
        elif name == 'X':
            sim.x(*op.qubits)
        elif name == 'Y':
            sim.y(*op.qubits)
        elif name == 'Z':
            sim.z(*op.qubits)
        elif name == 'S':
            sim.s(*op.qubits)
        elif name == 'S_DAG':
            sim.s_dag(*op.qubits)
        elif name in ('CNOT', 'CX'):
            sim.cnot(*op.qubits)
        elif name == 'CZ':
            sim.cz(*op.qubits)
        elif name == 'SWAP':
            sim.swap(*op.qubits)
        elif name == 'I':
            pass  # Identity does nothing in TableauSimulator
        elif name == 'IDLE':
            pass  # State evolution is identity; noise is handled by the noise model
        else:
            raise NonCliffordError(
                f"Gate {name} not supported by StimTableauBackend. "
                f"Use BackendSelector to automatically route to a universal backend."
            )

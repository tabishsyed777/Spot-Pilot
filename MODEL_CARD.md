# SpotPilot model card

**Author:** Tabish Syed. **Version:** 0.1.0. **Purpose:** reproducible reinforcement learning experiments in simulated cloud scheduling.

## Model

NumPy Double DQN with a 24→64→7 ReLU network. Actions choose up to two spot/on-demand workers or checkpoint existing jobs. Earliest-deadline dispatch is rule based. The policy sees summarized job state and current synthetic market conditions, not the future trace.

## Training

Three learner seeds (7, 17, 27), each trained for 80000 steps on the standard suite without curriculum, masking, or expert replay. Validation uses 20 fixed standard scenarios starting at seed 50000. Each checkpoint maximizes its own validation mean return. Seed 7 is the default deployment/demo checkpoint, chosen before final evaluation.

Selected steps: seed 7 at 80000, seed 17 at 70000, seed 27 at 60000. Per-run JSON files record actual configurations and library versions. Training scenario streams overlap across learner seeds. Checkpoints contain weights only and do not support exact resumption.

## Evaluation and intended use

100 paired scenarios per suite across standard, volatile markets, and a lower budget; all three learned policies and five conventional baselines are reported. The default model achieved 70.75% standard completion, slightly below spot-only at 71.08%, while spending somewhat less. Shifted-market performance varies materially across learner seeds. See artifacts/RESULTS.md for complete results, including failures to generalize.

Use for study, reproducibility, and extending the simulator. There is no production cloud execution adapter, and no measured cloud savings claim.

## Controls and limitations

An external-to-policy guard rejects unaffordable actions. An event auditor independently checks charges, worker events, deadlines, and completions. These mechanisms share the host process and assume trusted scenario/ledger inputs; they are not defenses against arbitrary malicious code.

The observation is partially observable, but the network is feed-forward. Current interruption probabilities are perfectly disclosed in the simulator. Two workers, synthetic prices, fixed work speed, no startup latency, and simplified checkpoint/storage costs limit transfer to real clouds. No guarantee of completing every job or remaining bug-free is implied by the passing tests.

Models load via NumPy with pickle disabled and shape/finite-weight validation. The MIT license and source attribution are preserved.

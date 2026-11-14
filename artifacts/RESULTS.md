# SpotPilot results

Recorded on 2026-11-14. Three training runs of 80000 steps each (240000 total), selected only using standard-suite validation seeds 50000–50019. Default model: seed 7, designated before evaluation.

Evaluation: 100 scenario seeds per policy per suite, eight policies, three suites: **2400 audited episodes and 28800 job outcomes**. Every audit passed and no episode exceeded its budget. Audit success includes schedules that missed deadlines; it is an accounting check, not a performance score.

## Standard

| Policy | Completion rate | Mean credits | Mean return | Lost work units |
|---|---:|---:|---:|---:|
| Idle | 0.00% | 0.00 | -24.000 | 0.00 |
| Spot only | 71.08% | 155.10 | 22.104 | 24.68 |
| On-demand only | 49.75% | 215.82 | 7.100 | 5.05 |
| Deadline heuristic | 65.75% | 213.60 | 16.808 | 28.25 |
| Checkpoint heuristic | 64.33% | 213.69 | 16.207 | 22.99 |
| Double DQN, seed 7 | 70.75% | 151.45 | 22.075 | 22.95 |
| Double DQN, seed 17 | 70.92% | 155.73 | 22.025 | 24.00 |
| Double DQN, seed 27 | 71.33% | 149.40 | 22.584 | 22.35 |

## Volatile

| Policy | Completion rate | Mean credits | Mean return | Lost work units |
|---|---:|---:|---:|---:|
| Idle | 0.00% | 0.00 | -24.000 | 0.00 |
| Spot only | 36.83% | 203.99 | -4.109 | 31.86 |
| On-demand only | 50.92% | 215.82 | 7.954 | 4.87 |
| Deadline heuristic | 41.50% | 217.44 | -0.529 | 25.75 |
| Checkpoint heuristic | 36.50% | 217.54 | -3.594 | 19.04 |
| Double DQN, seed 7 | 46.25% | 210.30 | 3.483 | 20.14 |
| Double DQN, seed 17 | 37.17% | 192.25 | -2.985 | 23.75 |
| Double DQN, seed 27 | 32.58% | 156.74 | -5.532 | 23.22 |

## Tight Budget

| Policy | Completion rate | Mean credits | Mean return | Lost work units |
|---|---:|---:|---:|---:|
| Idle | 0.00% | 0.00 | -24.000 | 0.00 |
| Spot only | 67.42% | 138.99 | 19.991 | 22.12 |
| On-demand only | 32.17% | 149.52 | -4.256 | 5.32 |
| Deadline heuristic | 63.17% | 147.32 | 16.567 | 24.58 |
| Checkpoint heuristic | 60.42% | 147.96 | 15.021 | 19.00 |
| Double DQN, seed 7 | 66.00% | 133.28 | 19.313 | 19.27 |
| Double DQN, seed 17 | 66.92% | 139.54 | 19.716 | 20.92 |
| Double DQN, seed 27 | 67.08% | 132.97 | 20.069 | 19.64 |

## Interpretation

The default seed-7 policy completed 70.75% of standard-suite jobs, compared with 71.08% for spot-only. Its mean spend was 151.45 versus 155.10 credits. The learned policy trades slightly lower completion for lower spend here; it does not dominate the spot-only baseline.

Across the three learned policies, standard completion ranged from 70.75% to 71.33%. Volatile-suite completion ranged much more widely, from 32.58% to 46.25%; the on-demand-only baseline achieved 50.92%. Under the tight budget, learned completion ranged from 66.00% to 67.08%, compared with 67.42% for spot-only. These results show incomplete generalization across market conditions, not successful universal adaptation.

Three learner seeds provide a limited view of variability. The policies share training scenario distributions and evaluation scenarios. No significance or production savings claim is made. The transparent heuristics are not tuned industry baselines, and the workload omits many real cloud costs. No models were retrained after viewing these final benchmark results.

## Checkpoint selection

| Learner seed | Selected step | Final step |
|---|---:|---:|
| 7 | 80000 | 80000 |
| 17 | 70000 | 80000 |
| 27 | 60000 | 80000 |

## Verification performed

- 23 pytest cases passed, including the Gymnasium contract, hard-budget rejection, deadline equality, checkpoint survival, interrupted checkpoint failure, resize loss, malformed model rejection, tampered evidence detection, randomized schedules, and real subprocess recovery.
- All 2400 final benchmark episodes passed the independent event auditor.
- Standard and volatile trained-model demos completed with successful audits.
- The README Gymnasium example ran successfully.
- The worker demo resumed from checkpoint 15 after termination at chunk 17, recomputed two chunks, and matched the uninterrupted 40-chunk reference.
- The same recovery test passed in a non-root, network-disabled, read-only Docker container with writable temporary checkpoint storage.
- Cloud provisioning, live-provider billing, and real spot interruptions were not tested or implemented.

## Reproduction

Run the benchmark command in the README. Training configuration and library versions are preserved per seed. Charts can be regenerated with `pip install matplotlib` followed by `python scripts/report.py`. Real worker timings are environment-dependent; compare the success flag and final digest, not elapsed time.

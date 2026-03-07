# Baseline Evaluation for M32

**Script:** `scripts/evaluate_agent.py --baselines-only --episodes 200 --seed 99`  
**Test set:** `data/test_set_200.pkl` (seed=99, 200 episodes)  
**Date:** 2026-03-07

---

## Results

| Policy | Accuracy | Mean Return | Std Return | Mean Steps |
|--------|----------|-------------|------------|------------|
| Fixed-K(3) | 45.0% | -0.40 | 0.99 | 4.0 |
| Fixed-K(5) | 51.5% | -0.47 | 1.00 | 6.0 |
| Heuristic(0.8) | 46.0% | -0.33 | 1.04 | 3.5 |

---

## Notes

- **Best accuracy:** Fixed-K(5) at 51.5% -- retrieving more chunks before deciding helps.
- **Best return:** Heuristic(0.8) at −0.33 -- its adaptive stop trades slightly fewer chunks for comparable accuracy, saving step cost.
- **Fixed-K(3)** is the weakest on accuracy (45.0%) but uses the fewest steps (4.0), penalising return less per episode.
- All three baselines sit below 55 % accuracy, leaving substantial room for the offline RL agents (CQL / IQL) to improve.

# Dataset Statistics for M31

Generated from:
- `data/offline_buffer_2k.pkl` (train, seed=42, 2000 episodes)
- `data/test_set_200.pkl` (test, seed=99, 200 episodes)

Behavior policies (5 total, equal share of episodes):
- FixedK(k=3)
- FixedK(k=5)
- Heuristic(confidence_threshold=0.8)
- EpsilonGreedy(eps=0.3, base=FixedK-2, stop_prob=0.3)
- EpsilonGreedy(eps=0.3, base=FixedK-4, stop_prob=0.3)

---
## 1. Total Transitions in Buffer

| Dataset | Episodes | Transitions |
|---------|----------|-------------|
| Train (`offline_buffer_2k.pkl`) | 2,000 | 8,352 |
| Test  (`test_set_200.pkl`)      | 200  | 857  |

## 2. Episodes per Procedure

| Procedure | Train episodes | Test episodes |
|-----------|---------------|--------------|
| `45378` | 201 | 23 |
| `70450` | 195 | 27 |
| `70486` | 199 | 19 |
| `70553` | 208 | 28 |
| `71260` | 205 | 20 |
| `72148` | 178 | 20 |
| `74177` | 193 | 12 |
| `77067` | 191 | 18 |
| `92507` | 217 | 13 |
| `92550` | 213 | 20 |

## 3. Mean +/- Std Episode Length per Behavior Policy

| Policy | Mean Length | Std Length | N episodes (train+test) |
|--------|-------------|------------|------------------------|
| `EpsilonGreedy(eps=0.3,base=FixedK-2)` | 2.94 | 0.92 | 440 |
| `EpsilonGreedy(eps=0.3,base=FixedK-4)` | 4.53 | 1.46 | 440 |
| `FixedK(k=3)` | 4.00 | 0.00 | 440 |
| `FixedK(k=5)` | 6.00 | 0.00 | 440 |
| `Heuristic(thresh=0.8)` | 3.46 | 0.98 | 440 |

## 4. Mean Return per Behavior Policy

| Policy | Mean Return | Std Return |
|--------|-------------|------------|
| `EpsilonGreedy(eps=0.3,base=FixedK-2)` | -0.4257 | 0.9486 |
| `EpsilonGreedy(eps=0.3,base=FixedK-4)` | -0.3666 | 0.9650 |
| `FixedK(k=3)` | -0.2364 | 0.9980 |
| `FixedK(k=5)` | -0.3045 | 0.9807 |
| `Heuristic(thresh=0.8)` | -0.3052 | 1.0319 |

## 5. Decision Balance

| Decision | Train count | Train % | Test count | Test % | Combined % |
|----------|-------------|---------|------------|--------|------------|
| approve | 765 | 38.2% | 63 | 31.5% | 37.6% |
| deny | 234 | 11.7% | 28 | 14.0% | 11.9% |
| pend | 1001 | 50.0% | 109 | 54.5% | 50.5% |

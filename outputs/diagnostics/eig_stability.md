# EIG selection stability

Gold ranking uses one large-budget EIG pass per (domain, user) over a frozen mid-session posterior. Each smaller budget is repeated with fresh RNGs.

## game_a
3 users. Gold pool spread (max-median)/max per user: [0.980, 0.962, 0.935].

| n_samples | P(argmax = gold) | P(argmax in gold top-5) | mean Kendall tau |
|---|---|---|---|
| 100 | 0.07 | 0.47 | 0.754 |
| 200 | 0.03 | 0.40 | 0.803 |
| 500 | 0.12 | 0.37 | 0.851 |
| 800 | 0.18 | 0.63 | 0.869 |
| 1600 | 0.23 | 0.52 | 0.886 |

- user 0: pool=51, gold argmax targets `gamma`, gold top-5 EIG = [0.0120, 0.0120, 0.0119, 0.0111, 0.0108], spread = 0.980
- user 1: pool=51, gold argmax targets `gamma`, gold top-5 EIG = [0.0131, 0.0123, 0.0123, 0.0122, 0.0119], spread = 0.962
- user 2: pool=51, gold argmax targets `gamma`, gold top-5 EIG = [0.0130, 0.0117, 0.0115, 0.0112, 0.0112], spread = 0.935

## stock
3 users. Gold pool spread (max-median)/max per user: [1.000, 1.000, 1.000].

| n_samples | P(argmax = gold) | P(argmax in gold top-5) | mean Kendall tau |
|---|---|---|---|
| 100 | 0.08 | 0.68 | 0.545 |
| 200 | 0.08 | 0.72 | 0.570 |
| 500 | 0.20 | 0.73 | 0.576 |
| 800 | 0.08 | 0.73 | 0.566 |
| 1600 | 0.17 | 0.63 | 0.580 |

- user 0: pool=51, gold argmax targets `gamma`, gold top-5 EIG = [0.0387, 0.0387, 0.0381, 0.0380, 0.0377], spread = 1.000
- user 1: pool=51, gold argmax targets `gamma`, gold top-5 EIG = [0.0169, 0.0165, 0.0164, 0.0158, 0.0157], spread = 1.000
- user 2: pool=51, gold argmax targets `gamma`, gold top-5 EIG = [0.0252, 0.0240, 0.0240, 0.0235, 0.0234], spread = 1.000

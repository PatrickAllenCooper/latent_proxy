# Identifiability diagnostics

Per-choice Fisher information I_j = (dp/dtheta_j)^2 / (p(1-p)) computed with common random numbers over (scenario, theta) pairs.

## game_a
51 scenarios x 200 thetas (10200 pairs). Median |EU_A-EU_B|/tau = 0.007; saturated (p<0.02 or p>0.98): 0.328.

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 1.293e-08 | 2.446e-121 | 5.418e-01 | 0.413 |
| alpha | 7.323e-11 | 1.020e-135 | 1.653e-02 | 0.313 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

### By target_param subgroup

**gamma-targeted** (3400 pairs, median |dEU|/tau = 68.774, saturated = 0.970)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 7.184e-43 | 0.000e+00 | 1.101e-02 | 0.222 |
| alpha | 1.332e-43 | 0.000e+00 | 2.350e-03 | 0.209 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**alpha-targeted** (3400 pairs, median |dEU|/tau = 0.000, saturated = 0.014)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 3.429e-05 | 4.872e-16 | 1.328e+01 | 0.601 |
| alpha | 1.716e-07 | 4.837e-19 | 2.992e-01 | 0.460 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**lambda_-targeted** (3400 pairs, median |dEU|/tau = 0.000, saturated = 0.001)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 4.879e-08 | 1.982e-19 | 6.525e-02 | 0.416 |
| alpha | 2.508e-10 | 1.446e-22 | 1.282e-03 | 0.271 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**Loss-branch probe**: max fraction of MC wealth draws below the reference point (0.0) across 153 probes = 0.00000 (any draw below ref: False).
**Lambda identifiable anywhere** (I_lambda > 1e-6): False (max I_lambda = 0.000e+00).

## game_b
51 scenarios x 200 thetas (10200 pairs). Median |EU_A-EU_B|/tau = 0.000; saturated (p<0.02 or p>0.98): 0.332.

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 2.774e-25 | 0.000e+00 | 1.237e-04 | 0.144 |
| alpha | 4.539e-28 | 0.000e+00 | 8.980e-07 | 0.099 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

### By target_param subgroup

**gamma-targeted** (3400 pairs, median |dEU|/tau = 215.582, saturated = 0.989)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 5.649e-167 | 0.000e+00 | 1.306e-08 | 0.086 |
| alpha | 2.351e-167 | 0.000e+00 | 4.968e-09 | 0.078 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**alpha-targeted** (3400 pairs, median |dEU|/tau = 0.000, saturated = 0.006)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 2.501e-16 | 0.000e+00 | 1.837e-02 | 0.222 |
| alpha | 3.279e-19 | 0.000e+00 | 8.321e-05 | 0.151 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**lambda_-targeted** (3400 pairs, median |dEU|/tau = 0.000, saturated = 0.001)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 6.413e-21 | 0.000e+00 | 7.717e-06 | 0.123 |
| alpha | 6.265e-24 | 0.000e+00 | 2.635e-08 | 0.069 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**Loss-branch probe**: max fraction of MC wealth draws below the reference point (0.0) across 153 probes = 0.00000 (any draw below ref: False).
**Lambda identifiable anywhere** (I_lambda > 1e-6): False (max I_lambda = 0.000e+00).

## stock
51 scenarios x 200 thetas (10200 pairs). Median |EU_A-EU_B|/tau = 0.000; saturated (p<0.02 or p>0.98): 0.092.

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 0.000e+00 | 0.000e+00 | 2.614e+00 | 0.316 |
| alpha | 0.000e+00 | 0.000e+00 | 2.642e+00 | 0.312 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

### By target_param subgroup

**gamma-targeted** (3400 pairs, median |dEU|/tau = 1.324, saturated = 0.276)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 5.973e-01 | 2.640e-04 | 1.033e+01 | 0.941 |
| alpha | 5.497e-01 | 7.443e-05 | 6.665e+00 | 0.935 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**alpha-targeted** (3400 pairs, median |dEU|/tau = 0.000, saturated = 0.000)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.006 |
| alpha | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**lambda_-targeted** (3400 pairs, median |dEU|/tau = 0.000, saturated = 0.000)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.002 |
| alpha | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**Loss-branch probe**: max fraction of MC wealth draws below the reference point (0.0) across 153 probes = 0.00000 (any draw below ref: False).
**Lambda identifiable anywhere** (I_lambda > 1e-6): False (max I_lambda = 0.000e+00).

## supply_chain
51 scenarios x 200 thetas (10200 pairs). Median |EU_A-EU_B|/tau = 0.000; saturated (p<0.02 or p>0.98): 0.338.

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 2.032e-66 | 0.000e+00 | 3.832e-04 | 0.147 |
| alpha | 1.398e-135 | 0.000e+00 | 3.224e-06 | 0.109 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

### By target_param subgroup

**gamma-targeted** (3400 pairs, median |dEU|/tau = 895.335, saturated = 0.985)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 0.000e+00 | 0.000e+00 | 6.247e-27 | 0.058 |
| alpha | 0.000e+00 | 0.000e+00 | 3.106e-27 | 0.055 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**alpha-targeted** (3400 pairs, median |dEU|/tau = 0.000, saturated = 0.018)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 4.611e-19 | 0.000e+00 | 2.694e-02 | 0.207 |
| alpha | 1.088e-21 | 0.000e+00 | 1.727e-04 | 0.151 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**lambda_-targeted** (3400 pairs, median |dEU|/tau = 0.000, saturated = 0.010)

| param | median I | p10 I | p90 I | frac I > 1e-6 |
|---|---|---|---|---|
| gamma | 3.744e-21 | 0.000e+00 | 2.149e-03 | 0.175 |
| alpha | 5.742e-24 | 0.000e+00 | 9.577e-06 | 0.121 |
| lambda_ | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000 |

**Loss-branch probe**: max fraction of MC wealth draws below the reference point (0.0) across 153 probes = 0.00000 (any draw below ref: False).
**Lambda identifiable anywhere** (I_lambda > 1e-6): False (max I_lambda = 6.769e-27).

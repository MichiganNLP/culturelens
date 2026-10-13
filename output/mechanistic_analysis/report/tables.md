# Mechanistic study: tables

Primary judge: gemma-4-31b. Second judge: qwen-3.6-27b. Intervals are 95% bootstrap intervals.

## 1. Behaviour: share of open-ended answers judged aligned (primary / second judge)

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| No country, no norm | 43.5% / 43.9% | 41.4% / 39.4% | 51.9% / 51.8% |
| Country only | 57.9% / 58.6% | 64.1% / 62.3% | 75.2% / 73.8% |
| True norm, no country | 76.4% / 75.5% | 78.7% / 76.1% | 82.9% / 81.9% |
| Country + true norm | 93.1% / 92.0% | 94.0% / 92.1% | 95.9% / 95.0% |
| Country + false norm (resisting) | 17.9% / 17.4% | 17.1% / 15.3% | 44.0% / 41.8% |
| Resisting, among records right with the country alone | 26.1% / 25.5% | 23.5% / 21.6% | 52.2% / 50.1% |
| Resisting when the answer without a country is wrong and with the country alone is wrong | 5.6% (n=3347) | 5.0% (n=2776) | 16.6% (n=1893) |
| Resisting when the answer without a country is wrong and with the country alone is right | 19.2% (n=1883) | 16.8% (n=2655) | 42.6% (n=2559) |
| Resisting when the answer without a country is right and with the country alone is wrong | 12.1% (n=552) | 10.2% (n=551) | 31.5% (n=403) |
| Resisting when the answer without a country is right and with the country alone is right | 29.9% (n=3479) | 28.9% (n=3279) | 57.8% (n=4406) |
| Judges agree on resisting (agreement, κ) | 96.2%, 0.87 | 95.0%, 0.82 | 96.5%, 0.93 |

## 2. Readout: AUROC for resisting, labels from gemma-4-31b

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Words of the prompt (no model) | 0.652 [0.636, 0.667] | 0.683 [0.669, 0.698] | 0.683 [0.672, 0.694] |
| State: no country, no claim | 0.623 [0.608, 0.639] | 0.666 [0.650, 0.680] | 0.691 [0.680, 0.701] |
| State: country only | 0.622 [0.607, 0.638] | 0.680 [0.665, 0.694] | 0.710 [0.699, 0.720] |
| State: false norm, no country | 0.747 [0.733, 0.761] | 0.790 [0.777, 0.802] | 0.844 [0.836, 0.851] |
| State: country + false norm | 0.791 [0.779, 0.804] | 0.841 [0.829, 0.851] | 0.917 [0.911, 0.922] |
| Gain from the country (country + false norm minus false norm alone) | 0.045 [0.034, 0.054] | 0.051 [0.041, 0.060] | 0.073 [0.067, 0.079] |
| Country + false norm state, among records right with the country alone | 0.779 [0.765, 0.794] | 0.831 [0.819, 0.843] | 0.915 [0.909, 0.921] |
| Layer | 16 | 26 | 36 |

## 2. Readout: AUROC for resisting, labels from qwen-3.6-27b

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Words of the prompt (no model) | 0.654 [0.638, 0.669] | 0.683 [0.667, 0.699] | 0.681 [0.670, 0.691] |
| State: no country, no claim | 0.634 [0.618, 0.649] | 0.666 [0.649, 0.681] | 0.694 [0.683, 0.704] |
| State: country only | 0.630 [0.615, 0.646] | 0.682 [0.666, 0.697] | 0.715 [0.704, 0.725] |
| State: false norm, no country | 0.756 [0.743, 0.769] | 0.796 [0.784, 0.808] | 0.839 [0.831, 0.847] |
| State: country + false norm | 0.798 [0.786, 0.811] | 0.837 [0.824, 0.848] | 0.910 [0.905, 0.916] |
| Gain from the country (country + false norm minus false norm alone) | 0.043 [0.033, 0.052] | 0.041 [0.032, 0.051] | 0.072 [0.065, 0.078] |
| Country + false norm state, among records right with the country alone | 0.781 [0.766, 0.796] | 0.828 [0.815, 0.841] | 0.908 [0.900, 0.915] |
| Layer | 16 | 26 | 36 |

## 2. Readout: AUROC for resisting, labels from both

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Words of the prompt (no model) | 0.651 [0.635, 0.667] | 0.687 [0.670, 0.704] | 0.671 [0.660, 0.682] |
| State: no country, no claim | 0.630 [0.613, 0.646] | 0.669 [0.652, 0.686] | 0.695 [0.684, 0.705] |
| State: country only | 0.627 [0.612, 0.643] | 0.687 [0.671, 0.703] | 0.716 [0.705, 0.726] |
| State: false norm, no country | 0.757 [0.742, 0.771] | 0.804 [0.791, 0.817] | 0.841 [0.833, 0.849] |
| State: country + false norm | 0.795 [0.783, 0.808] | 0.848 [0.835, 0.859] | 0.912 [0.906, 0.917] |
| Gain from the country (country + false norm minus false norm alone) | 0.039 [0.028, 0.049] | 0.044 [0.033, 0.053] | 0.071 [0.065, 0.077] |
| Country + false norm state, among records right with the country alone | 0.776 [0.760, 0.791] | 0.834 [0.820, 0.847] | 0.908 [0.900, 0.915] |
| Layer | 16 | 26 | 36 |

## 3. Country × claim interaction

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| True- vs false-claim interaction, cosine: early layers → report layer | 0.99 → 0.67 | 0.96 → 0.71 | 0.97 → 0.71 |
| Share of the country + false-norm shift that is additive | 44.4% | 63.3% | 62.9% |
| Size of the country effect relative to the false-claim effect | 0.90 | 0.74 | 0.74 |
| Conflict feature → resisting, AUROC (gemma-4-31b) | 0.575 [0.560, 0.590] | 0.575 [0.558, 0.591] | 0.731 [0.720, 0.741] |
| Conflict feature → resisting, AUROC (qwen-3.6-27b) | 0.579 [0.565, 0.595] | 0.578 [0.561, 0.594] | 0.723 [0.712, 0.733] |
| Conflict feature → resisting, AUROC (both) | 0.580 [0.564, 0.595] | 0.577 [0.560, 0.595] | 0.724 [0.714, 0.734] |

## 4. Where the claim is read: change in margin (nats) when the answer cannot attend to a span

**Llama-3.1-8B** (400 records; baseline margin country_opposite_norm -7.3, country_norm +14.2)

| Blocked | 0-4 | 4-8 | 8-12 | 12-16 | 16-20 | 20-24 | 24-28 | 28-32 | every layer |
|---|---|---|---|---|---|---|---|---|---|
| claim, false-norm prompt | +0.03 | +0.04 | +1.54 | +6.32 | +0.91 | +0.63 | +0.37 | +0.29 | +12.46 |
| country, false-norm prompt | +0.07 | +0.02 | -0.00 | -0.69 | -0.19 | -0.01 | -0.00 | +0.00 | -0.84 |
| control, false-norm prompt | +0.04 | +0.00 | +0.04 | -0.19 | -0.06 | -0.01 | -0.01 | -0.01 | -0.02 |
| claim, true-norm prompt | +0.01 | -0.01 | -0.70 | -4.37 | -0.87 | -0.40 | -0.27 | -0.16 | -9.04 |
| country, true-norm prompt | -0.04 | -0.07 | -0.10 | +0.29 | +0.22 | +0.04 | +0.01 | -0.01 | +0.13 |

**Gemma-3-12B** (400 records; baseline margin country_opposite_norm -14.8, country_norm +23.5)

| Blocked | 0-6 | 6-12 | 12-18 | 18-24 | 24-30 | 30-36 | 36-42 | 42-48 | every layer |
|---|---|---|---|---|---|---|---|---|---|
| claim, false-norm prompt | +0.44 | +0.77 | +2.94 | +10.25 | +3.35 | +0.46 | +0.60 | +0.14 | +22.36 |
| country, false-norm prompt | +0.08 | +0.12 | +1.11 | -0.84 | -0.29 | -0.01 | -0.03 | -0.02 | +0.24 |
| control, false-norm prompt | +0.00 | +0.04 | +0.05 | -0.18 | +0.04 | +0.03 | -0.01 | -0.04 | +0.41 |
| claim, true-norm prompt | -0.36 | -0.62 | -2.27 | -5.30 | -3.40 | -0.40 | -0.71 | -0.15 | -16.03 |
| country, true-norm prompt | -0.10 | -0.17 | -0.68 | +0.73 | +0.35 | -0.00 | +0.01 | +0.08 | +0.09 |

## 5. Answers under knockout: share judged aligned (gemma-4-31b)

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Country + false norm | 18.0% [14.5%, 21.8%] (n=400) | 14.2% [11.0%, 17.8%] (n=400) | 47.0% [40.0%, 54.0%] (n=200) |
| … claim blocked, every layer | 58.0% [53.2%, 63.0%] (n=400) | 66.5% [61.8%, 71.0%] (n=400) | – |
| … claim blocked, claim window | 47.0% [42.2%, 51.7%] (n=400) | 59.0% [54.0%, 63.5%] (n=400) | – |
| … claim blocked, later window | 19.5% [15.8%, 23.2%] (n=400) | 16.5% [13.0%, 20.2%] (n=400) | – |
| … country line blocked | 16.2% [12.8%, 20.0%] (n=400) | 10.0% [7.2%, 13.0%] (n=400) | – |
| … situation text blocked (control) | 13.8% [10.5%, 17.2%] (n=400) | 15.0% [11.8%, 18.5%] (n=400) | – |
| … claim blocked, selected heads only | 46.5% [41.5%, 51.2%] (n=400) | 23.5% [19.5%, 28.0%] (n=400) | – |
| … claim blocked, as many random heads | 20.5% [16.8%, 24.5%] (n=400) | 13.0% [9.8%, 16.5%] (n=400) | – |
| … claim erased after the first block | 56.2% [51.5%, 61.0%] (n=400) | 64.8% [59.8%, 69.2%] (n=400) | 78.0% [72.5%, 83.5%] (n=200) |
| … claim erased before the reading band | 61.0% [56.2%, 65.8%] (n=400) | 68.2% [63.7%, 72.8%] (n=400) | 76.5% [70.5%, 82.0%] (n=200) |
| … claim erased after the reading band | 26.2% [22.2%, 30.8%] (n=400) | 21.8% [17.8%, 26.0%] (n=400) | 54.5% [47.5%, 61.5%] (n=200) |
| False norm, no country in prompt | 28.2% [23.8%, 33.0%] (n=400) | 22.2% [18.2%, 26.2%] (n=400) | 50.0% [43.5%, 57.0%] (n=200) |
| Country only | 58.8% [54.0%, 63.5%] (n=400) | 68.0% [63.5%, 72.5%] (n=400) | – |
| Country + true norm | 93.8% [91.2%, 96.0%] (n=400) | 95.5% [93.2%, 97.5%] (n=400) | – |
| … claim blocked, every layer | 58.0% [53.2%, 62.7%] (n=400) | 67.5% [63.0%, 72.0%] (n=400) | – |

Paired differences in percentage points (gemma-4-31b):

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Effect of blocking the false claim, every layer | +40.0 [+35.0, +45.3], p=0.0005 | +52.2 [+47.0, +57.5], p=0.0005 | – |
| Effect of blocking it in the claim window only | +29.0 [+24.2, +34.0], p=0.0005 | +44.8 [+39.8, +50.0], p=0.0005 | – |
| Effect of blocking it in a later window (control) | +1.5 [-1.0, +4.0], p=0.321 | +2.2 [+0.0, +4.5], p=0.0885 | – |
| Effect of blocking it for the selected heads only | +28.5 [+23.8, +33.2], p=0.0005 | +9.2 [+5.2, +13.0], p=0.0005 | – |
| Effect of blocking it for as many random heads (control) | +2.5 [-0.5, +5.5], p=0.134 | -1.2 [-4.0, +1.2], p=0.451 | – |
| Effect of erasing the claim after the first block | +38.2 [+33.0, +43.8], p=0.0005 | +50.5 [+45.2, +55.8], p=0.0005 | +31.0 [+24.0, +38.5], p=0.0005 |
| Effect of erasing it before the reading band | +43.0 [+37.8, +48.5], p=0.0005 | +54.0 [+48.8, +59.0], p=0.0005 | +29.5 [+22.0, +37.0], p=0.0005 |
| Effect of erasing it after the reading band (control) | +8.2 [+4.5, +12.0], p=0.0005 | +7.5 [+4.0, +11.2], p=0.0005 | +7.5 [+1.5, +13.5], p=0.0305 |
| Effect of blocking situation text (control) | -4.2 [-7.2, -1.2], p=0.0135 | +0.8 [-2.2, +3.8], p=0.734 | – |
| Effect of blocking the country line | -1.8 [-5.2, +1.8], p=0.376 | -4.2 [-7.5, -1.0], p=0.018 | – |
| Effect of naming the country, false norm stated | -10.2 [-14.8, -5.8], p=0.0005 | -8.0 [-12.8, -3.2], p=0.0015 | -3.0 [-11.0, +4.5], p=0.53 |
| Effect of blocking the true claim, every layer | -35.8 [-40.8, -30.8], p=0.0005 | -28.0 [-33.0, -23.2], p=0.0005 | – |
| Effect of claim blocked vs never stated | -0.8 [-4.2, +2.8], p=0.788 | -1.5 [-5.5, +2.8], p=0.564 | – |

By what the model answers with the country alone (gemma-4-31b):

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Records right with the country alone: Country + false norm | 25.5% [20.0%, 31.1%] (n=235) | 17.6% [13.2%, 22.1%] (n=272) | – |
| Records right with the country alone: claim blocked, claim window | 66.0% [59.6%, 71.9%] (n=235) | 73.2% [67.6%, 78.3%] (n=272) | – |
| Records right with the country alone: claim blocked, every layer | 88.9% [84.7%, 92.8%] (n=235) | 85.7% [81.2%, 89.7%] (n=272) | – |
| Records right with the country alone: False norm, no country in prompt | 37.4% [31.1%, 43.4%] (n=235) | 25.0% [19.9%, 30.5%] (n=272) | – |
| Records wrong with the country alone: Country + false norm | 7.3% [3.6%, 11.5%] (n=165) | 7.0% [3.1%, 11.7%] (n=128) | – |
| Records wrong with the country alone: claim blocked, claim window | 20.0% [13.9%, 26.7%] (n=165) | 28.9% [21.9%, 36.7%] (n=128) | – |
| Records wrong with the country alone: claim blocked, every layer | 13.9% [9.1%, 19.4%] (n=165) | 25.8% [18.8%, 33.6%] (n=128) | – |
| Records wrong with the country alone: False norm, no country in prompt | 15.2% [9.7%, 20.6%] (n=165) | 16.4% [10.2%, 23.4%] (n=128) | – |

## 5. Answers under knockout: share judged aligned (qwen-3.6-27b)

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Country + false norm | 17.8% [14.2%, 21.5%] (n=400) | 13.0% [10.0%, 16.2%] (n=400) | 45.5% [38.5%, 52.5%] (n=200) |
| … claim blocked, every layer | 62.0% [57.5%, 66.8%] (n=400) | 65.2% [60.5%, 69.8%] (n=400) | – |
| … claim blocked, claim window | 50.0% [45.2%, 54.8%] (n=400) | 57.2% [52.2%, 62.3%] (n=400) | – |
| … claim blocked, later window | 18.8% [15.0%, 22.5%] (n=400) | 13.2% [10.2%, 16.8%] (n=400) | – |
| … country line blocked | 16.0% [12.5%, 19.8%] (n=400) | 11.5% [8.5%, 14.8%] (n=400) | – |
| … situation text blocked (control) | 13.2% [10.2%, 16.5%] (n=400) | 11.5% [8.5%, 14.8%] (n=400) | – |
| … claim blocked, selected heads only | 46.0% [41.0%, 50.7%] (n=400) | 22.5% [18.2%, 26.5%] (n=400) | – |
| … claim blocked, as many random heads | 20.2% [16.5%, 24.0%] (n=400) | 13.0% [10.0%, 16.5%] (n=400) | – |
| … claim erased after the first block | 57.5% [52.8%, 62.3%] (n=400) | 62.5% [57.5%, 67.2%] (n=400) | 77.0% [71.0%, 83.0%] (n=200) |
| … claim erased before the reading band | 62.5% [58.0%, 67.2%] (n=400) | 67.0% [62.5%, 71.5%] (n=400) | 76.5% [70.5%, 82.0%] (n=200) |
| … claim erased after the reading band | 25.8% [21.5%, 30.2%] (n=400) | 19.8% [15.8%, 23.6%] (n=399) | 48.0% [41.0%, 54.5%] (n=200) |
| False norm, no country in prompt | 27.8% [23.2%, 32.2%] (n=400) | 20.0% [16.2%, 24.0%] (n=400) | 49.5% [42.5%, 56.0%] (n=200) |
| Country only | 58.8% [53.8%, 63.5%] (n=400) | 67.0% [62.3%, 71.2%] (n=400) | – |
| Country + true norm | 93.8% [91.2%, 96.0%] (n=400) | 94.8% [92.5%, 96.8%] (n=400) | – |
| … claim blocked, every layer | 59.8% [55.0%, 64.5%] (n=400) | 68.0% [63.7%, 72.5%] (n=400) | – |

Paired differences in percentage points (qwen-3.6-27b):

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Effect of blocking the false claim, every layer | +44.2 [+38.8, +49.8], p=0.0005 | +52.2 [+47.0, +57.2], p=0.0005 | – |
| Effect of blocking it in the claim window only | +32.2 [+27.0, +37.2], p=0.0005 | +44.2 [+39.2, +49.0], p=0.0005 | – |
| Effect of blocking it in a later window (control) | +1.0 [-1.8, +3.8], p=0.554 | +0.2 [-1.5, +2.2], p=1 | – |
| Effect of blocking it for the selected heads only | +28.2 [+23.5, +33.2], p=0.0005 | +9.5 [+6.0, +13.2], p=0.0005 | – |
| Effect of blocking it for as many random heads (control) | +2.5 [-0.5, +5.5], p=0.133 | +0.0 [-2.2, +2.2], p=1 | – |
| Effect of erasing the claim after the first block | +39.8 [+34.5, +45.2], p=0.0005 | +49.5 [+44.2, +55.0], p=0.0005 | +31.5 [+24.5, +39.0], p=0.0005 |
| Effect of erasing it before the reading band | +44.8 [+39.5, +49.8], p=0.0005 | +54.0 [+49.0, +59.0], p=0.0005 | +31.0 [+23.5, +39.0], p=0.0005 |
| Effect of erasing it after the reading band (control) | +8.0 [+4.0, +11.8], p=0.0005 | +6.8 [+3.0, +10.5], p=0.0005 | +2.5 [-3.0, +9.0], p=0.524 |
| Effect of blocking situation text (control) | -4.5 [-7.2, -1.8], p=0.0035 | -1.5 [-4.5, +1.5], p=0.412 | – |
| Effect of blocking the country line | -1.8 [-5.5, +2.0], p=0.428 | -1.5 [-4.5, +1.5], p=0.409 | – |
| Effect of naming the country, false norm stated | -10.0 [-14.5, -5.5], p=0.0005 | -7.0 [-11.5, -2.5], p=0.007 | -4.0 [-12.5, +3.5], p=0.387 |
| Effect of blocking the true claim, every layer | -34.0 [-39.0, -29.2], p=0.0005 | -26.8 [-31.5, -22.2], p=0.0005 | – |
| Effect of claim blocked vs never stated | +3.2 [-0.5, +7.0], p=0.112 | -1.8 [-5.5, +2.2], p=0.466 | – |

By what the model answers with the country alone (qwen-3.6-27b):

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Records right with the country alone: Country + false norm | 26.0% [20.9%, 31.5%] (n=235) | 16.0% [11.9%, 20.5%] (n=268) | – |
| Records right with the country alone: claim blocked, claim window | 68.5% [62.6%, 74.5%] (n=235) | 72.8% [67.2%, 77.6%] (n=268) | – |
| Records right with the country alone: claim blocked, every layer | 90.6% [86.8%, 94.0%] (n=235) | 86.9% [82.5%, 90.7%] (n=268) | – |
| Records right with the country alone: False norm, no country in prompt | 37.0% [30.6%, 43.0%] (n=235) | 24.6% [19.8%, 29.5%] (n=268) | – |
| Records wrong with the country alone: Country + false norm | 6.1% [2.4%, 10.3%] (n=165) | 6.8% [3.0%, 11.4%] (n=132) | – |
| Records wrong with the country alone: claim blocked, claim window | 23.6% [17.0%, 30.3%] (n=165) | 25.8% [18.9%, 33.3%] (n=132) | – |
| Records wrong with the country alone: claim blocked, every layer | 21.2% [15.2%, 27.9%] (n=165) | 21.2% [14.4%, 28.0%] (n=132) | – |
| Records wrong with the country alone: False norm, no country in prompt | 14.5% [9.7%, 20.0%] (n=165) | 10.6% [6.1%, 15.9%] (n=132) | – |

## 5. Answers under knockout: share judged aligned (both)

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Country + false norm | 16.5% [13.2%, 20.2%] (n=400) | 11.5% [8.5%, 14.8%] (n=400) | 44.0% [37.0%, 51.0%] (n=200) |
| … claim blocked, every layer | 54.8% [50.0%, 59.8%] (n=400) | 62.0% [57.2%, 66.5%] (n=400) | – |
| … claim blocked, claim window | 44.8% [40.0%, 49.5%] (n=400) | 54.8% [49.8%, 59.5%] (n=400) | – |
| … claim blocked, later window | 17.0% [13.5%, 20.8%] (n=400) | 12.8% [9.8%, 16.2%] (n=400) | – |
| … country line blocked | 14.0% [10.5%, 17.5%] (n=400) | 9.5% [6.8%, 12.5%] (n=400) | – |
| … situation text blocked (control) | 11.0% [8.0%, 14.2%] (n=400) | 10.5% [7.8%, 13.5%] (n=400) | – |
| … claim blocked, selected heads only | 42.8% [38.0%, 47.5%] (n=400) | 20.8% [16.8%, 25.0%] (n=400) | – |
| … claim blocked, as many random heads | 17.8% [14.0%, 21.5%] (n=400) | 11.0% [8.2%, 14.2%] (n=400) | – |
| … claim erased after the first block | 52.8% [48.0%, 57.8%] (n=400) | 59.8% [55.0%, 64.5%] (n=400) | 74.5% [68.5%, 80.5%] (n=200) |
| … claim erased before the reading band | 57.8% [53.2%, 62.5%] (n=400) | 64.8% [60.0%, 69.3%] (n=400) | 73.5% [67.0%, 79.0%] (n=200) |
| … claim erased after the reading band | 22.8% [18.8%, 27.0%] (n=400) | 18.3% [14.5%, 22.1%] (n=399) | 48.0% [41.0%, 54.5%] (n=200) |
| False norm, no country in prompt | 24.2% [20.0%, 28.5%] (n=400) | 18.8% [15.2%, 22.8%] (n=400) | 46.5% [40.0%, 53.5%] (n=200) |
| Country only | 55.0% [50.2%, 59.8%] (n=400) | 63.0% [58.2%, 67.5%] (n=400) | – |
| Country + true norm | 92.5% [90.0%, 95.0%] (n=400) | 94.0% [91.5%, 96.2%] (n=400) | – |
| … claim blocked, every layer | 54.8% [50.0%, 59.8%] (n=400) | 64.8% [60.2%, 69.5%] (n=400) | – |

Paired differences in percentage points (both):

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Effect of blocking the false claim, every layer | +38.2 [+32.8, +43.5], p=0.0005 | +50.5 [+45.5, +55.5], p=0.0005 | – |
| Effect of blocking it in the claim window only | +28.2 [+23.5, +33.0], p=0.0005 | +43.2 [+38.2, +48.0], p=0.0005 | – |
| Effect of blocking it in a later window (control) | +0.5 [-2.0, +3.0], p=0.838 | +1.2 [-0.5, +3.0], p=0.292 | – |
| Effect of blocking it for the selected heads only | +26.2 [+21.5, +31.0], p=0.0005 | +9.2 [+5.8, +12.8], p=0.0005 | – |
| Effect of blocking it for as many random heads (control) | +1.2 [-1.8, +4.0], p=0.484 | -0.5 [-2.8, +1.8], p=0.834 | – |
| Effect of erasing the claim after the first block | +36.2 [+31.0, +41.8], p=0.0005 | +48.2 [+43.0, +53.5], p=0.0005 | +30.5 [+23.0, +38.0], p=0.0005 |
| Effect of erasing it before the reading band | +41.2 [+36.0, +46.5], p=0.0005 | +53.2 [+48.0, +58.2], p=0.0005 | +29.5 [+21.5, +37.5], p=0.0005 |
| Effect of erasing it after the reading band (control) | +6.2 [+2.5, +9.8], p=0.001 | +6.8 [+3.3, +10.3], p=0.0015 | +4.0 [-2.0, +10.0], p=0.266 |
| Effect of blocking situation text (control) | -5.5 [-8.2, -2.8], p=0.001 | -1.0 [-4.0, +1.8], p=0.632 | – |
| Effect of blocking the country line | -2.5 [-6.0, +1.0], p=0.208 | -2.0 [-4.8, +0.8], p=0.227 | – |
| Effect of naming the country, false norm stated | -7.8 [-12.0, -3.2], p=0.002 | -7.2 [-11.8, -3.0], p=0.0035 | -2.5 [-11.0, +5.0], p=0.627 |
| Effect of blocking the true claim, every layer | -37.8 [-42.8, -33.0], p=0.0005 | -29.2 [-34.2, -24.2], p=0.0005 | – |
| Effect of claim blocked vs never stated | -0.2 [-4.0, +3.2], p=1 | -1.0 [-5.0, +3.2], p=0.722 | – |

By what the model answers with the country alone (both):

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Records right with the country alone: Country + false norm | 24.5% [19.1%, 30.0%] (n=220) | 14.7% [10.3%, 19.4%] (n=252) | – |
| Records right with the country alone: claim blocked, claim window | 65.9% [59.5%, 71.8%] (n=220) | 71.8% [65.9%, 77.0%] (n=252) | – |
| Records right with the country alone: claim blocked, every layer | 87.7% [83.2%, 91.8%] (n=220) | 85.3% [81.0%, 89.3%] (n=252) | – |
| Records right with the country alone: False norm, no country in prompt | 33.6% [27.3%, 40.0%] (n=220) | 22.2% [17.5%, 27.4%] (n=252) | – |
| Records wrong with the country alone: Country + false norm | 6.7% [3.3%, 10.6%] (n=180) | 6.1% [2.7%, 10.1%] (n=148) | – |
| Records wrong with the country alone: claim blocked, claim window | 18.9% [13.3%, 24.4%] (n=180) | 25.7% [18.9%, 33.1%] (n=148) | – |
| Records wrong with the country alone: claim blocked, every layer | 14.4% [9.4%, 19.4%] (n=180) | 22.3% [15.5%, 29.1%] (n=148) | – |
| Records wrong with the country alone: False norm, no country in prompt | 12.8% [8.3%, 17.8%] (n=180) | 12.8% [7.4%, 18.2%] (n=148) | – |

## 6. Erasing the claim's positions after a given block: change in margin (nats), false-norm prompt

**Llama-3.1-8B** (400 records)

| After block | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| False-norm prompt | +12.57 | +12.38 | +12.41 | +12.24 | +11.82 | +11.22 | +9.05 | +5.16 | +3.36 | +2.38 | +2.19 | +1.47 | +0.84 | +0.50 | +0.24 | -0.16 |
| True-norm prompt | -8.93 | -9.04 | -8.96 | -8.72 | -8.34 | -7.73 | -6.35 | -4.00 | -2.65 | -1.84 | -1.80 | -1.24 | -0.66 | -0.33 | -0.14 | -0.01 |

**Gemma-3-12B** (400 records)

| After block | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 | 44 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| False-norm prompt | +20.86 | +20.76 | +20.42 | +19.52 | +19.42 | +14.23 | +8.52 | +5.47 | +2.47 | +1.18 | +1.12 | +0.20 |
| True-norm prompt | -17.42 | -17.29 | -17.06 | -16.38 | -14.51 | -10.57 | -7.46 | -5.14 | -2.67 | -1.36 | -1.23 | -0.29 |

**Qwen-3.5-27B** (200 records)

| After block | 0 | 8 | 16 | 24 | 32 | 40 | 48 | 56 |
|---|---|---|---|---|---|---|---|---|
| False-norm prompt | +8.62 | +8.55 | +8.15 | +7.39 | +5.49 | +3.47 | +3.04 | +0.84 |
| True-norm prompt | -7.80 | -7.64 | -7.63 | -6.98 | -6.37 | -3.60 | -2.72 | -0.48 |

## 7. Heads that carry the claim (development sample, margin)

**Llama-3.1-8B**: window 12–16, 128 heads, whole-window effect +6.44 nats (200 records).

Largest single heads: L14.H24 (+0.71), L12.H4 (+0.71), L13.H21 (+0.59), L13.H23 (+0.44), L14.H5 (+0.41), L13.H4 (+0.34), L13.H22 (+0.34), L12.H22 (+0.33).

| Heads blocked together | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Top-ranked | +0.71 | +1.59 | +3.11 | +5.14 | +7.12 | +7.55 | +8.71 |
| Random | +0.29 | +0.36 | +0.59 | +1.33 | +1.96 | +1.23 | +3.38 |

Selected: 8 heads ({'14': [24, 5], '12': [4, 22], '13': [21, 23, 4, 22]}).

**Gemma-3-12B**: window 18–24, 96 heads, whole-window effect +10.91 nats (200 records).

Largest single heads: L20.H0 (+1.62), L23.H8 (+0.78), L23.H3 (+0.60), L22.H9 (+0.51), L21.H11 (+0.42), L23.H10 (+0.34), L23.H5 (+0.34), L23.H4 (+0.31).

| Heads blocked together | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Top-ranked | +1.62 | +2.47 | +4.01 | +6.21 | +9.25 | +11.76 | +15.02 |
| Random | -0.22 | -0.10 | -0.51 | -1.24 | -1.09 | +0.17 | +5.89 |

Selected: 8 heads ({'20': [0], '23': [8, 3, 10, 5, 4], '22': [9], '21': [11]}).

## 8. Full benchmark: resisting a false norm with and without the country line (gemma-4-31b)

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| False norm, country named | 17.9% [17.1%, 18.7%] | 17.1% [16.4%, 17.9%] | 44.0% [43.0%, 45.0%] |
| False norm, no country | 28.4% [27.5%, 29.4%] | 25.7% [24.8%, 26.6%] | 44.6% [43.6%, 45.6%] |
| Effect of naming the country (points) | -10.6 [-11.5, -9.6], p=0.0005 (n=9260) | -8.6 [-9.5, -7.6], p=0.0005 (n=9261) | -0.6 [-1.6, +0.5], p=0.298 (n=9261) |
| … among records right with the country alone | -14.0 [-15.3, -12.6], p=0.0005 (n=5362) | -9.3 [-10.7, -7.9], p=0.0005 (n=5934) | -0.3 [-1.6, +1.0], p=0.687 (n=6965) |
| … among records wrong with the country alone | -5.9 [-7.0, -4.8], p=0.0005 (n=3898) | -7.3 [-8.5, -6.0], p=0.0005 (n=3327) | -1.5 [-3.3, +0.3], p=0.128 (n=2296) |

## 9. Does the margin track the judged answer? AUROC of the margin for the answer being judged aligned (gemma-4-31b)

| Prompt and intervention | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Country + false norm | 0.664 [0.594, 0.732]; margin > 0 in 24.2%, aligned 18.0% | 0.648 [0.568, 0.724]; margin > 0 in 26.2%, aligned 14.2% | 0.538 [0.458, 0.618]; margin > 0 in 34.0%, aligned 47.0% |
| … claim blocked, every layer | 0.622 [0.565, 0.679]; margin > 0 in 68.8%, aligned 58.0% | 0.589 [0.527, 0.644]; margin > 0 in 64.5%, aligned 66.5% | – |
| … claim blocked, claim window | 0.689 [0.637, 0.741]; margin > 0 in 47.8%, aligned 47.0% | 0.629 [0.573, 0.681]; margin > 0 in 42.0%, aligned 59.0% | – |
| … claim blocked, later window | 0.609 [0.536, 0.680]; margin > 0 in 25.8%, aligned 19.5% | 0.647 [0.570, 0.719]; margin > 0 in 27.0%, aligned 16.5% | – |
| … claim erased after the first block | 0.623 [0.568, 0.679]; margin > 0 in 69.2%, aligned 56.2% | 0.596 [0.537, 0.651]; margin > 0 in 63.0%, aligned 64.8% | 0.652 [0.561, 0.745]; margin > 0 in 66.5%, aligned 78.0% |
| … claim erased before the reading band | 0.625 [0.571, 0.684]; margin > 0 in 69.0%, aligned 61.0% | 0.555 [0.497, 0.613]; margin > 0 in 58.0%, aligned 68.2% | 0.592 [0.501, 0.685]; margin > 0 in 63.5%, aligned 76.5% |
| … claim erased after the reading band | 0.683 [0.622, 0.744]; margin > 0 in 35.0%, aligned 26.2% | 0.625 [0.557, 0.691]; margin > 0 in 40.2%, aligned 21.8% | – |


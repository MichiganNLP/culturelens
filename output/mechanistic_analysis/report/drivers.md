# What drives resisting a false norm: tables

Judged open-ended answers, full benchmark. Primary judge gemma-4-31b; intervals are 95% bootstrap intervals over records.

## 1. Effect of each cue on alignment (percentage points, paired on records)

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Country, nothing else stated | +14.4 [+13.4, +15.4], p<0.001 | +22.7 [+21.6, +23.8], p<0.001 | +23.3 [+22.2, +24.4], p<0.001 |
| Country, value stated | +11.1 [+9.7, +12.6], p<0.001 | +16.7 [+15.1, +18.3], p<0.001 | +14.0 [+12.6, +15.4], p<0.001 |
| Country, true norm stated | +16.7 [+15.8, +17.5], p<0.001 | +15.2 [+14.4, +16.1], p<0.001 | +13.1 [+12.4, +13.9], p<0.001 |
| Country, false norm stated | -10.6 [-11.6, -9.6], p<0.001 | -8.6 [-9.5, -7.7], p<0.001 | -0.6 [-1.7, +0.5], p=0.288 |
| Value, nothing else stated | +15.3 [+13.8, +16.8], p<0.001 | +16.9 [+15.3, +18.5], p<0.001 | +19.0 [+17.5, +20.5], p<0.001 |
| Value, country stated | +12.7 [+11.2, +14.3], p<0.001 | +8.6 [+7.1, +10.1], p<0.001 | +10.6 [+9.4, +11.8], p<0.001 |
| Value, country and true norm stated | +1.6 [+0.7, +2.5], p<0.001 | -0.2 [-1.1, +0.7], p=0.758 | +2.9 [+2.2, +3.5], p<0.001 |
| True norm, nothing else stated | +32.9 [+31.8, +33.9], p<0.001 | +37.4 [+36.3, +38.5], p<0.001 | +30.9 [+29.9, +31.9], p<0.001 |
| True norm, country stated | +35.2 [+34.1, +36.2], p<0.001 | +29.9 [+29.0, +30.9], p<0.001 | +20.7 [+19.9, +21.6], p<0.001 |
| False norm, nothing else stated | -15.1 [-16.1, -14.1], p<0.001 | -15.6 [-16.8, -14.5], p<0.001 | -7.3 [-8.4, -6.3], p<0.001 |
| False norm, country stated | -40.0 [-41.2, -39.0], p<0.001 | -46.9 [-48.0, -45.8], p<0.001 | -31.2 [-32.3, -30.1], p<0.001 |
| Country × true norm (interaction) | +2.3 [+1.1, +3.6], p=0.002 | -7.5 [-8.7, -6.1], p<0.001 | -10.2 [-11.4, -9.0], p<0.001 |

Second judge (qwen-3.6-27b):

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Country, nothing else stated | +14.7 [+13.7, +15.6], p<0.001 | +22.9 [+21.9, +24.0], p<0.001 | +22.0 [+20.9, +23.0], p<0.001 |
| Country, value stated | +11.9 [+10.4, +13.4], p<0.001 | +17.3 [+15.7, +18.9], p<0.001 | +13.4 [+12.1, +14.8], p<0.001 |
| Country, true norm stated | +16.5 [+15.6, +17.4], p<0.001 | +16.0 [+15.1, +16.9], p<0.001 | +13.1 [+12.3, +13.9], p<0.001 |
| Country, false norm stated | -9.8 [-10.8, -8.9], p<0.001 | -9.0 [-10.0, -8.1], p<0.001 | -0.3 [-1.5, +0.8], p=0.559 |
| Value, nothing else stated | +15.7 [+14.3, +17.3], p<0.001 | +15.4 [+13.8, +16.9], p<0.001 | +20.1 [+18.7, +21.5], p<0.001 |
| Value, country stated | +13.3 [+11.8, +14.8], p<0.001 | +8.2 [+6.7, +9.7], p<0.001 | +12.3 [+11.0, +13.5], p<0.001 |
| Value, country and true norm stated | +1.5 [+0.6, +2.5], p=0.002 | -0.5 [-1.5, +0.5], p=0.361 | +3.5 [+2.8, +4.2], p<0.001 |
| True norm, nothing else stated | +31.6 [+30.6, +32.7], p<0.001 | +36.7 [+35.7, +37.8], p<0.001 | +30.1 [+29.0, +31.1], p<0.001 |
| True norm, country stated | +33.5 [+32.5, +34.5], p<0.001 | +29.8 [+28.9, +30.8], p<0.001 | +21.2 [+20.3, +22.1], p<0.001 |
| False norm, nothing else stated | -16.7 [-17.7, -15.6], p<0.001 | -15.1 [-16.2, -14.0], p<0.001 | -9.8 [-10.8, -8.7], p<0.001 |
| False norm, country stated | -41.2 [-42.3, -40.1], p<0.001 | -47.1 [-48.1, -46.0], p<0.001 | -32.0 [-33.2, -30.8], p<0.001 |
| Country × true norm (interaction) | +1.9 [+0.6, +3.1], p=0.008 | -6.9 [-8.2, -5.7], p<0.001 | -8.9 [-10.0, -7.7], p<0.001 |

## 2. How answers are labelled (aligned / partially aligned / misaligned)

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Country only | 57.9% / 22.9% / 19.2% | 64.1% / 21.7% / 14.2% | 75.2% / 11.6% / 13.2% |
| Country + false norm | 17.9% / 11.2% / 70.9% | 17.1% / 6.8% / 76.0% | 44.0% / 2.4% / 53.6% |
| False norm, no country | 28.4% / 16.6% / 55.0% | 25.7% / 9.2% / 65.0% | 44.6% / 5.8% / 49.6% |

## 3. What predicts resisting (logistic regression, odds ratios; intervals from resampling countries)

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Right with no country and no norm | 1.63 [1.43, 1.83] | 1.87 [1.63, 2.19] | 1.74 [1.59, 1.89] |
| Right with the country alone | 3.45 [3.02, 4.04] | 3.49 [2.96, 4.16] | 3.24 [2.90, 3.71] |
| Right with the true norm alone | 1.40 [1.16, 1.68] | 1.02 [0.85, 1.25] | 1.11 [0.92, 1.29] |
| Right with country + true norm | 1.74 [1.21, 2.65] | 1.92 [1.29, 3.47] | 1.86 [1.44, 2.45] |
| Record has a value | 1.15 [1.01, 1.30] | 1.50 [1.31, 1.71] | 1.16 [1.04, 1.30] |
| Category: business_culture (vs etiquette) | 2.04 [1.72, 2.40] | 1.43 [1.19, 1.69] | 1.30 [1.11, 1.52] |
| Category: communication (vs etiquette) | 1.20 [0.99, 1.46] | 1.33 [1.08, 1.64] | 1.05 [0.90, 1.21] |
| Category: core_concepts (vs etiquette) | 1.18 [0.88, 1.67] | 1.12 [0.82, 1.50] | 0.61 [0.47, 0.82] |
| Category: dos_and_donts (vs etiquette) | 1.93 [1.54, 2.37] | 1.92 [1.56, 2.42] | 1.44 [1.18, 1.80] |
| Category: family (vs etiquette) | 1.24 [0.98, 1.55] | 0.89 [0.72, 1.13] | 0.73 [0.61, 0.88] |
| Category: greetings (vs etiquette) | 1.47 [1.18, 1.84] | 1.60 [1.31, 1.95] | 1.28 [1.09, 1.53] |
| Category: other_considerations (vs etiquette) | 1.37 [0.83, 2.12] | 1.29 [0.82, 1.96] | 1.13 [0.78, 1.60] |
| Category: religion (vs etiquette) | 1.03 [0.61, 1.56] | 0.91 [0.50, 1.45] | 1.10 [0.73, 1.59] |

## 4. By category: resisting, and resisting among records right with the country alone

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| etiquette | 14.5% / 20.3% (n=2579) | 14.5% / 19.7% (n=2579) | 44.2% / 50.9% (n=2579) |
| business_culture | 25.4% / 36.6% (n=1722) | 20.1% / 27.1% (n=1722) | 51.2% / 59.3% (n=1722) |
| communication | 15.1% / 22.7% (n=1494) | 16.8% / 23.6% (n=1494) | 42.7% / 51.5% (n=1494) |
| family | 14.1% / 22.9% (n=1228) | 12.5% / 17.0% (n=1228) | 32.1% / 40.8% (n=1228) |
| greetings | 17.5% / 25.1% (n=816) | 17.6% / 27.1% (n=816) | 46.7% / 56.5% (n=816) |
| core_concepts | 17.3% / 24.4% (n=608) | 18.1% / 23.0% (n=608) | 34.0% / 41.0% (n=608) |
| dos_and_donts | 28.9% / 37.6% (n=512) | 31.6% / 37.7% (n=512) | 58.8% / 62.9% (n=512) |
| other_considerations | 17.4% / 27.1% (n=155) | 17.4% / 24.7% (n=155) | 42.6% / 57.8% (n=155) |
| religion | 15.0% / 20.7% (n=147) | 14.3% / 19.4% (n=147) | 46.3% / 53.5% (n=147) |

## 5. Do models resist on the same records?

| Pair | Cohen's κ |
|---|---|
| Llama-3.1-8B / Gemma-3-12B | 0.33 [0.30, 0.35] |
| Llama-3.1-8B / Qwen-3.5-27B | 0.20 [0.18, 0.22] |
| Gemma-3-12B / Qwen-3.5-27B | 0.21 [0.19, 0.23] |

| Resisted by | 0 models | 1 models | 2 models | 3 models |
|---|---|---|---|---|
| Observed | 47.6% | 32.0% | 14.3% | 6.1% |
| If models were independent | 38.1% | 46.1% | 14.4% | 1.3% |

## 6. By country (75 countries)

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Resisting, lowest to highest country | 6.0% – 31.3% | 6.0% – 36.1% | 23.1% – 67.2% |
| Spearman ρ across countries: resisting vs country-only accuracy | +0.32 [+0.06, +0.53] | +0.39 [+0.18, +0.57] | +0.56 [+0.38, +0.69] |
| … resisting among records right with the country alone vs country-only accuracy | +0.13 | +0.30 | +0.44 |
| Spearman ρ across countries: internal conflict signal vs resisting | +0.28 [+0.05, +0.49] | +0.31 [+0.06, +0.53] | +0.77 [+0.65, +0.87] |
| … internal conflict signal vs country-only accuracy | +0.37 | +0.08 | +0.35 |
| Spearman ρ: resisting vs WVS self-expression axis | +0.17 [-0.17, +0.46] (country-only accuracy +0.55; 43 countries) | +0.21 [-0.11, +0.45] (country-only accuracy +0.44; 43 countries) | +0.25 [-0.05, +0.49] (country-only accuracy +0.41; 43 countries) |
| Spearman ρ: resisting vs WVS secular-rational axis | +0.10 [-0.22, +0.42] (country-only accuracy +0.26; 43 countries) | +0.13 [-0.18, +0.42] (country-only accuracy +0.15; 43 countries) | +0.18 [-0.12, +0.45] (country-only accuracy +0.24; 43 countries) |
| Spearman ρ: resisting vs WVS distance from the United States | -0.06 [-0.37, +0.31] (country-only accuracy -0.56; 42 countries) | -0.12 [-0.42, +0.22] (country-only accuracy -0.34; 42 countries) | -0.17 [-0.45, +0.16] (country-only accuracy -0.30; 42 countries) |
| Lowest five | SSD 6.0%, ITA 9.4%, PNG 10.5%, ARG 10.7%, SOM 10.7% | BRA 6.0%, PER 7.0%, MKD 7.7%, CHL 7.8%, AUT 7.9% | MKD 23.1%, HUN 24.3%, CYP 25.8%, VEN 27.3%, MUS 27.9% |
| Highest five | AUS 31.3%, IDN 26.5%, CAN 25.3%, USA 25.2%, GBR 25.0% | KOR 36.1%, THA 30.6%, USA 30.3%, IDN 30.1%, CAN 29.1% | KOR 67.2%, JPN 67.2%, GBR 64.7%, USA 62.6%, THA 60.0% |

By world region: resisting / resisting among records right with the country alone / country-only accuracy

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Africa (9 countries) | 14.2% / 22.5% / 55.1% | 15.0% / 21.0% / 61.4% | 36.1% / 43.9% / 71.8% |
| Americas (9 countries) | 17.0% / 24.1% / 62.4% | 14.8% / 20.1% / 65.1% | 40.6% / 47.4% / 76.5% |
| Asia (29 countries) | 18.8% / 28.1% / 56.0% | 20.6% / 28.1% / 64.7% | 49.7% / 58.3% / 76.0% |
| Europe (22 countries) | 17.3% / 23.8% / 60.2% | 13.2% / 18.0% / 63.9% | 38.4% / 46.4% / 75.0% |
| Oceania (6 countries) | 20.8% / 30.7% / 58.5% | 17.8% / 24.2% / 63.8% | 47.0% / 55.5% / 74.9% |

Agreement between models on the country profile (Spearman ρ): Llama-3.1-8B / Gemma-3-12B +0.56; Llama-3.1-8B / Qwen-3.5-27B +0.55; Gemma-3-12B / Qwen-3.5-27B +0.70.

## 7. How answers to country + false norm treat the claim (full benchmark)

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Follows the claim | 70.4% [69.4%, 71.3%] | 72.9% [72.0%, 73.8%] | 46.8% [45.8%, 47.8%] |
| Follows it, with doubt | 4.5% [4.0%, 4.9%] | 8.6% [8.0%, 9.2%] | 9.2% [8.7%, 9.8%] |
| Challenges the claim | 18.6% [17.8%, 19.4%] | 15.0% [14.3%, 15.6%] | 43.8% [42.8%, 44.8%] |
| Sets the claim aside | 4.8% [4.4%, 5.2%] | 1.8% [1.6%, 2.1%] | 0.1% [0.0%, 0.1%] |
| No clear recommendation | 1.7% [1.4%, 1.9%] | 1.7% [1.4%, 2.0%] | 0.1% [0.0%, 0.2%] |
| Restates the claim as fact | 75.3% | 85.2% | 58.2% |
| Presents its advice as the country's custom | 97.3% | 98.4% | 93.1% |

Among answers judged aligned (resisting):

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Follows the claim | 12.2% | 14.8% | 2.0% |
| Follows it, with doubt | 3.3% | 11.7% | 3.1% |
| Challenges the claim | 76.2% | 68.5% | 94.7% |
| Sets the claim aside | 7.7% | 3.8% | 0.1% |
| No clear recommendation | 0.5% | 1.3% | 0.1% |

Among answers not judged aligned:

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Follows the claim | 83.1% | 84.9% | 82.0% |
| Follows it, with doubt | 4.7% | 7.9% | 14.0% |
| Challenges the claim | 6.1% | 3.9% | 3.8% |
| Sets the claim aside | 4.2% | 1.4% | 0.1% |
| No clear recommendation | 1.9% | 1.8% | 0.1% |

Share judged aligned, by stance:

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Follows the claim | 3.1% | 3.5% | 1.9% |
| Follows it, with doubt | 13.0% | 23.3% | 14.9% |
| Challenges the claim | 73.1% | 78.5% | 95.2% |
| Sets the claim aside | 28.7% | 35.5% | – |
| No clear recommendation | 5.8% | 12.6% | – |

## 8. Stance under the mechanistic interventions: Llama-3.1-8B

| Prompt and intervention | n | Follows the claim | Follows it, with doubt | Challenges the claim | Sets the claim aside | No clear recommendation | Restates claim | Names the country's custom |
|---|---|---|---|---|---|---|---|---|
| Country + false norm, untouched | 400 | 69.8% | 4.5% | 18.2% | 5.5% | 2.0% | 74.5% | 96.5% |
| … claim blocked in the claim window | 400 | 31.2% | 2.0% | 48.5% | 16.0% | 2.2% | 28.5% | 96.8% |
| … claim blocked for the selected heads | 400 | 32.8% | 5.5% | 50.5% | 9.8% | 1.5% | 35.2% | 97.0% |
| … claim blocked in every layer | 400 | 13.8% | 1.5% | 53.0% | 30.5% | 1.2% | 11.0% | 96.8% |
| … claim erased before the band | 400 | 14.8% | 2.0% | 56.8% | 25.0% | 1.5% | 13.0% | 97.5% |
| … claim erased after the first block | 400 | 11.8% | 1.0% | 50.2% | 33.2% | 3.8% | 8.0% | 95.2% |
| False norm, no country | 400 | 43.0% | 2.8% | 31.2% | 18.8% | 4.2% | 41.2% | 5.2% |
| Country only (claim never shown) | 400 | 11.5% | 1.8% | 53.8% | 31.0% | 2.0% | 8.8% | 96.2% |

## 8. Stance under the mechanistic interventions: Gemma-3-12B

| Prompt and intervention | n | Follows the claim | Follows it, with doubt | Challenges the claim | Sets the claim aside | No clear recommendation | Restates claim | Names the country's custom |
|---|---|---|---|---|---|---|---|---|
| Country + false norm, untouched | 400 | 76.2% | 7.8% | 11.2% | 3.0% | 1.8% | 88.8% | 98.0% |
| … claim blocked in the claim window | 400 | 20.5% | 3.5% | 57.8% | 17.2% | 1.0% | 17.8% | 99.2% |
| … claim blocked for the selected heads | 400 | 64.0% | 7.8% | 20.5% | 5.2% | 2.5% | 74.8% | 98.2% |
| … claim blocked in every layer | 400 | 7.2% | 0.8% | 60.2% | 29.8% | 2.0% | 5.0% | 98.2% |
| … claim erased before the band | 400 | 8.5% | 0.0% | 62.0% | 27.5% | 2.0% | 6.2% | 99.0% |
| … claim erased after the first block | 400 | 8.5% | 1.8% | 59.8% | 28.5% | 1.5% | 4.8% | 98.2% |
| False norm, no country | 400 | 57.2% | 8.0% | 29.0% | 2.8% | 3.0% | 69.5% | 2.8% |
| Country only (claim never shown) | 400 | 7.0% | 1.5% | 63.0% | 27.8% | 0.8% | 3.5% | 98.8% |

## 8. Stance under the mechanistic interventions: Qwen-3.5-27B

| Prompt and intervention | n | Follows the claim | Follows it, with doubt | Challenges the claim | Sets the claim aside | No clear recommendation | Restates claim | Names the country's custom |
|---|---|---|---|---|---|---|---|---|
| Country + false norm, untouched | 200 | 39.0% | 11.0% | 50.0% | 0.0% | 0.0% | 50.0% | 93.5% |
| … claim erased before the band | 200 | 9.0% | 2.5% | 79.0% | 9.0% | 0.5% | 9.0% | 99.5% |
| … claim erased after the first block | 200 | 8.5% | 0.5% | 78.5% | 12.0% | 0.5% | 6.5% | 99.5% |
| False norm, no country | 200 | 35.0% | 3.5% | 59.5% | 1.0% | 1.0% | 41.0% | 7.0% |

## 9. Framing of the false claim, country stated: resisting, and change against the bare statement on the same records

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Stated as is (benchmark) | 18.0% (n=400) | 15.2% (n=400) | 48.0% (n=200) |
| Attributed to a guide to local customs | 6.5%; -11.5 [-15.2, -8.0], p<0.001 | 7.5%; -7.8 [-11.2, -4.5], p<0.001 | 5.0%; -43.0 [-49.5, -36.0], p<0.001 |
| Stated before the country line | 38.8%; +20.8 [+16.2, +25.5], p<0.001 | 43.5%; +28.2 [+23.5, +32.8], p<0.001 | 80.5%; +32.5 [+26.0, +39.5], p<0.001 |
| Attributed to hearsay ("Someone told me this") | 30.5%; +12.5 [+8.0, +17.0], p<0.001 | 45.0%; +29.8 [+25.2, +34.5], p<0.001 | 86.0%; +38.0 [+31.0, +45.0], p<0.001 |
| Followed by the user's doubt ("I am not sure this is right") | 43.2%; +25.2 [+20.2, +30.2], p<0.001 | 59.5%; +44.2 [+39.2, +49.2], p<0.001 | 91.0%; +43.0 [+36.0, +50.0], p<0.001 |

Second judge (qwen-3.6-27b):

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Stated as is (benchmark) | 16.0% | 14.0% | 48.0% |
| Attributed to a guide to local customs | 5.5%; -10.5 [-14.0, -7.0], p<0.001 | 6.2%; -7.8 [-10.8, -4.8], p<0.001 | 3.5%; -44.5 [-51.5, -37.5], p<0.001 |
| Stated before the country line | 39.0%; +23.0 [+18.2, +27.8], p<0.001 | 43.0%; +29.0 [+24.2, +33.5], p<0.001 | 80.5%; +32.5 [+25.5, +40.0], p<0.001 |
| Attributed to hearsay ("Someone told me this") | 32.2%; +16.2 [+11.8, +20.8], p<0.001 | 41.5%; +27.5 [+22.5, +32.2], p<0.001 | 75.0%; +27.0 [+20.0, +34.5], p<0.001 |
| Followed by the user's doubt ("I am not sure this is right") | 42.5%; +26.5 [+21.8, +31.5], p<0.001 | 56.0%; +42.0 [+36.5, +47.0], p<0.001 | 89.0%; +41.0 [+34.0, +48.5], p<0.001 |

Share of answers that challenge the claim / follow it with doubt / follow it:

| | Llama-3.1-8B | Gemma-3-12B | Qwen-3.5-27B |
|---|---|---|---|
| Attributed to a guide to local customs | 6.8% / 3.0% / 86.5% | 6.0% / 7.5% / 84.2% | 8.0% / 17.5% / 74.5% |
| Stated before the country line | 43.8% / 4.5% / 44.5% | 45.2% / 8.8% / 39.0% | 85.0% / 4.0% / 11.0% |
| Attributed to hearsay ("Someone told me this") | 37.0% / 4.5% / 54.8% | 46.2% / 10.8% / 39.0% | 88.0% / 3.5% / 7.0% |
| Followed by the user's doubt ("I am not sure this is right") | 50.5% / 7.8% / 37.5% | 71.5% / 9.2% / 15.2% | 95.5% / 1.5% / 3.0% |

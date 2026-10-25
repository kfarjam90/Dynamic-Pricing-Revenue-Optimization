# Pricing Review: Executive Summary

## Bottom line
Raising prices on 19 SKUs and trimming 5 is expected to lift 30-day gross profit by **€53.7k (+10.5%, €513.9k → €567.6k)** while revenue rises 4.3% and units fall 4.0%. 18 of 24 recommendations are high-confidence and carry 97% of the lift. The lift stays positive in every stress test, but 6 prices lie beyond the range we have tested, so we recommend a controlled pilot before full rollout.

## Main findings
- **Baseline (clean data):** revenue €4.94M, gross profit €2.43M (49.1% margin), 75,649 units, average price €65.35. Revenue and profit reconcile exactly to price × cost × units after fixing a €99,999 revenue placeholder, one duplicate row, a negative session count, a 115% discount, and a missing competitor price.
- **Stockouts are the biggest data problem.** Reported stockout rate is 21.4% (23.0% after correcting 46 days that sold exactly the opening stock). France is worst at 31.6% (SKU-002 and SKU-020 are out of stock about 95% of days). Sales on those days are a floor on demand, not demand.
- **Demand is price-elastic.** A 1% price rise cuts demand by about **1.7%** (pooled, controlling for traffic, promo, weekday, competitor price). Ignoring stockouts understates this (−1.4); a no-controls regression gives −0.26. By category: **Sneakers −2.25** (most sensitive), Sportswear −1.78, Apparel −1.48, Accessories −1.41. 22 of 24 SKUs are clearly elastic; SKU-015 and SKU-016 are roughly unit-elastic.
- **Forecast quality:** on a chronological hold-out (last 30 days) the censored demand model cuts RMSE to 2.9 units/day versus 5.2 for the best baseline. A gradient-boosting model trained on censoring-corrected targets matches it (3.0), while the same model trained on raw sales is worse (3.8).
- **Revenue-max ≠ profit-max.** With elastic demand and ~49% margins, revenue is maximised by cutting price (15 SKUs priced below current), but that earns €545k profit versus €568k at profit-maximising prices. Where stock is scarce the two coincide.

## Three priority recommendations
| # | Action | Current → New | 30-day profit lift |
|---|---|---|---|
| 1 | **FR Sportswear, SKU-020 and SKU-023:** raise to the +15% limit. Both are near-permanently sold out. | €60.10 → €69.11; €80.64 → €92.73 | **+€16.1k** |
| 2 | **Sneakers, SKU-001 (DE) and SKU-002 (FR):** raise ~12%, to where demand just meets stock. | €90.72 → €101.45; €114.73 → €128.49 | **+€12.5k** |
| 3 | **Stock-constrained Apparel/Accessories, SKU-011, 014, 017, 018:** raise 10–15%. Demand exceeds available stock at current prices. | e.g. €55.86 → €62.56; €38.08 → €41.88 | **+€12.7k** |

These three groups deliver about €41k, or 77% of the total. SKU-004, SKU-005 and SKU-019 show lifts under 1% and low confidence, so keep them at current prices unless the pilot says otherwise. No recommendation breaches a margin floor (recommended margins are 43–65%), the price-change limits, or available stock.

## Uncertainty
- **Elasticity estimation error alone:** 90% interval €49k–€57k.
- **Stress tests:** demand 25% more elastic → +€45k; baseline demand 15% lower → +€38k; both together → +€30k.
- **Not modelled:** competitor reactions (competitor price held at today's level), seasonality beyond four months, and promotions (all prices are everyday prices).
- **Extrapolation:** tested prices ran from −12% to +12% of list. Six recommendations (SKU-007, 010, 016, 017, 020, 023 at +15%) sit beyond that, worth about €23k of the lift.
- **Weakest estimates:** SKU-002, 007, 016 and 020 are 70–95% stockout days, so their own elasticities are imprecise. We shrink them towards their category average.

## Pre-rollout controlled experiment
1. **Scope:** the top 10 SKUs by lift, including all six beyond the tested range, in FR and DE first.
2. **Design:** randomise by SKU-day (or by traffic split) between current price (control), recommended price, and a half-step price. Keep promotions off and log competitor prices daily.
3. **Duration:** 3 weeks. The daily demand noise is about 12%, which is enough to detect a 15–20% unit change per SKU.
4. **Metrics:** profit per session (primary), units, conversion, and days of stock cover.
5. **Guardrails:** stop a SKU if units fall more than 25% below the model forecast for 5 consecutive days. Roll out where measured profit per session beats control; re-estimate elasticities from the experiment.


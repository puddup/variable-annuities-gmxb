# QaBit · Variable Annuities, GMxB (Deep) Hedging

A unifying framework for pricing **variable annuities with guarantees** (GMDB / GMAB /
GMWB), a **neural hedging policy** that learns to replicate them across market regimes, and
a **model-risk instrument** that reads off where a real GMxB desk bleeds — including its
structural short-gamma book.

---

## The series

### [Part 1 — A Unifying Valuation Framework for Variable Annuities](01_variable_annuities_framework.md)
One contract class, three composable guarantee floors (living / death / surrender), two
valuation schemes (static and static-plus-surrender). The death benefit as a Titanic
option; the surrender option by Least-Squares Monte Carlo; fair fees on the fee–penalty
surface.

### [Part 2 — Deep Hedging Across Regimes](02_deep_hedging_across_regimes.md)
Classical mark-to-market hedging and the rate-risk floor it cannot cross, then a single
neural policy — read as a **PINN**: the financial loss is the objective, structural priors
are hard-coded boundary conditions, the network learns the residual. Vanilla, barrier,
cliquet and lookback, across three regimes.

### [Part 3 — GMxB Model Risk & the Greeks of the Hedge](03_gmxb_model_risk_and_greeks.md)
The deep hedge across the whole GM(A/W/D)B family, the **short-gamma / vega / vanna** book
made precise (fixed vs ratchet vs reset, with and without lapse), and the model-risk
surfaces that rank the threats: **equity vol first, rates second, mortality a distant
third.**

---

## The headline result

> Train one deep hedge on a calibrated VA world (Heston vol, Vasicek rates, stochastic
> mortality), freeze it, and evaluate under shifted worlds. The hedged-P&L surfaces are a
> decision-grade map of model risk.

---

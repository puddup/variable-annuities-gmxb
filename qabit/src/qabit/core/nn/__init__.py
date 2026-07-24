"""Neural hedging — losses, features, instruments, and the deep hedger.

* :mod:`loss`        — :class:`HedgingLoss` family (variance, MSE, ES, Jones).
* :mod:`feature`     — context-bound :class:`Feature` objects + :class:`FeatureSet`,
                       spanning the full SDE state (equity, rate, stochastic vol,
                       hazard, inventory).
* :mod:`instrument`  — :class:`Hedgeable` protocol, :class:`FixedHedge`,
                       :class:`ResetHedge`.
* :mod:`hedger`      — :class:`HedgingMLP`, :class:`DeepHedger`, and the
                       liability adapters that select which Monte-Carlo function
                       the hedger calls on the target.
"""

from qabit.core.nn.loss import (
    HedgingLoss,
    VarianceLoss,
    MSELoss,
    ExpectedShortfall,
    JonesLoss,
    DownsideJonesLoss,
    EntropicLoss,
)
from qabit.core.nn.feature import (
    FeatureContext,
    Feature,
    Moneyness,
    Spot,
    TimeToMaturity,
    Volatility,
    InstantaneousVol,
    ShortRate,
    HazardRate,
    Survival,
    BarrierBreached,
    Deceased,
    AnniversaryMax,
    GuaranteeMoneyness,
    AnniversaryPhase,
    PendingDeathBenefit,
    PeriodReturn,
    PrevHoldings,
    KnockOutLiveness,
    CliquetLiveness,
    MortalityLiveness,
    FeatureSet,
)
from qabit.core.nn.adversial import (
    AmbiguityAverseLoss,
    QuantilePartitioner,
    KMeansPartitioner,
)
from qabit.core.nn.regret import (
    RegretRobustLoss,
    RelativeRegretLoss,
    TiltedMixtureLoss,
    AnchoredLoss,
    measure_floor_stats,
    robust_floor,
)
from qabit.core.nn.instrument import Hedgeable, FixedHedge, ResetHedge
from qabit.core.nn.hedger import FeatureStandardizer, HedgingMLP, BoundedHedgingMLP, DeepHedger

__all__ = [
    "HedgingLoss",
    "VarianceLoss",
    "MSELoss",
    "ExpectedShortfall",
    "JonesLoss",
    "DownsideJonesLoss",
    "EntropicLoss",
    "FeatureContext",
    "Feature",
    "Moneyness",
    "Spot",
    "TimeToMaturity",
    "Volatility",
    "InstantaneousVol",
    "ShortRate",
    "HazardRate",
    "Survival",
    "BarrierBreached",
    "Deceased",
    "AnniversaryMax",
    "GuaranteeMoneyness",
    "AnniversaryPhase",
    "PendingDeathBenefit",
    "PeriodReturn",
    "PrevHoldings",
    "KnockOutLiveness",
    "CliquetLiveness",
    "MortalityLiveness",
    "FeatureSet",
    "AmbiguityAverseLoss",
    "QuantilePartitioner",
    "KMeansPartitioner",
    "RegretRobustLoss",
    "RelativeRegretLoss",
    "TiltedMixtureLoss",
    "AnchoredLoss",
    "measure_floor_stats",
    "robust_floor",
    "Hedgeable",
    "FixedHedge",
    "ResetHedge",
    "FeatureStandardizer",
    "HedgingMLP",
    "BoundedHedgingMLP",
    "DeepHedger"
]

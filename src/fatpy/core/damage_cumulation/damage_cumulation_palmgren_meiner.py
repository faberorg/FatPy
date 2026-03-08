"""Damage Accumulation Rule according to Palmgren-Miner.

Resources:
    [1] Graphical interpretation of the change in the S-N curve based on the
        chosen version can be found e.g. in this open-access paper:
        http://dx.doi.org/10.5545/sv-jme.2013.1348
    [2] Miner, M. A. (1945). Cumulative damage in fatigue. Journal of Applied
        Mechanics, 12(3), 159-164.
"""

# import numpy as np
# from numpy.typing import NDArray


def damage_cumulation_elementary(
    slope_k: float,
    constant: float,
    sig: float,
    number_occurrences: int,
) -> float:
    r"""Elementary version of Palmgren-Miner linear damage accumulation.

    The same slope k of the S-N curve below and above the fatigue limit.

    ??? abstract "Math Equations"
        $$
        D = n/N = n\,\frac{\sigma^k}{C}
        $$

    """
    total_occurrences: float = constant / sig**slope_k

    damage: float = number_occurrences / total_occurrences

    return damage


def damage_cumulation_basic(
    slope_k: float,
    constant: float,
    sig_fl: float,
    sig: float,
    number_occurrences: int,
) -> float:
    """Basic version of Palmgren-Miner linear damage accumulation.

    The S-N curve gets horizontal at the fatigue limit, no damage for stresses beneath.
    Otherwise elementary damage is calculated.
    """
    if sig < sig_fl:
        damage = 0.0
    else:
        damage = damage_cumulation_elementary(
            slope_k, constant, sig, number_occurrences
        )

    return damage


def damage_cumulation_haibach(
    slope_k: float,
    constant: float,
    sig_fl: float,
    sig: float,
    number_occurrences: int,
) -> float:
    r"""Haibach version of Palmgren-Miner linear damage accumulation.

    the original slope_k is modified below fatigue limit to 2*slope_k-1.

    ??? abstract "Math Equations"
        $$
        D = \frac{n}{C}\,\frac{\sigma^{2k-1}}{\sigma_\mathrm{FL}^{k-1}}
        $$

    """
    if sig < sig_fl:
        damage: float = (
            number_occurrences
            * sig ** (2 * slope_k - 1)
            / (constant * sig_fl ** (slope_k - 1))
        )
    else:
        damage = damage_cumulation_elementary(
            slope_k, constant, sig, number_occurrences
        )

    return damage

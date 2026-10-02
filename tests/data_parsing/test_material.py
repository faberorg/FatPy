"""Test functions for the material data model.

Covers the `SNCurve` class (regression fit, stress/life conversions,
survival probability shift, dict round trips) and the nomenclature-driven
`Material` container (WG6 aliases, dict/JSON round trips).
"""

from dataclasses import fields
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest

from fatpy.data_parsing.material import NOMENCLATURE, Material, SNCurve

# ===========================================================================
# SNCurve
# ===========================================================================


@pytest.fixture
def curve() -> SNCurve:
    """A fully defined torsion curve with known scatter."""
    return SNCurve(
        description="median",
        fatigue_limit=300.0,
        exponent=5.0,
        n_cycles_at_limit=1e6,
        load_mode="tor",
        stdev_log_stress=0.1,
    )


def test_sncurve_requires_its_parameters() -> None:
    """A curve cannot be built without fatigue limit, exponent and knee."""
    with pytest.raises(TypeError):
        SNCurve(description="incomplete")  # type: ignore[call-arg]


def test_from_test_points_fits_exact_power_law() -> None:
    """Two points with no scatter are fit exactly (w=2, FL=10 at N=1e6)."""
    fitted = SNCurve.from_test_points(
        "synthetic",
        stress_amp=[1000.0, 100.0],
        n_cycles=[100.0, 10_000.0],
        n_cycles_at_limit=1e6,
    )
    assert fitted.exponent == pytest.approx(2.0)
    assert fitted.fatigue_limit == pytest.approx(10.0)
    assert fitted.n_cycles_at_limit == 1e6
    assert fitted.stdev_log_stress is None
    assert fitted.stress_amp_at(100.0) == pytest.approx(1000.0)
    assert fitted.cycles_at(1000.0) == pytest.approx(100.0)
    assert fitted.stress_amp_at(1e6) == pytest.approx(fitted.fatigue_limit)
    assert fitted.coefficient == pytest.approx(1e8)


def test_from_test_points_ignores_runouts() -> None:
    """A runout far off the trend line does not affect the regression."""
    fitted = SNCurve.from_test_points(
        "with runout",
        stress_amp=[1000.0, 100.0, 5000.0],
        n_cycles=[100.0, 10_000.0, 50.0],
        runout=[False, False, True],
        n_cycles_at_limit=1e6,
    )
    assert fitted.exponent == pytest.approx(2.0)
    assert fitted.fatigue_limit == pytest.approx(10.0)


def test_from_test_points_forwards_extra_fields() -> None:
    """Keyword arguments such as `load_mode` reach the constructor."""
    fitted = SNCurve.from_test_points(
        "tor", [1000.0, 100.0], [100.0, 10_000.0], load_mode="tor"
    )
    assert fitted.load_mode == "tor"


def test_from_test_points_requires_two_failures() -> None:
    """Fitting needs at least two broken (non-runout) specimens."""
    with pytest.raises(ValueError, match="two failed"):
        SNCurve.from_test_points("bad", stress_amp=[1000.0], n_cycles=[100.0])
    with pytest.raises(ValueError, match="two failed"):
        SNCurve.from_test_points(
            "bad",
            stress_amp=[1000.0, 100.0],
            n_cycles=[100.0, 10_000.0],
            runout=[True, False],
        )


def test_from_test_points_rejects_increasing_stress() -> None:
    """A non-decreasing stress-vs-life trend is refused, not silently fit."""
    with pytest.raises(ValueError, match="does not decrease"):
        SNCurve.from_test_points(
            "wrong order", stress_amp=[100.0, 1000.0], n_cycles=[100.0, 10_000.0]
        )


def test_stress_amp_and_cycles_are_inverse(curve: SNCurve) -> None:
    """`cycles_at` and `stress_amp_at` invert each other above the limit."""
    n = curve.cycles_at(450.0)
    assert curve.stress_amp_at(n) == pytest.approx(450.0)


def test_below_fatigue_limit_is_infinite_life(curve: SNCurve) -> None:
    """At or below the fatigue limit the life is infinite."""
    assert curve.cycles_at(300.0) == float("inf")
    assert curve.stress_amp_at(1e9) == 300.0


def test_at_survival_prob_unchanged_at_same_probability(curve: SNCurve) -> None:
    """Asking for the curve's own survival probability changes nothing."""
    assert curve.at_survival_prob(0.5).fatigue_limit == pytest.approx(300.0)


def test_at_survival_prob_matches_lognormal_formula(curve: SNCurve) -> None:
    """A higher (safer) survival probability lowers the fatigue limit."""
    z = NormalDist().inv_cdf
    expected = 300.0 * 10 ** ((z(1.0 - 0.975) - z(0.5)) * 0.1)
    design = curve.at_survival_prob(0.975)
    assert design.fatigue_limit == pytest.approx(expected)
    assert design.fatigue_limit < curve.fatigue_limit
    assert curve.at_survival_prob(0.1).fatigue_limit > curve.fatigue_limit


def test_at_survival_prob_requires_stdev(curve: SNCurve) -> None:
    """`at_survival_prob` refuses a curve with no scatter information."""
    curve.stdev_log_stress = None
    with pytest.raises(ValueError, match="stdev_log_stress"):
        curve.at_survival_prob(0.975)


@pytest.mark.parametrize("survival_prob", [0.0, 1.0])
def test_at_survival_prob_out_of_range_raises(
    curve: SNCurve, survival_prob: float
) -> None:
    """Probabilities outside (0, 1) are rejected (by `NormalDist`)."""
    with pytest.raises(ValueError):
        curve.at_survival_prob(survival_prob)


def test_sncurve_dict_round_trip_with_arrays() -> None:
    """Arrays become lists in `to_dict` and are rebuilt by `from_dict`."""
    original = SNCurve(
        description="c1",
        fatigue_limit=300.0,
        exponent=5.0,
        n_cycles_at_limit=1e6,
        stress_amp=np.array([300.0, 250.0]),
        n_cycles=np.array([1e6, 5e6]),
        runout=np.array([False, True]),
    )
    data = original.to_dict()
    assert data["stress_amp"] == [300.0, 250.0]
    assert data["runout"] == [False, True]
    assert "stdev_log_stress" not in data
    rebuilt = SNCurve.from_dict(data)
    assert rebuilt == original
    assert rebuilt.runout is not None and rebuilt.runout.dtype == bool


def test_sncurve_equality_ignores_test_points(curve: SNCurve) -> None:
    """`==` compares the curve parameters only, never raising on arrays."""
    with_points = SNCurve.from_dict(curve.to_dict() | {"stress_amp": [500.0, 400.0]})
    assert with_points == curve


# ===========================================================================
# Material
# ===========================================================================


@pytest.fixture
def steel() -> Material:
    """A partially filled-in `Material`; everything else stays ``None``."""
    return Material(
        material_name="42CrMo4",
        source="unit test",
        elastic_modulus=210_000.0,
        yield_strength=980.0,
        fat_lim_ten_m1=488.0,
        fat_lim_tor_m1=404.0,
    )


def test_unset_parameters_are_none(steel: Material) -> None:
    """No check at creation: unknown parameters simply stay ``None``."""
    assert steel.hardness is None
    assert Material(material_name="empty").fat_lim_ten_m1 is None


def test_nomenclature_covers_every_parameter_field() -> None:
    """Every field except identity/containers has a nomenclature entry."""
    others = {"material_name", "source", "sn_curves", "extra"}
    assert set(NOMENCLATURE) == {f.name for f in fields(Material)} - others


def test_describe_accepts_wg6_alias() -> None:
    """`describe` resolves the WG6 ``-1`` spelling to the attribute."""
    spec = Material.describe("fat_lim_ten_-1")
    assert spec.key == "fat_lim_ten_m1"
    assert spec.symbol == "FL_ten,-1"
    assert spec.unit == "MPa"


def test_describe_unknown_key_raises() -> None:
    """A key outside the nomenclature raises `KeyError`."""
    with pytest.raises(KeyError):
        Material.describe("hardnes")


def test_describe_marks_new_entries_as_proposed() -> None:
    """Rows not yet in the WG6 Excel file are flagged `proposed`."""
    for key in ("socie_findley_coef", "char_length", "thresh_sif_range"):
        assert Material.describe(key).proposed is True


def test_from_dict_normalizes_aliases_and_keeps_extra() -> None:
    """`from_dict` maps WG6 keys to attributes and keeps unknown ones."""
    material = Material.from_dict(
        {
            "material_name": "S355",
            "fat_lim_ten_-1": 250.0,
            "cyc_hard_coeff": 1100.0,
            "hardness": None,
            "unknown_param": 1.23,
        }
    )
    assert material.fat_lim_ten_m1 == 250.0
    assert material.cyc_hard_coef == 1100.0
    assert material.hardness is None
    assert material.extra == {"unknown_param": 1.23}


def test_from_dict_without_material_name_raises() -> None:
    """``material_name`` is the only required field."""
    with pytest.raises(TypeError):
        Material.from_dict({"fat_lim_ten_-1": 250.0})


def test_to_dict_uses_wg6_spelling_and_skips_none(steel: Material) -> None:
    """`to_dict` uses WG6 spellings and omits undefined parameters."""
    data = steel.to_dict()
    assert data["fat_lim_ten_-1"] == 488.0
    assert "hardness" not in data


def test_to_dict_uses_plain_key_for_proposed_parameters() -> None:
    """A ``proposed`` nomenclature entry keeps its Python spelling."""
    data = Material(material_name="S355", char_length=12.5).to_dict()
    assert data["char_length"] == 12.5
    assert "L_CHAR" not in data


def test_to_dict_excludes_sn_curves_by_default(steel: Material, curve: SNCurve) -> None:
    """`to_dict` omits S-N curves unless `include_sn_curves` is set."""
    steel.sn_curves.append(curve)
    assert "sn_curves" not in steel.to_dict()
    assert "sn_curves" in steel.to_dict(include_sn_curves=True)


def test_dict_round_trip(steel: Material, curve: SNCurve) -> None:
    """A material, curves included, survives `to_dict` / `from_dict`."""
    steel.sn_curves.append(curve)
    assert Material.from_dict(steel.to_dict(include_sn_curves=True)) == steel


def test_from_dict_accepts_existing_sn_curve_instances(curve: SNCurve) -> None:
    """`from_dict` passes through `SNCurve` objects given directly."""
    material = Material.from_dict({"material_name": "S355", "sn_curves": [curve]})
    assert material.sn_curves == [curve]


def test_json_round_trip(steel: Material, curve: SNCurve, tmp_path: Path) -> None:
    """A material (with an S-N curve) survives a JSON round trip."""
    steel.sn_curves.append(curve)
    path = tmp_path / "material.json"
    steel.to_json(path)
    assert Material.from_json(path) == steel


def test_sn_curve_lookup(steel: Material, curve: SNCurve) -> None:
    """`sn_curve` finds a curve by load mode and stress ratio."""
    steel.sn_curves.append(curve)
    assert steel.sn_curve("tor") is curve
    assert steel.sn_curve("tor", stress_ratio=0.0) is None
    assert steel.sn_curve("ten") is None


def test_summary_lists_name_parameters_and_curves(
    steel: Material, curve: SNCurve
) -> None:
    """`summary` shows the name, defined parameters (with units) and curves."""
    steel.sn_curves.append(curve)
    text = steel.summary()
    assert "42CrMo4" in text
    assert "Young's modulus" in text and "MPa" in text
    assert "[S-N curve] median" in text
    assert "Hardness" not in text


def test_repr_shows_only_defined_values(steel: Material, curve: SNCurve) -> None:
    """`__repr__` shows set parameters, curve count and extras only."""
    steel.sn_curves.append(curve)
    steel.extra["custom"] = 1.0
    text = repr(steel)
    assert "elastic_modulus=210000.0" in text
    assert "hardness" not in text
    assert "sn_curves=<1>" in text
    assert "extra={'custom': 1.0}" in text

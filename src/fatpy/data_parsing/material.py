"""Material properties parsing module.

Container for the material properties used as input to FatPy fatigue methods.
The module only stores data: it does not check whether the parameters a
method needs are defined (each method does that itself) and it never
converts units.

Parameter names follow the WG6 nomenclature file ``nomenclature_v01.xlsx``.

Conventions:
    - Units are fixed by the nomenclature (see `NOMENCLATURE`): stresses
      and moduli in MPa (also E and G), lengths in mm, etc. Values must be
      given in these units.
    - Fatigue limits are stress amplitudes.
    - A parameter that is not known is ``None``.
    - WG6 keys ending in ``-1`` (e.g. ``fat_lim_ten_-1``) are not valid Python
      names, so the attribute is spelled ``fat_lim_ten_m1`` instead.
      `Material.from_dict` and `Material.describe` accept both spellings.

Examples:
    >>> steel = Material(
    ...     material_name="42CrMo4",
    ...     ult_tensile_strength=1100,
    ...     yield_strength=980,
    ...     fat_lim_ten_m1=488,
    ...     fat_lim_tor_m1=404,
    ... )
    >>> steel.fat_lim_ten_m1
    488
"""

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Material", "SNCurve", "ParamSpec", "NOMENCLATURE", "LoadMode"]


# ---------------------------------------------------------------------------
# S-N curve
# ---------------------------------------------------------------------------

LoadMode = Literal["ten", "ben", "rben", "tor", "tor_A", "tor_B"]


@dataclass
class SNCurve:
    """One S-N curve of the material: σ_a^w · N = C, constant below the knee.

    A material can hold several curves (different load modes, stress ratios,
    survival probabilities, specimens...). Give each one a clear
    `description`.

    Attributes:
        description: Free-text identifier (e.g. "push-pull, batch A, R=-1").
        fatigue_limit: Stress amplitude at the knee, in MPa.
        exponent: Basquin exponent w (slope in log-log space is ``-1/w``).
        n_cycles_at_limit: Number of cycles at the knee.
        load_mode: Type of loading, one of `LoadMode`.
        stress_ratio: Load ratio R = σ_min / σ_max.
        survival_prob: Survival probability the curve represents (0.5 =
            median curve).
        stdev_log_stress: Scatter of ``log10(stress_amp)`` around the fit.
        stress_amp: Raw stress-amplitude test points, in MPa.
        n_cycles: Raw cycles-to-failure test points.
        runout: ``True`` for a specimen that did not fail (ran out).
    """

    description: str
    fatigue_limit: float
    exponent: float
    n_cycles_at_limit: float
    load_mode: LoadMode = "ten"
    stress_ratio: float = -1.0
    survival_prob: float = 0.5
    stdev_log_stress: float | None = None
    # experimental points (optional, ignored by ==)
    stress_amp: NDArray[np.float64] | None = field(default=None, compare=False)
    n_cycles: NDArray[np.float64] | None = field(default=None, compare=False)
    runout: NDArray[np.bool_] | None = field(default=None, compare=False)

    @classmethod
    def from_test_points(
        cls,
        description: str,
        stress_amp: ArrayLike,
        n_cycles: ArrayLike,
        runout: ArrayLike | None = None,
        *,
        n_cycles_at_limit: float = 1e7,
        **kwargs: Any,
    ) -> "SNCurve":
        """Fit the exponent and fatigue limit from raw test points.

        Performs a least-squares fit in log-log space using only the failed
        (non-runout) specimens.

        Args:
            description: Free-text identifier for the resulting curve.
            stress_amp: Stress amplitude of each specimen, in MPa.
            n_cycles: Cycles to failure (or to runout) of each specimen.
            runout: ``True`` for specimens that did not fail. Defaults to
                treating every specimen as failed.
            n_cycles_at_limit: Life, in cycles, at which the fatigue limit
                is evaluated (the assumed knee position).
            **kwargs: Extra fields forwarded to the `SNCurve` constructor
                (e.g. `load_mode`, `stress_ratio`).

        Returns:
            The fitted `SNCurve`.

        Raises:
            ValueError: If fewer than two specimens failed, or if the fitted
                curve does not decrease with life.
        """
        s = np.asarray(stress_amp, dtype=float)
        n = np.asarray(n_cycles, dtype=float)
        ro = np.zeros_like(s, bool) if runout is None else np.asarray(runout, bool)
        broken = ~ro
        if broken.sum() < 2:
            raise ValueError("At least two failed specimens are needed for the fit")
        slope, intercept = np.polyfit(np.log10(n[broken]), np.log10(s[broken]), 1)
        if slope >= 0:
            raise ValueError(
                "Stress does not decrease with life in the test points; "
                "check the order of stress_amp and n_cycles"
            )
        residuals = np.log10(s[broken]) - (intercept + slope * np.log10(n[broken]))
        return cls(
            description=description,
            fatigue_limit=float(
                10 ** (intercept + slope * math.log10(n_cycles_at_limit))
            ),
            exponent=float(-1.0 / slope),
            n_cycles_at_limit=n_cycles_at_limit,
            stdev_log_stress=(
                float(np.std(residuals, ddof=1)) if broken.sum() > 2 else None
            ),
            stress_amp=s,
            n_cycles=n,
            runout=ro,
            **kwargs,
        )

    @property
    def coefficient(self) -> float:
        """C in σ_a^w · N = C."""
        return float(self.fatigue_limit**self.exponent * self.n_cycles_at_limit)

    def stress_amp_at(self, n_cycles: float) -> float:
        """Stress amplitude for a given life, flat at the fatigue limit beyond the knee.

        Args:
            n_cycles: Target life, in cycles.

        Returns:
            Stress amplitude, in MPa.
        """
        n = max(n_cycles, 1.0)
        if n >= self.n_cycles_at_limit:
            return self.fatigue_limit
        return float(
            self.fatigue_limit * (self.n_cycles_at_limit / n) ** (1.0 / self.exponent)
        )

    def cycles_at(self, stress_amp: float) -> float:
        """Life for a given stress amplitude, infinite at or below the fatigue limit.

        Args:
            stress_amp: Stress amplitude, in MPa.

        Returns:
            Life, in cycles (``math.inf`` at or below the fatigue limit).
        """
        if stress_amp <= self.fatigue_limit:
            return math.inf
        return float(
            self.n_cycles_at_limit * (self.fatigue_limit / stress_amp) ** self.exponent
        )

    def at_survival_prob(self, survival_prob: float) -> "SNCurve":
        """Return the same curve shifted to another survival probability.

        Assumes a log-normal scatter of the stress amplitude with standard
        deviation `stdev_log_stress` and the same slope.

        Args:
            survival_prob: Target survival probability, strictly between 0
                and 1 (e.g. 0.975 for a design curve).

        Returns:
            A new `SNCurve` (test points are not copied).

        Raises:
            ValueError: If `stdev_log_stress` is not defined, or (from
                `statistics.NormalDist`) if `survival_prob` is outside (0, 1).
        """
        if self.stdev_log_stress is None:
            raise ValueError(f"S-N curve '{self.description}' has no stdev_log_stress")
        z = NormalDist().inv_cdf
        shift = z(1.0 - survival_prob) - z(1.0 - self.survival_prob)
        return SNCurve(
            description=f"{self.description} (P_s = {survival_prob:g})",
            fatigue_limit=self.fatigue_limit * 10 ** (shift * self.stdev_log_stress),
            exponent=self.exponent,
            n_cycles_at_limit=self.n_cycles_at_limit,
            load_mode=self.load_mode,
            stress_ratio=self.stress_ratio,
            survival_prob=survival_prob,
            stdev_log_stress=self.stdev_log_stress,
        )

    def to_dict(self) -> dict[str, Any]:
        """All defined fields as plain Python values (arrays become lists)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if value is not None:
                out[f.name] = value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SNCurve":
        """Inverse of `to_dict`."""
        data = dict(data)
        for name, dtype in (
            ("stress_amp", float),
            ("n_cycles", float),
            ("runout", bool),
        ):
            if data.get(name) is not None:
                data[name] = np.asarray(data[name], dtype=dtype)
        return cls(**data)


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """Nomenclature entry of a material parameter (one row of the WG6 file).

    Attributes:
        key: Attribute name used on `Material`.
        name: Human-readable parameter name.
        symbol: Mathematical symbol (e.g. ``σ_y``).
        unit: Physical unit the value must be given in (e.g. ``MPa``).
        group: Section of the nomenclature the parameter belongs to.
        wg6_key: Spelling used in the WG6 Excel file, if different from
            ``key``.
        proposed: ``True`` if ``key`` is not yet part of the Excel file.
    """

    key: str
    name: str
    symbol: str
    unit: str
    group: str
    wg6_key: str | None = None
    proposed: bool = False


def _param(
    name: str,
    symbol: str,
    unit: str,
    group: str,
    wg6_key: str | None = None,
    proposed: bool = False,
) -> Any:
    """Optional `Material` field carrying its nomenclature entry as metadata."""
    spec = dict(
        name=name,
        symbol=symbol,
        unit=unit,
        group=group,
        wg6_key=wg6_key,
        proposed=proposed,
    )
    return field(default=None, metadata={"spec": spec})


@dataclass
class Material:
    """Material data passed as input to FatPy fatigue methods.

    Fill in only what you know; everything else stays ``None``. Each
    parameter's name, symbol and unit are listed in `NOMENCLATURE` (or
    with `Material.describe`).
    """

    material_name: str
    source: str | None = None  # reference of the data (paper, DOI, test report)
    # physical / elastic
    density: float | None = _param("Density", "ρ", "kg/m³", "physical")
    elastic_modulus: float | None = _param("Young's modulus", "E", "MPa", "elastic")
    elastic_modulus_shear: float | None = _param(
        "Elastic modulus in torsion", "G", "MPa", "elastic"
    )
    poissons_ratio: float | None = _param("Poisson's ratio", "ν", "-", "elastic")
    poissons_ratio_pl: float | None = _param(
        "Poisson's ratio of plasticized material", "ν_pl", "-", "elastic"
    )
    poissons_ratio_eff: float | None = _param(
        "Effective Poisson's ratio", "ν_eff", "-", "elastic"
    )
    thermal_expansion_coef: float | None = _param(
        "Thermal expansion coefficient", "α", "1/°C", "physical"
    )
    # static
    yield_strength: float | None = _param("Yield strength", "σ_y", "MPa", "static")
    yield_strain: float | None = _param("Yield strain", "ε_y", "-", "static")
    ult_tensile_strength: float | None = _param(
        "Ultimate tensile strength", "σ_UTS", "MPa", "static"
    )
    fract_toughness: float | None = _param(
        "Fracture toughness", "K_IC", "MPa√m", "static"
    )
    thresh_sif_range: float | None = _param(
        "Range of the threshold stress intensity factor",
        "ΔK_th",
        "MPa√m",
        "static",
        wg6_key="D_K_TH",
        proposed=True,
    )
    hardness: float | None = _param("Hardness", "HV/HB/HRC", "-", "static")
    # fatigue limits (amplitudes)
    fat_lim_ten_m1: float | None = _param(
        "Fatigue limit in fully reversed push-pull",
        "FL_ten,-1",
        "MPa",
        "fatigue limit",
        wg6_key="fat_lim_ten_-1",
    )
    fat_lim_ten_0: float | None = _param(
        "Fatigue limit in repeated tension", "FL_ten,0", "MPa", "fatigue limit"
    )
    fat_lim_ben_m1: float | None = _param(
        "Fatigue limit in reversed bending",
        "FL_ben,-1",
        "MPa",
        "fatigue limit",
        wg6_key="fat_lim_ben_-1",
    )
    fat_lim_ben_0: float | None = _param(
        "Fatigue limit in repeated bending", "FL_ben,0", "MPa", "fatigue limit"
    )
    fat_lim_rben_m1: float | None = _param(
        "Fatigue limit in rotating bending",
        "RBEND",
        "MPa",
        "fatigue limit",
        wg6_key="RBEND",
        proposed=True,
    )
    fat_lim_tor_m1: float | None = _param(
        "Fatigue limit in fully reversed torsion",
        "FL_tor,-1",
        "MPa",
        "fatigue limit",
        wg6_key="fat_lim_tor_-1",
    )
    fat_lim_tor_0: float | None = _param(
        "Fatigue limit in repeated torsion", "FL_tor,0", "MPa", "fatigue limit"
    )
    fat_lim_tor_A_m1: float | None = _param(
        "Fatigue limit in reversed torsion, mode A",
        "FL_tor,A,-1",
        "MPa",
        "fatigue limit",
        wg6_key="fat_lim_tor_A_-1",
    )
    fat_lim_tor_B_m1: float | None = _param(
        "Fatigue limit in reversed torsion, mode B",
        "FL_tor,B,-1",
        "MPa",
        "fatigue limit",
        wg6_key="fat_lim_tor_B-1",
    )
    # cyclic / strain-life
    fat_strength_coef: float | None = _param(
        "Fatigue strength coefficient", "σ_f'", "MPa", "cyclic"
    )
    fat_strength_exp: float | None = _param(
        "Fatigue strength exponent", "b", "-", "cyclic"
    )
    fat_ductility_coef: float | None = _param(
        "Fatigue ductility coefficient", "ε_f'", "-", "cyclic"
    )
    fat_ductility_exp: float | None = _param(
        "Fatigue ductility exponent", "c", "-", "cyclic"
    )
    fat_strength_coef_tor: float | None = _param(
        "Fatigue strength coefficient in torsion", "τ_f'", "MPa", "cyclic"
    )
    fat_strength_exp_tor: float | None = _param(
        "Fatigue strength exponent in torsion", "b_t", "-", "cyclic"
    )
    fat_ductility_coef_tor: float | None = _param(
        "Fatigue ductility coefficient in torsion", "γ_f'", "-", "cyclic"
    )
    fat_ductility_exp_tor: float | None = _param(
        "Fatigue ductility exponent in torsion", "c_t", "-", "cyclic"
    )
    cyc_hard_coef: float | None = _param(
        "Cyclic hardening coefficient", "K'", "MPa", "cyclic", wg6_key="cyc_hard_coeff"
    )
    cyc_hard_exp: float | None = _param(
        "Cyclic hardening exponent", "n'", "-", "cyclic", wg6_key="cyc_hard_ext"
    )
    eps_ac: float | None = _param("Non-damaging plastic strain", "ε_ac", "-", "cyclic")
    # coefficients of specific methods
    neuber_coef: float | None = _param(
        "Neuber's coefficient", "neuber_coef", "-", "method"
    )
    bergmann_coef: float | None = _param(
        "Bergmann's coefficient",
        "bergmann_coef",
        "-",
        "method",
        wg6_key="bergmann_coeff",
    )
    walker_exp: float | None = _param("Walker's exponent", "walker_exp", "-", "method")
    socie_coef_ten: float | None = _param(
        "Tensile coefficient in Socie's combined model", "socie_coef_ten", "-", "method"
    )
    socie_coef_tor: float | None = _param(
        "Shear coefficient in Socie's combined model", "socie_coef_tor", "-", "method"
    )
    WB_coef: float | None = _param(
        "Wang's and Brown's coefficient", "S_WB", "-", "method"
    )
    socie_findley_coef: float | None = _param(
        "Coefficient in Socie's proposal of Findley",
        "C_FIN",
        "-",
        "method",
        wg6_key="C_FIN",
        proposed=True,
    )
    char_length: float | None = _param(
        "Material characteristic length",
        "L_CHAR",
        "mm",
        "method",
        wg6_key="L_CHAR",
        proposed=True,
    )
    # microstructure
    grain_size: float | None = _param("Grain size", "d", "µm", "microstructure")
    precipitate_size: float | None = _param(
        "Precipitate size", "d_precip", "nm", "microstructure"
    )
    precipitate_vol_frac: float | None = _param(
        "Precipitate volume fraction", "V_precip", "%", "microstructure"
    )
    inclusion_size: float | None = _param(
        "Inclusion size", "d_incl", "µm", "microstructure"
    )
    inclusion_density: float | None = _param(
        "Inclusion density", "ρ_incl", "1/mm³", "microstructure"
    )
    disloc_density: float | None = _param(
        "Dislocation density", "ρ_dislocation", "m⁻²", "microstructure"
    )
    stacking_fault_energy: float | None = _param(
        "Stacking fault energy", "SFE", "mJ/m²", "microstructure"
    )
    res_stress_micro: float | None = _param(
        "Residual stresses (micro-scale)", "σ_res_micro", "MPa", "microstructure"
    )
    # S-N curves and parameters outside the nomenclature
    sn_curves: list[SNCurve] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Compact representation showing only the defined parameters."""
        items = [f"material_name={self.material_name!r}"]
        items += [
            f"{k}={getattr(self, k)!r}"
            for k in NOMENCLATURE
            if getattr(self, k) is not None
        ]
        if self.sn_curves:
            items.append(f"sn_curves=<{len(self.sn_curves)}>")
        if self.extra:
            items.append(f"extra={self.extra!r}")
        return f"Material({', '.join(items)})"

    @staticmethod
    def describe(key: str) -> ParamSpec:
        """Name, symbol and unit of a parameter.

        Args:
            key: Attribute name or WG6 spelling.

        Returns:
            The `ParamSpec` describing `key`.

        Raises:
            KeyError: If `key` is not in the nomenclature.
        """
        return NOMENCLATURE[_ALIASES.get(key, key)]

    def sn_curve(
        self, load_mode: LoadMode, stress_ratio: float = -1.0
    ) -> SNCurve | None:
        """First S-N curve matching a load mode and stress ratio.

        Args:
            load_mode: One of `LoadMode`.
            stress_ratio: Load ratio R = σ_min / σ_max.

        Returns:
            The matching `SNCurve`, or ``None`` if there isn't one.
        """
        for curve in self.sn_curves:
            if curve.load_mode == load_mode and curve.stress_ratio == stress_ratio:
                return curve
        return None

    # -- dictionaries / files -------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Material":
        """Create a material from a dict with WG6 keys.

        ``None`` values are skipped and keys that are not in the nomenclature
        are kept in `extra`.

        Args:
            data: Mapping of WG6 keys (or attribute names) to values, e.g.
                one row of a materials table. Must contain ``material_name``.

        Returns:
            The resulting `Material`.
        """
        known = {f.name for f in fields(cls)} - {"extra"}
        kwargs: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for raw_key, value in data.items():
            if value is None:
                continue
            key = _ALIASES.get(raw_key, raw_key)
            if key == "sn_curves":
                value = [
                    c if isinstance(c, SNCurve) else SNCurve.from_dict(c) for c in value
                ]
            if key in known:
                kwargs[key] = value
            else:
                extra[raw_key] = value
        return cls(**kwargs, extra=extra)

    def to_dict(self, include_sn_curves: bool = False) -> dict[str, Any]:
        """All defined parameters with their WG6 keys.

        Args:
            include_sn_curves: Also export the S-N curves.

        Returns:
            Mapping from WG6 key to value.
        """
        out: dict[str, Any] = {"material_name": self.material_name}
        if self.source:
            out["source"] = self.source
        for key, spec in NOMENCLATURE.items():
            value = getattr(self, key)
            if value is not None:
                out[spec.wg6_key if spec.wg6_key and not spec.proposed else key] = value
        out.update(self.extra)
        if include_sn_curves and self.sn_curves:
            out["sn_curves"] = [c.to_dict() for c in self.sn_curves]
        return out

    def to_json(self, path: str | Path) -> None:
        """Save the material, S-N curves included, to a JSON file."""
        text = json.dumps(
            self.to_dict(include_sn_curves=True), indent=2, ensure_ascii=False
        )
        Path(path).write_text(text, encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Material":
        """Load a material saved with `to_json`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        """Readable table of the defined parameters, grouped as in the nomenclature."""
        header = f"Material: {self.material_name}"
        if self.source:
            header += f"  ({self.source})"
        lines = [header]
        group = None
        for key, spec in NOMENCLATURE.items():
            value = getattr(self, key)
            if value is None:
                continue
            if spec.group != group:
                group = spec.group
                lines.append(f"  [{group}]")
            lines.append(
                f"    {spec.name:<45} {spec.symbol:<12} = {value:g} {spec.unit}"
            )
        for c in self.sn_curves:
            lines.append(
                f"  [S-N curve] {c.description}: {c.load_mode}, R={c.stress_ratio:g}, "
                f"FL={c.fatigue_limit:g}, w={c.exponent:g}, N_k={c.n_cycles_at_limit:g}"
            )
        return "\n".join(lines)


#: Every material parameter, keyed by attribute name (built from the fields).
NOMENCLATURE: dict[str, ParamSpec] = {
    f.name: ParamSpec(key=f.name, **f.metadata["spec"])
    for f in fields(Material)
    if "spec" in f.metadata
}
_ALIASES: dict[str, str] = {
    s.wg6_key: s.key for s in NOMENCLATURE.values() if s.wg6_key is not None
}

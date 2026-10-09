r"""Finite element model parsing module.

Reading of FE results (nodal solution) exported to the FatPy CSV format.
FatPy does not depend on the FE software: export the results of Abaqus,
ANSYS, ... to this format. Element and integration point results will follow.

The results of a file are returned as `NodalResults`: stress and strain
histories in Voigt form, shape ``(n_nodes, n_steps, 6)``, with the node ids
and coordinates, ready for the multiaxial criteria (``(n_steps, 6)`` per node
with `NodalResults.at_node`). Results of several files are scaled and summed
by `fatpy.data_parsing.loads.superpose`.

Unit system:
    FatPy works in N and mm: stresses in MPa, strains in mm/mm, coordinates
    in mm, time in s (or load steps). Values must be given in these units;
    nothing is converted.

FatPy CSV format:
    One row per node per time step, comma separated, with this exact header
    line and column order:

    | Columns                          | Content                       | Unit  |
    |----------------------------------|-------------------------------|-------|
    | ``node_id``                      | node number (integer)         | -     |
    | ``x, y, z``                      | node coordinates              | mm    |
    | ``time``                         | physical time or step number  | s, -  |
    | ``EE11, EE22, EE33, EE23, EE13, EE12`` | elastic strain          | mm/mm |
    | ``PE11, PE22, PE33, PE23, PE13, PE12`` | plastic strain          | mm/mm |
    | ``S11, S22, S33, S23, S13, S12`` | stress                        | MPa   |

    - Coordinates and tensors are given in the same reference system (e.g.
      the global coordinate system of the FE model). Coordinates are the
      undeformed ones and must be the same at every time step.
    - Tensor components are in the Voigt order (11, 22, 33, 23, 13, 12).
    - Shear strains EE23, EE13, EE12, PE23, PE13, PE12 are **tensor** strains
      ε_ij = γ_ij / 2, as in `fatpy.utils.voigt` and
      `fatpy.struct_mech.strain`. Abaqus (EE12, PE12) and ANSYS (EPEL, EPPL)
      export **engineering** shear strains γ_ij: divide them by 2 when
      writing the CSV file. Shear stresses are not affected.
    - Every node has one row at every time step, and (node_id, time) pairs
      are unique. Rows can be in any order.
    - Example (node 12 at time 1, uniaxial 100 MPa, E = 200 000 MPa,
      ν = 0.3, no plasticity)::

        node_id,x,y,z,time,EE11,EE22,EE33,EE23,EE13,EE12,PE11,PE22,PE33,PE23,PE13,PE12,S11,S22,S33,S23,S13,S12
        12,10.0,0.0,5.0,1,5e-4,-1.5e-4,-1.5e-4,0,0,0,0,0,0,0,0,0,100.0,0,0,0,0,0

Checks:
    `read_fe_csv` names the file, line, column or node of every problem:
    missing, extra, repeated or out-of-order columns, non-numeric, NaN or
    inf values, non-integer node ids, duplicate or missing (node_id, time)
    rows, moving coordinates. For a linear-elastic file (e.g. a unit load
    result, ``linear_elastic=True``) every plastic strain must satisfy
    |PE| <= `PLASTIC_STRAIN_TOLERANCE` (1e-12).

Future work:
    - Element and integration point results in the CSV format.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fatpy.utils import voigt
from fatpy.utils.signal import CsvFormatError, read_numeric_csv

#: Columns of the FatPy FE result CSV format, in this order.
FE_CSV_COLUMNS: tuple[str, ...] = (
    "node_id",
    "x",
    "y",
    "z",
    "time",
    *(f"EE{s}" for s in ("11", "22", "33", "23", "13", "12")),
    *(f"PE{s}" for s in ("11", "22", "33", "23", "13", "12")),
    *(f"S{s}" for s in ("11", "22", "33", "23", "13", "12")),
)
#: Largest |plastic strain| [mm/mm] accepted in a linear-elastic result file.
PLASTIC_STRAIN_TOLERANCE = 1e-12
#: Largest difference [mm] between coordinates of the same node.
COORDINATE_TOLERANCE = 1e-6
_EE = slice(5, 11)
_PE = slice(11, 17)
_S = slice(17, 23)


class FEResultError(ValueError):
    """Raised when an FE result file is invalid or inconsistent.

    The message names the file, and the line, column or node when known.
    """


class PlasticStrainError(FEResultError):
    """Raised when a file declared linear-elastic contains plastic strain."""


def _read_only(values: ArrayLike) -> NDArray[np.float64]:
    """Read-only float64 copy of `values`."""
    array = np.array(values, dtype=np.float64)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class NodeHistory:
    """Results of one node, ready for the multiaxial criteria.

    Attributes:
        node_id: Node number.
        coordinates: Node coordinates (x, y, z) [mm], shape (3,).
        time: Time instants [s] or load steps, shape (m,).
        elastic_strain: Elastic strain, Voigt, shape (m, 6) [mm/mm].
        plastic_strain: Plastic strain, Voigt, shape (m, 6) [mm/mm].
        stress: Stress, Voigt, shape (m, 6) [MPa].
    """

    node_id: int
    coordinates: NDArray[np.float64]
    time: NDArray[np.float64]
    elastic_strain: NDArray[np.float64]
    plastic_strain: NDArray[np.float64]
    stress: NDArray[np.float64]

    @property
    def total_strain(self) -> NDArray[np.float64]:
        """Elastic plus plastic strain, Voigt, shape (m, 6) [mm/mm]."""
        return self.elastic_strain + self.plastic_strain


@dataclass(frozen=True, eq=False)
class NodalResults:
    """Stress and strain histories of FE nodes, in Voigt form.

    Nodes are sorted by increasing `node_ids`, time steps by increasing
    `time`. All arrays are read-only float64 (int64 for `node_ids`).

    Attributes:
        node_ids: Node numbers, unique, shape (n,).
        coordinates: Node coordinates (x, y, z) [mm], shape (n, 3).
        time: Time instants [s] or load steps, shape (m,).
        elastic_strain: Elastic strain, shape (n, m, 6) [mm/mm].
        plastic_strain: Plastic strain, shape (n, m, 6) [mm/mm].
        stress: Stress, shape (n, m, 6) [MPa].
        source: Description of where the results come from (file names).
    """

    node_ids: NDArray[np.int64]
    coordinates: NDArray[np.float64]
    time: NDArray[np.float64]
    elastic_strain: NDArray[np.float64]
    plastic_strain: NDArray[np.float64]
    stress: NDArray[np.float64]
    source: str = ""
    _index: dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Store read-only arrays and check their shapes.

        Raises:
            FEResultError: If the shapes are inconsistent or node ids repeat.
        """
        node_ids = np.array(self.node_ids, dtype=np.int64)
        node_ids.setflags(write=False)
        object.__setattr__(self, "node_ids", node_ids)
        for name in ("coordinates", "time", "elastic_strain", "plastic_strain"):
            object.__setattr__(self, name, _read_only(getattr(self, name)))
        object.__setattr__(self, "stress", _read_only(self.stress))
        n, m = self.node_ids.size, self.time.size
        expected = {
            "node_ids": (n,),
            "coordinates": (n, 3),
            "time": (m,),
            "elastic_strain": (n, m, voigt.VOIGT_COMPONENTS_COUNT),
            "plastic_strain": (n, m, voigt.VOIGT_COMPONENTS_COUNT),
            "stress": (n, m, voigt.VOIGT_COMPONENTS_COUNT),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise FEResultError(
                    f"{self.source}: {name} has shape {getattr(self, name).shape}, "
                    f"expected {shape} for {n} nodes and {m} time steps"
                )
        index = {int(node): i for i, node in enumerate(self.node_ids)}
        if len(index) != n:
            raise FEResultError(f"{self.source}: node ids are not unique")
        object.__setattr__(self, "_index", index)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return int(self.node_ids.size)

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(self.time.size)

    @property
    def total_strain(self) -> NDArray[np.float64]:
        """Elastic plus plastic strain, shape (n, m, 6) [mm/mm]."""
        return self.elastic_strain + self.plastic_strain

    def index_of(self, node_id: int) -> int:
        """Position of a node in the arrays.

        Raises:
            KeyError: If there is no such node.
        """
        try:
            return self._index[int(node_id)]
        except KeyError:
            raise KeyError(f"No node {node_id} in {self.source}") from None

    def at_node(self, node_id: int) -> NodeHistory:
        """Results of one node, with (m, 6) Voigt histories.

        Raises:
            KeyError: If there is no such node.
        """
        i = self.index_of(node_id)
        return NodeHistory(
            int(node_id),
            self.coordinates[i],
            self.time,
            self.elastic_strain[i],
            self.plastic_strain[i],
            self.stress[i],
        )


def _check_fe_header(path: str | Path) -> None:
    """Raise `FEResultError` unless the header is exactly `FE_CSV_COLUMNS`."""
    with open(path, encoding="utf-8-sig") as file:
        header = [c.strip() for c in file.readline().strip().split(",")]
    missing = [c for c in FE_CSV_COLUMNS if c not in header]
    extra = [c for c in header if c not in FE_CSV_COLUMNS]
    repeated = sorted({c for c in header if header.count(c) > 1})
    problems = []
    if missing:
        problems.append(f"missing column(s) {missing}")
    if extra:
        problems.append(f"unexpected column(s) {extra}")
    if repeated:
        problems.append(f"repeated column(s) {repeated}")
    if not problems and tuple(header) != FE_CSV_COLUMNS:
        i = next(
            i
            for i, (a, b) in enumerate(zip(header, FE_CSV_COLUMNS, strict=True))
            if a != b
        )
        problems.append(
            f"wrong column order: column {i + 1} is {header[i]!r}, expected "
            f"{FE_CSV_COLUMNS[i]!r}"
        )
    if problems:
        raise FEResultError(
            f"{path}, line 1: {'; '.join(problems)}. The header must be: "
            f"{','.join(FE_CSV_COLUMNS)}"
        )


@dataclass(frozen=True)
class _NodeFilter:
    """Keep a CSV line if its first value is one of `ids`."""

    ids: frozenset[float]

    def __call__(self, text: str) -> bool:
        """``True`` for a wanted node (and for unreadable lines, to report them)."""
        try:
            return float(text.split(",", 1)[0]) in self.ids
        except ValueError:
            return True


def _node_filter(node_ids: Sequence[int]) -> _NodeFilter:
    """Line filter keeping the rows of the given nodes."""
    return _NodeFilter(frozenset(float(n) for n in node_ids))


def read_fe_csv(
    path: str | Path,
    node_ids: Sequence[int] | None = None,
    linear_elastic: bool = False,
) -> NodalResults:
    """Read an FE result file in the FatPy CSV format (nodal solution).

    See the module docstring for the format. Values are read as float64 in
    chunks; with `node_ids` only the rows of these nodes are converted and
    checked, which saves memory for large files.

    Args:
        path: CSV file.
        node_ids: Nodes to load, all if ``None``.
        linear_elastic: ``True`` if the results come from a linear-elastic
            analysis (e.g. a unit load case): every plastic strain must then
            satisfy |PE| <= `PLASTIC_STRAIN_TOLERANCE`.

    Returns:
        The `NodalResults`, nodes and time steps sorted.

    Raises:
        FEResultError: If a column is missing, extra, repeated or out of
            order; if a value is non-numeric, NaN or inf; if a node id is not
            an integer; if a (node_id, time) row is repeated or missing; if
            the coordinates of a node change; if a requested node is not in
            the file. The message names the file, line, column or node.
        PlasticStrainError: If `linear_elastic` and a plastic strain exceeds
            the tolerance.
    """
    _check_fe_header(path)
    keep = None if node_ids is None else _node_filter(node_ids)
    try:
        table = read_numeric_csv(path, keep)
        table.check_finite()
    except CsvFormatError as err:
        raise FEResultError(str(err)) from err
    data, lines = table.data, table.lines
    if data.shape[0] == 0:
        raise FEResultError(
            f"{path}: no data row"
            + ("" if node_ids is None else f" for the node ids {list(node_ids)}")
        )

    nodes = data[:, 0]
    not_integer = np.flatnonzero(nodes != np.round(nodes))
    if not_integer.size:
        row = not_integer[0]
        raise FEResultError(
            f"{path}, line {lines[row]}, column 'node_id': node ids must be "
            f"integers, got {nodes[row]!r}"
        )
    if node_ids is not None:
        absent = sorted(set(int(n) for n in node_ids) - set(nodes.astype(np.int64)))
        if absent:
            raise FEResultError(f"{path}: node ids {absent} are not in the file")

    unique_nodes, node_index = np.unique(nodes, return_inverse=True)
    unique_times, time_index = np.unique(data[:, 4], return_inverse=True)
    order = np.lexsort((time_index, node_index))
    pairs = node_index[order] * unique_times.size + time_index[order]
    repeated = np.flatnonzero(np.diff(pairs) == 0)
    if repeated.size:
        first, second = order[repeated[0]], order[repeated[0] + 1]
        raise FEResultError(
            f"{path}, lines {min(lines[first], lines[second])} and "
            f"{max(lines[first], lines[second])}: duplicate row for node "
            f"{int(nodes[first])} at time {data[first, 4]:g}"
        )
    n_nodes, n_steps = unique_nodes.size, unique_times.size
    if data.shape[0] != n_nodes * n_steps:
        counts = np.bincount(node_index, minlength=n_nodes)
        incomplete = int(np.argmax(counts < n_steps))
        present = set(time_index[node_index == incomplete].tolist())
        missing_step = next(t for t in range(n_steps) if t not in present)
        raise FEResultError(
            f"{path}: node {int(unique_nodes[incomplete])} has no row at time "
            f"{unique_times[missing_step]:g} (every node needs one row at each "
            f"of the {n_steps} time steps)"
        )

    grid = data[order].reshape(n_nodes, n_steps, len(FE_CSV_COLUMNS))
    grid_lines = lines[order].reshape(n_nodes, n_steps)
    coordinates = grid[:, 0, 1:4]
    moved = np.abs(grid[:, :, 1:4] - coordinates[:, np.newaxis, :]).max(axis=2)
    if np.any(moved > COORDINATE_TOLERANCE):
        node, step = (int(v) for v in np.unravel_index(np.argmax(moved), moved.shape))
        raise FEResultError(
            f"{path}, line {grid_lines[node, step]}: coordinates of node "
            f"{int(unique_nodes[node])} differ from line {grid_lines[node, 0]} by "
            f"{moved[node, step]:g} mm; give the undeformed coordinates at every "
            "time step"
        )

    plastic = grid[:, :, _PE]
    if linear_elastic:
        over = np.abs(plastic) > PLASTIC_STRAIN_TOLERANCE
        if np.any(over):
            where = np.argwhere(over)
            first = int(np.argmin(grid_lines[where[:, 0], where[:, 1]]))
            i, step, component = (int(v) for v in where[first])
            raise PlasticStrainError(
                f"{path}, line {grid_lines[i, step]}: plastic strain "
                f"{FE_CSV_COLUMNS[_PE][component]} = "
                f"{plastic[i, step, component]:.3e} at node "
                f"{int(unique_nodes[i])}, time {unique_times[step]:g} (|PE| > "
                f"{PLASTIC_STRAIN_TOLERANCE:g}): is this simulation really "
                "linear elastic?"
            )

    return NodalResults(
        node_ids=unique_nodes.astype(np.int64),
        coordinates=coordinates,
        time=unique_times,
        elastic_strain=grid[:, :, _EE],
        plastic_strain=plastic,
        stress=grid[:, :, _S],
        source=str(path),
    )


def check_same_nodes(reference: NodalResults, other: NodalResults) -> None:
    """Check that two results share the same nodes and coordinates.

    Args:
        reference: Results the other one is compared with.
        other: Results to check.

    Raises:
        FEResultError: If the node ids differ, or if the coordinates of a
            node differ by more than `COORDINATE_TOLERANCE`.
    """
    if not np.array_equal(reference.node_ids, other.node_ids):
        ref, oth = set(reference.node_ids.tolist()), set(other.node_ids.tolist())
        raise FEResultError(
            f"{other.source} and {reference.source} have different nodes: "
            f"only in {other.source}: {sorted(oth - ref)[:10]}, only in "
            f"{reference.source}: {sorted(ref - oth)[:10]}"
        )
    distance = np.abs(other.coordinates - reference.coordinates).max(axis=1)
    if np.any(distance > COORDINATE_TOLERANCE):
        i = int(np.argmax(distance))
        raise FEResultError(
            f"{other.source} and {reference.source}: coordinates of node "
            f"{int(reference.node_ids[i])} differ by {distance[i]:g} mm; the "
            "files must use the same mesh and reference system"
        )

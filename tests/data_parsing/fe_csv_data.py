"""Synthetic FE result files in the FatPy CSV format, shared by the tests."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fatpy.data_parsing.fe_model import FE_CSV_COLUMNS

NODES = (3, 1, 2)
TIMES = (0.0, 1.0, 2.0)


def coords(node: float) -> list[float]:
    """Coordinates [mm] of a synthetic node."""
    return [10.0 * node, float(node), 0.5]


def fe_table(
    nodes: tuple[int, ...] = NODES,
    times: tuple[float, ...] = TIMES,
    stress: tuple[float, ...] | None = None,
) -> NDArray[np.float64]:
    """Rows of a synthetic FE result file, in the file order.

    Stress = node * 100 + step * 10 + [0, 1, ..., 5] MPa unless `stress`
    (multiplied by the node id) is given; elastic strain = stress * 1e-5;
    plastic strain = 0.
    """
    rows = []
    for node in nodes:
        for step, time in enumerate(times):
            if stress is None:
                s = node * 100.0 + step * 10.0 + np.arange(6.0)
            else:
                s = node * np.array(stress)
            rows.append([node, *coords(node), time, *(s * 1e-5), *np.zeros(6), *s])
    return np.array(rows)


def fe_lines(table: NDArray[np.float64]) -> list[str]:
    """CSV lines (header first) of a table."""
    lines = [",".join(FE_CSV_COLUMNS)]
    for row in table:
        lines.append(",".join([str(int(row[0]))] + [repr(float(v)) for v in row[1:]]))
    return lines


def write_lines(path: Path, lines: list[str]) -> Path:
    """Write CSV lines to `path`."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_fe(path: Path, table: NDArray[np.float64]) -> Path:
    """Write a synthetic FE result file."""
    return write_lines(path, fe_lines(table))


def set_cell(lines: list[str], line: int, column: str, text: str) -> list[str]:
    """Replace one value; `line` is the 1-based line number in the file."""
    fields = lines[line - 1].split(",")
    fields[FE_CSV_COLUMNS.index(column)] = text
    lines[line - 1] = ",".join(fields)
    return lines

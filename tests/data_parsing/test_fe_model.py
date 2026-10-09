"""Test functions for the FE model parsing module.

Covers the FatPy CSV reader (shapes, sorting, node subsets, shear strain
convention), every format error (header, values, duplicates, missing rows,
node ids, coordinates), the linear-elastic plastic strain check, the node
comparison used by superposition and the module documentation.
"""

from pathlib import Path

import numpy as np
import pytest
from fe_csv_data import (  # tests/data_parsing/fe_csv_data.py
    NODES,
    TIMES,
    coords,
    fe_lines,
    fe_table,
    set_cell,
    write_fe,
    write_lines,
)

from fatpy.data_parsing import fe_model
from fatpy.data_parsing.fe_model import (
    FE_CSV_COLUMNS,
    PLASTIC_STRAIN_TOLERANCE,
    FEResultError,
    NodalResults,
    PlasticStrainError,
    check_same_nodes,
    read_fe_csv,
)
from fatpy.utils import voigt

RTOL = 1e-9


def test_read_fe_csv_shapes_sorting_and_values(tmp_path: Path) -> None:
    """Rows in any order give (n_nodes, n_steps, 6) arrays sorted by node."""
    table = fe_table()
    shuffled = table[np.random.default_rng(0).permutation(len(table))]
    results = read_fe_csv(write_fe(tmp_path / "fe.csv", shuffled))
    np.testing.assert_array_equal(results.node_ids, [1, 2, 3])
    np.testing.assert_array_equal(results.time, TIMES)
    assert (results.n_nodes, results.n_steps) == (3, 3)
    assert results.stress.shape == (3, 3, 6)
    for i, node in enumerate((1, 2, 3)):
        np.testing.assert_array_equal(results.coordinates[i], coords(node))
        for step in range(3):
            expected = node * 100.0 + step * 10.0 + np.arange(6.0)
            np.testing.assert_allclose(results.stress[i, step], expected, rtol=RTOL)
            np.testing.assert_allclose(
                results.elastic_strain[i, step], expected * 1e-5, rtol=RTOL
            )
    np.testing.assert_array_equal(results.plastic_strain, 0.0)
    np.testing.assert_array_equal(results.total_strain, results.elastic_strain)
    assert results.source.endswith("fe.csv")
    for array in (results.stress, results.coordinates, results.node_ids):
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0


def test_one_node_history(tmp_path: Path) -> None:
    """at_node gives (n_steps, 6) Voigt arrays ready for the criteria."""
    results = read_fe_csv(write_fe(tmp_path / "fe.csv", fe_table()))
    node = results.at_node(2)
    assert node.node_id == 2
    assert node.stress.shape == (3, 6)
    np.testing.assert_array_equal(node.coordinates, coords(2))
    np.testing.assert_array_equal(node.stress, results.stress[1])
    np.testing.assert_array_equal(node.total_strain, node.elastic_strain)
    np.testing.assert_array_equal(node.plastic_strain, 0.0)
    np.testing.assert_array_equal(node.time, TIMES)
    voigt.check_shape(node.stress)
    with pytest.raises(KeyError, match="No node 9"):
        results.at_node(9)


def test_shear_strains_are_read_as_tensor_components(tmp_path: Path) -> None:
    """EE12 is ε_12 (tensor): it is stored unchanged and is the tensor term."""
    table = fe_table(nodes=(1,), times=(0.0,))
    table[0, FE_CSV_COLUMNS.index("EE12")] = 3e-4
    results = read_fe_csv(write_fe(tmp_path / "fe.csv", table))
    assert results.elastic_strain[0, 0, 5] == 3e-4
    tensor = voigt.voigt_to_tensor(results.elastic_strain[0, 0])
    assert tensor[0, 1] == tensor[1, 0] == 3e-4


def test_read_fe_csv_node_subset(tmp_path: Path) -> None:
    """Only the requested nodes are loaded."""
    path = write_fe(tmp_path / "fe.csv", fe_table())
    results = read_fe_csv(path, node_ids=[3, 1])
    np.testing.assert_array_equal(results.node_ids, [1, 3])
    np.testing.assert_allclose(results.stress[1, 0, 0], 300.0)


def test_node_subset_ignores_bad_rows_of_other_nodes(tmp_path: Path) -> None:
    """NaN in rows of nodes that are not loaded is not reported."""
    lines = set_cell(fe_lines(fe_table()), 2, "S11", "nan")  # node 3
    results = read_fe_csv(write_lines(tmp_path / "fe.csv", lines), node_ids=[1])
    np.testing.assert_array_equal(results.node_ids, [1])


def test_node_subset_missing_node_raises(tmp_path: Path) -> None:
    """A requested node that is not in the file is named."""
    path = write_fe(tmp_path / "fe.csv", fe_table())
    with pytest.raises(FEResultError, match=r"fe\.csv: node ids \[9\] are not"):
        read_fe_csv(path, node_ids=[1, 9])
    with pytest.raises(FEResultError, match=r"no data row for the node ids \[8\]"):
        read_fe_csv(path, node_ids=[8])


def test_header_only_raises(tmp_path: Path) -> None:
    """A file without data rows."""
    path = write_lines(tmp_path / "fe.csv", [",".join(FE_CSV_COLUMNS)])
    with pytest.raises(FEResultError, match=r"fe\.csv: no data row"):
        read_fe_csv(path)


def header_with(change: str) -> list[str]:
    """FE header with one modification."""
    header = list(FE_CSV_COLUMNS)
    if change == "missing":
        header.remove("PE12")
    elif change == "extra":
        header.append("extra")
    elif change == "order":
        header[8], header[9] = header[9], header[8]
    elif change == "repeated":
        header[header.index("PE12")] = "PE11"
    return header


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", r"missing column\(s\) \['PE12'\]"),
        ("extra", r"unexpected column\(s\) \['extra'\]"),
        ("order", r"wrong column order: column 9 is 'EE13', expected 'EE23'"),
        (
            "repeated",
            r"missing column\(s\) \['PE12'\]; repeated column\(s\) \['PE11'\]",
        ),
    ],
)
def test_header_errors(tmp_path: Path, change: str, message: str) -> None:
    """Missing, extra, repeated or out-of-order columns name the problem."""
    lines = fe_lines(fe_table())
    lines[0] = ",".join(header_with(change))
    path = write_lines(tmp_path / "fe.csv", lines)
    with pytest.raises(FEResultError, match=r"fe\.csv, line 1: " + message) as info:
        read_fe_csv(path)
    assert ",".join(FE_CSV_COLUMNS) in str(info.value)


def test_header_with_spaces_and_bom_is_accepted(tmp_path: Path) -> None:
    """Spaces around the names and a UTF-8 byte order mark are accepted."""
    lines = fe_lines(fe_table())
    lines[0] = "﻿" + ", ".join(FE_CSV_COLUMNS)
    assert read_fe_csv(write_lines(tmp_path / "fe.csv", lines)).n_nodes == 3


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("abc", r"line 3, column 'S22': non-numeric value 'abc'"),
        ("nan", r"line 3, column 'S22': value is nan"),
        ("", r"line 3, column 'S22': non-numeric value ''"),
        ("inf", r"line 3, column 'S22': value is inf"),
    ],
)
def test_bad_value_names_file_line_and_column(
    tmp_path: Path, text: str, message: str
) -> None:
    """Non-numeric, empty, NaN and inf values are located."""
    lines = set_cell(fe_lines(fe_table()), 3, "S22", text)
    with pytest.raises(FEResultError, match=r"fe\.csv, " + message):
        read_fe_csv(write_lines(tmp_path / "fe.csv", lines))


def test_wrong_number_of_values_raises(tmp_path: Path) -> None:
    """A row with a missing or an extra value."""
    lines = fe_lines(fe_table())
    lines[3] = lines[3].rsplit(",", 1)[0]
    with pytest.raises(FEResultError, match=r"line 4: 22 values for the 23 columns"):
        read_fe_csv(write_lines(tmp_path / "fe.csv", lines))


def test_duplicate_node_time_raises(tmp_path: Path) -> None:
    """A repeated (node_id, time) pair names both lines."""
    lines = fe_lines(fe_table())
    lines.append(lines[1])  # node 3, time 0 again, line 11
    with pytest.raises(
        FEResultError, match=r"lines 2 and 11: duplicate row for node 3 at time 0"
    ):
        read_fe_csv(write_lines(tmp_path / "fe.csv", lines))


def test_missing_row_raises(tmp_path: Path) -> None:
    """Every node needs a row at every time step."""
    lines = fe_lines(fe_table())
    del lines[5]  # node 1, time 1
    with pytest.raises(FEResultError, match=r"node 1 has no row at time 1 "):
        read_fe_csv(write_lines(tmp_path / "fe.csv", lines))


def test_non_integer_node_id_raises(tmp_path: Path) -> None:
    """node_id must be an integer."""
    lines = set_cell(fe_lines(fe_table()), 4, "node_id", "3.5")
    with pytest.raises(
        FEResultError, match=r"line 4, column 'node_id': node ids must be integers"
    ):
        read_fe_csv(write_lines(tmp_path / "fe.csv", lines))


def test_moving_coordinates_raise(tmp_path: Path) -> None:
    """The coordinates of a node must be the same at every time step."""
    lines = set_cell(fe_lines(fe_table()), 4, "x", "30.1")  # node 3, time 2
    with pytest.raises(
        FEResultError, match=r"line 4: coordinates of node 3 differ from line 2 by 0.1"
    ):
        read_fe_csv(write_lines(tmp_path / "fe.csv", lines))


def test_linear_elastic_check_tolerance(tmp_path: Path) -> None:
    """|PE| <= 1e-12 is accepted as zero in a linear-elastic file."""
    lines = set_cell(fe_lines(fe_table()), 5, "PE11", "-1e-12")
    lines = set_cell(lines, 6, "PE23", "5e-13")
    results = read_fe_csv(write_lines(tmp_path / "fe.csv", lines), linear_elastic=True)
    assert np.abs(results.plastic_strain).max() == PLASTIC_STRAIN_TOLERANCE


def test_linear_elastic_check_reports_plastic_strain(tmp_path: Path) -> None:
    """Plastic strain in a linear-elastic file names node, step and line."""
    lines = set_cell(fe_lines(fe_table()), 7, "PE12", "2e-5")  # node 1, time 2
    lines = set_cell(lines, 9, "PE11", "3e-5")
    path = write_lines(tmp_path / "fe.csv", lines)
    with pytest.raises(
        PlasticStrainError,
        match=r"fe\.csv, line 7: plastic strain PE12 = 2\.000e-05 at node 1, "
        r"time 2 \(\|PE\| > 1e-12\): is this simulation really linear elastic\?",
    ):
        read_fe_csv(path, linear_elastic=True)
    plastic = read_fe_csv(path)  # elastic-plastic results are accepted
    assert plastic.plastic_strain[0, 2, 5] == 2e-5
    np.testing.assert_allclose(
        plastic.total_strain[0, 2, 5], plastic.elastic_strain[0, 2, 5] + 2e-5
    )


def test_fe_errors_are_value_errors() -> None:
    """Every FE error can be caught as a FEResultError / ValueError."""
    assert issubclass(PlasticStrainError, FEResultError)
    assert issubclass(FEResultError, ValueError)


def test_nodal_results_validation() -> None:
    """Inconsistent shapes and repeated node ids are refused."""
    zeros = np.zeros((2, 3, 6))
    with pytest.raises(FEResultError, match=r"stress has shape \(2, 2, 6\)"):
        NodalResults(
            np.array([1, 2]), np.zeros((2, 3)), np.zeros(3), zeros, zeros, zeros[:, :2]
        )
    with pytest.raises(FEResultError, match="not unique"):
        NodalResults(
            np.array([1, 1]), np.zeros((2, 3)), np.zeros(3), zeros, zeros, zeros
        )


def test_check_same_nodes(tmp_path: Path) -> None:
    """Same mesh passes; other nodes or moved coordinates are named."""
    reference = read_fe_csv(write_fe(tmp_path / "a.csv", fe_table()))
    check_same_nodes(reference, read_fe_csv(write_fe(tmp_path / "b.csv", fe_table())))
    other = read_fe_csv(write_fe(tmp_path / "c.csv", fe_table(nodes=(1, 2, 4))))
    with pytest.raises(FEResultError, match=r"only in .*c\.csv: \[4\]"):
        check_same_nodes(reference, other)
    lines = fe_lines(fe_table())
    for line in (2, 3, 4):  # node 3 at its three time steps
        lines = set_cell(lines, line, "y", "3.5")
    moved = read_fe_csv(write_lines(tmp_path / "d.csv", lines))
    with pytest.raises(FEResultError, match=r"coordinates of node 3 differ by 0\.5"):
        check_same_nodes(reference, moved)


def test_module_docstring_documents_the_format() -> None:
    """The module documents units, Voigt order, shear convention, columns."""
    doc = fe_model.__doc__ or ""
    for item in ("N and mm", "MPa", "Voigt order", "tensor", "engineering"):
        assert item in doc
    assert ",".join(FE_CSV_COLUMNS) in doc
    assert "Element and integration point" in doc
    assert NODES and TIMES  # shared synthetic data is importable

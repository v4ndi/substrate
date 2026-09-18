"""Report exports preserve names, tabular content and plot data independently of tasks."""

from xml.etree import ElementTree
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from fmlib.automl.reporting import (
    ConfusionData,
    CurveData,
    EvaluationData,
    export_evaluation,
)
from fmlib.automl.types import EvaluationResult


def _workbook_rows(path):
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        shared = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = [
            "".join(item.itertext()) for item in shared.findall("s:si", namespace)
        ]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall("s:sheetData/s:row", namespace):
        cells = {}
        for cell in row.findall("s:c", namespace):
            value = cell.find("s:v", namespace)
            if value is not None:
                cells[cell.attrib["r"].rstrip("0123456789")] = (
                    strings[int(value.text)] if cell.get("t") == "s" else value.text
                )
        rows.append(cells)
    headers = rows[0]
    return [
        {headers[column]: value for column, value in row.items()} for row in rows[1:]
    ]


@pytest.mark.parametrize("task_name", ["binary", "regression", "multiclass", "uplift"])
@pytest.mark.parametrize("kind", ["raw", "calibrated"])
def test_export_preserves_workbooks_figures_and_aliases(tmp_path, task_name, kind):
    metrics = {
        "binary": {"roc_auc": 0.75},
        "regression": {"mse": 1.0, "mae": 0.5, "mape": 0.1},
        "multiclass": {
            "roc_auc_ovr_macro": 0.75,
            "accuracy": 0.6,
            "f1_macro": 0.55,
            "f1_weighted": 0.58,
        },
        "uplift": {"s_qini_auc": 0.2},
    }[task_name]
    grouped = pl.DataFrame({
        "scope": ["date_group"],
        "report_month": ["2026-01-01"],
        "group": ["a"],
        **{name: [value] for name, value in metrics.items()},
    })
    public_grouped = grouped.rename({"report_month": "as_of", "group": "channel"})
    by_class = (
        pl.DataFrame({"class_index": [0], "class_label": ["a"], "roc_auc": [0.75]})
        if task_name == "multiclass"
        else None
    )
    importance = pl.DataFrame({"feature": ["x", "y"], "importance": [0.8, 0.2]})
    result = EvaluationResult.for_kind(
        kind,
        metrics=metrics,
        metrics_by_group=public_grouped,
        metrics_by_class=by_class,
    )
    confusions = (
        {"multiclass_confusion_matrix": ConfusionData(np.eye(3), ("a", "b", "c"))}
        if task_name == "multiclass"
        else {}
    )
    curve = CurveData("qini", "", {"s": (np.array([0.0, 1.0]), np.array([0.0, 0.2]))})
    curves = {"qini_curves": curve} if task_name == "uplift" else {}
    data = EvaluationData(
        result=result,
        task_name=task_name,
        n_samples=10,
        grouped=grouped,
        date_column="report_month",
        date_label="as_of",
        group_column="group",
        feature_importance=importance,
        confusions=confusions,
        curves=curves,
    )
    open_figures = set(plt.get_fignums())
    exported = export_evaluation(data, tmp_path)
    assert set(plt.get_fignums()) == open_figures
    assert (
        getattr(result, f"excel_paths_{kind}")
        == getattr(result, f"figures_{kind}")
        == {}
    )
    assert getattr(exported, f"metrics_{kind}") == metrics
    assert getattr(exported, f"metrics_by_group_{kind}").equals(public_grouped)
    keys = {f"metrics_{kind}", f"feature_importance_{kind}"}
    if task_name == "multiclass":
        keys.add(f"metrics_by_class_{kind}")
    excel_paths = getattr(exported, f"excel_paths_{kind}")
    figures = getattr(exported, f"figures_{kind}")
    assert set(excel_paths) == keys
    for key, path in excel_paths.items():
        assert path == tmp_path / f"{key}.xlsx"
        assert _workbook_rows(path)
    rows = _workbook_rows(excel_paths[f"metrics_{kind}"])
    assert rows[0]["scope"] == "overall"
    assert rows[1]["scope"] == "date_group"
    assert rows[1]["as_of"] == "2026-01-01" and rows[1]["channel"] == "a"
    for name, value in metrics.items():
        assert float(rows[0][name]) == pytest.approx(value)
    assert [
        row["feature"]
        for row in _workbook_rows(excel_paths[f"feature_importance_{kind}"])
    ] == ["x", "y"]
    trend = {
        "binary": "roc_auc_by_date",
        "regression": "regression_metrics_by_date",
        "multiclass": "multiclass_metrics_by_date",
    }
    expected_figures = (
        {f"{trend[task_name]}_{kind}"}
        if task_name in trend
        else {f"qini_curves_{kind}"}
    )
    if task_name == "multiclass":
        expected_figures.add(f"multiclass_confusion_matrix_{kind}")
    assert set(figures) == expected_figures
    for key, figure in figures.items():
        assert (tmp_path / f"{key}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        axis = figure.axes[0]
        if key.removesuffix(f"_{kind}") in trend.values():
            assert axis.get_xlabel() == "as_of"
            np.testing.assert_allclose(
                axis.lines[0].get_ydata(), [next(iter(metrics.values()))]
            )
    if task_name == "uplift":
        np.testing.assert_allclose(
            figures[f"qini_curves_{kind}"].axes[0].lines[0].get_ydata(), [0.0, 0.2]
        )
    if task_name == "multiclass":
        np.testing.assert_allclose(
            figures[f"multiclass_confusion_matrix_{kind}"]
            .axes[0]
            .images[0]
            .get_array(),
            np.eye(3),
        )

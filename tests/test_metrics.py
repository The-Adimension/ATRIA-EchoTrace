"""Clinical metrics, with emphasis on the calibration contract.

The central guarantee: physical units are never reported unless pixel spacing is
actually known (RESEARCH.md §3.2).
"""

from __future__ import annotations

import pytest

from atria_echotrace.domain.metrics import (
    Calibration,
    chamber_metrics,
    fractional_area_change,
    frame_metrics,
)

# A square covering the middle half of the frame, in normalised [y, x].
SQUARE = [[250, 250], [250, 750], [750, 750], [750, 250]]
SMALLER = [[375, 375], [375, 625], [625, 625], [625, 375]]


# ---------------------------------------------------------------- calibration
def test_unknown_calibration_is_not_known() -> None:
    assert Calibration.unknown().is_known is False


def test_dataset_calibration_is_known() -> None:
    assert Calibration(0.308, 0.308, "dataset").is_known is True


@pytest.mark.parametrize(
    "calibration",
    [
        Calibration(None, 0.3, "dataset"),
        Calibration(0.3, None, "dataset"),
        Calibration(0.0, 0.3, "dataset"),
        Calibration(0.3, -1.0, "dataset"),
        Calibration(0.3, 0.3, "unknown"),
    ],
)
def test_incomplete_calibration_is_not_known(calibration: Calibration) -> None:
    assert calibration.is_known is False


def test_user_calibration_rejects_non_positive_spacing() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        Calibration.from_user(0.0, 0.3)
    with pytest.raises(ValueError, match="must be > 0"):
        Calibration.from_user(0.3, -0.1)


# -------------------------------------------------------------- frame metrics
def test_frame_metrics_area_in_pixels() -> None:
    result = frame_metrics(SQUARE, image_h=100, image_w=100)
    assert result.vertices == 4
    assert result.area_px == pytest.approx(2500.0)  # 50x50
    assert result.perimeter_px == pytest.approx(200.0)


def test_physical_area_withheld_when_uncalibrated() -> None:
    result = frame_metrics(SQUARE, 100, 100, Calibration.unknown())
    assert result.area_cm2 is None
    assert result.perimeter_cm is None
    assert result.calibration_source == "unknown"


def test_physical_area_reported_when_calibrated() -> None:
    # 2500 px² at 1 mm/px = 2500 mm² = 25 cm²
    result = frame_metrics(SQUARE, 100, 100, Calibration(1.0, 1.0, "dataset"))
    assert result.area_cm2 == pytest.approx(25.0)
    # 200 px perimeter at 1 mm/px = 200 mm = 20 cm
    assert result.perimeter_cm == pytest.approx(20.0)


def test_physical_area_uses_product_of_spacings() -> None:
    """Anisotropic pixels scale area by spacing_h * spacing_w exactly."""
    result = frame_metrics(SQUARE, 100, 100, Calibration(0.5, 2.0, "dataset"))
    assert result.area_cm2 == pytest.approx(2500 * 0.5 * 2.0 / 100)


def test_anisotropic_perimeter_is_remeasured_not_scaled() -> None:
    """A single scalar cannot scale a perimeter under anisotropy.

    The square is 50 px on each side; at 0.5 mm/px vertically and 2.0 horizontally the
    physical shape is 25 mm x 100 mm, so the perimeter is 250 mm = 25 cm — not the
    12.5 cm or 50 cm a naive single-factor scaling would give.
    """
    result = frame_metrics(SQUARE, 100, 100, Calibration(0.5, 2.0, "dataset"))
    assert result.perimeter_cm == pytest.approx(25.0)


def test_degenerate_polygon_yields_zeroes_without_raising() -> None:
    result = frame_metrics([], 100, 100, Calibration(0.3, 0.3, "dataset"))
    assert result.vertices == 0
    assert result.area_px == 0.0
    # No polygon means no measurement, not a zero-area measurement in cm².
    assert result.area_cm2 is None


# ------------------------------------------------------------------------ FAC
def test_fac_of_typical_contraction() -> None:
    assert fractional_area_change(100.0, 40.0) == pytest.approx(60.0)


def test_fac_of_no_change_is_zero() -> None:
    assert fractional_area_change(100.0, 100.0) == pytest.approx(0.0)


def test_fac_is_negative_when_es_exceeds_ed() -> None:
    """Reported as computed rather than clamped; the UI flags the anomaly."""
    assert fractional_area_change(100.0, 150.0) == pytest.approx(-50.0)


@pytest.mark.parametrize("ed_area", [0.0, -5.0])
def test_fac_is_none_without_a_valid_ed_area(ed_area: float) -> None:
    """None, not 0.0 — zero would read as 'no contraction' rather than 'unknown'."""
    assert fractional_area_change(ed_area, 10.0) is None


def test_chamber_metrics_pairs_ed_and_es() -> None:
    result = chamber_metrics(SQUARE, SMALLER, 100, 100, Calibration(1.0, 1.0, "dataset"))
    assert result.ed.area_px == pytest.approx(2500.0)
    assert result.es.area_px == pytest.approx(625.0)
    assert result.fac_percent == pytest.approx(75.0)
    assert result.ed.area_cm2 == pytest.approx(25.0)


def test_chamber_metrics_fac_is_valid_without_calibration() -> None:
    """FAC is dimensionless, so it survives an uncalibrated frame."""
    result = chamber_metrics(SQUARE, SMALLER, 100, 100, Calibration.unknown())
    assert result.fac_percent == pytest.approx(75.0)
    assert result.ed.area_cm2 is None


@pytest.mark.parametrize(
    ("ed", "es"),
    [
        (SQUARE, []),          # ES not traced yet
        ([], SMALLER),         # ED not traced yet
        (SQUARE, [[1, 1], [2, 2]]),  # ES has too few vertices to bound an area
    ],
)
def test_fac_is_undefined_until_both_phases_are_traced(ed, es) -> None:
    """A half-finished study must not report a confident FAC.

    With an untraced ES the area is 0, so the ratio evaluates to a clean-looking
    100% — arithmetically true and clinically meaningless.
    """
    assert chamber_metrics(ed, es, 100, 100).fac_percent is None


def test_fac_appears_once_both_phases_are_traced() -> None:
    assert chamber_metrics(SQUARE, SMALLER, 100, 100).fac_percent == pytest.approx(75.0)


def test_untraced_frame_is_distinguishable_from_uncalibrated_one() -> None:
    """The UI needs to tell "no trace" apart from "no pixel spacing"."""
    calibrated = Calibration(1.0, 1.0, "dataset")
    untraced = frame_metrics([], 100, 100, calibrated)
    assert untraced.vertices == 0
    assert untraced.area_cm2 is None
    # ...but the calibration itself is still known, which is what the UI keys on.
    assert untraced.calibration_source == "dataset"

    uncalibrated = frame_metrics(SQUARE, 100, 100, Calibration.unknown())
    assert uncalibrated.vertices == 4
    assert uncalibrated.area_cm2 is None
    assert uncalibrated.calibration_source == "unknown"


def test_chamber_metrics_supports_differing_es_dimensions() -> None:
    result = chamber_metrics(SQUARE, SQUARE, 100, 100, es_image_h=200, es_image_w=200)
    assert result.ed.area_px == pytest.approx(2500.0)
    assert result.es.area_px == pytest.approx(10000.0)


def test_as_dict_serialises_none_physical_units() -> None:
    payload = chamber_metrics(SQUARE, SMALLER, 100, 100, Calibration.unknown()).as_dict()
    assert payload["ed"]["area_cm2"] is None
    assert payload["calibration_source"] == "unknown"
    assert payload["fac_percent"] == pytest.approx(75.0)


# ------------------------------------------------- real dataset, real numbers
def test_real_camus_case_produces_physiological_fac(camus_case) -> None:
    """A calibrated CAMUS case gives a positive FAC and real cm² areas."""
    result = chamber_metrics(
        camus_case.ed.lv_polygon,
        camus_case.es.lv_polygon,
        camus_case.ed.image_h,
        camus_case.ed.image_w,
        camus_case.calibration,
    )
    assert result.ed.area_cm2 is not None
    assert result.es.area_cm2 is not None
    assert result.fac_percent is not None and result.fac_percent > 0
    # An adult LV end-diastolic area in an apical view is a few tens of cm².
    assert 5.0 < result.ed.area_cm2 < 120.0
    assert result.es.area_cm2 < result.ed.area_cm2


def test_real_echonet_case_withholds_physical_area(echonet_case) -> None:
    result = chamber_metrics(
        echonet_case.ed.lv_polygon,
        echonet_case.es.lv_polygon,
        echonet_case.ed.image_h,
        echonet_case.ed.image_w,
        echonet_case.calibration,
    )
    assert echonet_case.calibration.is_known is False
    assert result.ed.area_cm2 is None
    assert result.es.area_cm2 is None
    assert result.ed.area_px > 0
    assert result.fac_percent is not None


def test_every_sample_case_has_positive_areas(dataset_repo) -> None:
    """All 50 real frames rasterise to a strictly positive area."""
    for case in dataset_repo.cases.values():
        for instant, frame in case.frames.items():
            result = frame_metrics(
                frame.lv_polygon, frame.image_h, frame.image_w, frame.calibration
            )
            assert result.area_px > 0, f"{frame.stem} ({instant}) has zero LV area"
            assert result.perimeter_px > 0

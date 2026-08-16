"""Geometry: area, conversion, parsing, rasterisation and overlap metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atria_echotrace.domain.geometry import (
    compute_dice,
    compute_iou,
    denormalize_polygon,
    normalize_polygon,
    parse_polygon,
    polygon_dice,
    polygon_perimeter,
    polygon_to_mask,
    sanitize_polygon,
    shoelace_area,
)


# ------------------------------------------------------------------ area
def test_shoelace_area_of_unit_square() -> None:
    assert shoelace_area([[0, 0], [0, 10], [10, 10], [10, 0]]) == pytest.approx(100.0)


def test_shoelace_area_of_right_triangle() -> None:
    assert shoelace_area([[0, 0], [4, 0], [0, 3]]) == pytest.approx(6.0)


def test_shoelace_area_is_winding_independent() -> None:
    clockwise = [[0, 0], [0, 10], [10, 10], [10, 0]]
    counter = list(reversed(clockwise))
    assert shoelace_area(clockwise) == pytest.approx(shoelace_area(counter))


def test_shoelace_area_is_axis_order_independent() -> None:
    """Swapping the coordinate pair negates the signed area, which abs() removes.

    This is what lets one implementation serve both [y, x] and [x, y] polygons.
    """
    as_xy = [[0, 0], [4, 0], [4, 3], [0, 3]]
    as_yx = [[y, x] for x, y in as_xy]
    assert shoelace_area(as_xy) == pytest.approx(shoelace_area(as_yx))


def test_shoelace_area_ignores_duplicated_closing_vertex() -> None:
    """CAMUS polygons repeat the first vertex last; it must contribute nothing."""
    open_polygon = [[0, 0], [0, 10], [10, 10], [10, 0]]
    closed_polygon = open_polygon + [[0, 0]]
    assert shoelace_area(closed_polygon) == pytest.approx(shoelace_area(open_polygon))


def test_shoelace_area_closes_open_contours_implicitly() -> None:
    """EchoNet polygons are not closed; wrap-around must still yield the full area."""
    assert shoelace_area([[0, 0], [0, 10], [10, 10], [10, 0]]) == pytest.approx(100.0)


def test_shoelace_area_of_concave_polygon() -> None:
    # An L shape: 3x3 square minus a 2x2 corner.
    l_shape = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]
    assert shoelace_area(l_shape) == pytest.approx(5.0)


@pytest.mark.parametrize(
    "degenerate",
    [[], [[1, 1]], [[0, 0], [1, 1]], None],
)
def test_shoelace_area_of_degenerate_input_is_zero(degenerate) -> None:
    assert shoelace_area(degenerate) == 0.0


def test_shoelace_area_of_collinear_points_is_zero() -> None:
    assert shoelace_area([[0, 0], [1, 1], [2, 2], [3, 3]]) == pytest.approx(0.0)


def test_perimeter_of_square_is_closed() -> None:
    assert polygon_perimeter([[0, 0], [0, 10], [10, 10], [10, 0]]) == pytest.approx(40.0)


# ------------------------------------------------------- coordinate conversion
def test_denormalize_maps_yx_to_xy_pixels() -> None:
    # y=500 of 200px -> 100; x=250 of 400px -> 100
    assert denormalize_polygon([[500, 250]], image_h=200, image_w=400) == [[100.0, 100.0]]


def test_normalize_maps_xy_pixels_to_yx() -> None:
    assert normalize_polygon([[100.0, 100.0]], image_h=200, image_w=400) == [[500, 250]]


def test_normalize_denormalize_round_trip_within_one_unit() -> None:
    """The UI depends on this round trip; quantisation must stay sub-unit."""
    original = [[123, 456], [789, 12], [0, 1000], [1000, 0], [500, 500]]
    pixels = denormalize_polygon(original, image_h=552, image_w=669)
    restored = normalize_polygon(pixels, image_h=552, image_w=669)
    for before, after in zip(original, restored):
        assert abs(before[0] - after[0]) <= 1
        assert abs(before[1] - after[1]) <= 1


def test_normalize_clamps_out_of_bounds_pixels() -> None:
    result = normalize_polygon([[-50.0, -50.0], [9999.0, 9999.0]], image_h=100, image_w=100)
    assert result == [[0, 0], [1000, 1000]]


@pytest.mark.parametrize("image_h,image_w", [(0, 100), (100, 0)])
def test_conversion_with_zero_dimensions_returns_empty(image_h: int, image_w: int) -> None:
    assert denormalize_polygon([[1, 1]], image_h, image_w) == []
    assert normalize_polygon([[1, 1]], image_h, image_w) == []


# --------------------------------------------------------------- model parsing
def test_parse_polygon_from_fenced_json() -> None:
    response = (
        'Some preamble.\n\nFinal Answer: ```json[{"polygon_2d": [[10, 20], [30, 40], '
        '[50, 60]], "label": "left_ventricle_endocardium"}]```'
    )
    assert parse_polygon(response, "LV") == [[10, 20], [30, 40], [50, 60]]


def test_parse_polygon_strips_thinking_trace() -> None:
    """MedGemma may emit a trace before <unused95>; the notebook discards it."""
    response = (
        'ignored reasoning with a decoy ```json[{"polygon_2d": [[1, 1], [2, 2], [3, 3]], '
        '"label": "left_ventricle_endocardium"}]``` <unused95> '
        '```json[{"polygon_2d": [[9, 9], [8, 8], [7, 7]], '
        '"label": "left_ventricle_endocardium"}]```'
    )
    assert parse_polygon(response, "LV") == [[9, 9], [8, 8], [7, 7]]


def test_parse_polygon_selects_the_requested_structure() -> None:
    response = (
        '```json[{"polygon_2d": [[1, 1], [2, 2], [3, 3]], "label": "left_atrium"}, '
        '{"polygon_2d": [[4, 4], [5, 5], [6, 6]], "label": "left_ventricle_endocardium"}]```'
    )
    assert parse_polygon(response, "LV") == [[4, 4], [5, 5], [6, 6]]
    assert parse_polygon(response, "LA") == [[1, 1], [2, 2], [3, 3]]


def test_parse_polygon_single_object_fallback_ignores_label() -> None:
    """The notebook accepts a lone object regardless of its label."""
    response = '```json[{"polygon_2d": [[1, 2], [3, 4], [5, 6]], "label": "something_else"}]```'
    assert parse_polygon(response, "LV") == [[1, 2], [3, 4], [5, 6]]


def test_parse_polygon_bare_key_fallback_without_fence() -> None:
    """Robustness rule adopted from the author's Space (RESEARCH.md §0.3)."""
    response = 'Final Answer: {"polygon_2d": [[7, 8], [9, 10], [11, 12]]}'
    assert parse_polygon(response, "LV") == [[7, 8], [9, 10], [11, 12]]


@pytest.mark.parametrize(
    "response",
    [
        "",
        "no json here at all",
        "```json{not valid json}```",
        '```json[{"label": "left_ventricle_endocardium"}]```',
    ],
)
def test_parse_polygon_returns_none_for_unusable_output(response: str) -> None:
    assert parse_polygon(response, "LV") is None


def test_parse_polygon_tolerates_non_dict_entries() -> None:
    """A malformed list must not raise; the notebook's version would have."""
    response = (
        '```json["garbage", {"polygon_2d": [[1, 1], [2, 2], [3, 3]], '
        '"label": "left_ventricle_endocardium"}]```'
    )
    assert parse_polygon(response, "LV") == [[1, 1], [2, 2], [3, 3]]


# ------------------------------------------------------------------ sanitising
def test_sanitize_clamps_and_rounds() -> None:
    assert sanitize_polygon([[-10, 1200.6], [5.4, 5.5], [1, 2]]) == [[0, 1000], [5, 6], [1, 2]]


def test_sanitize_drops_malformed_points() -> None:
    polygon = [[1, 2], ["a", "b"], [3], None, [4, 5], [6, 7]]
    assert sanitize_polygon(polygon) == [[1, 2], [4, 5], [6, 7]]


def test_sanitize_rejects_non_finite_values() -> None:
    assert sanitize_polygon([[float("nan"), 1], [float("inf"), 2], [3, 4]]) is None


@pytest.mark.parametrize("bad", [None, "text", [], [[1, 2]], [[1, 2], [3, 4]]])
def test_sanitize_requires_three_valid_points(bad) -> None:
    assert sanitize_polygon(bad) is None


# ------------------------------------------------------ masks and overlap
def test_polygon_to_mask_fills_expected_area() -> None:
    # A square covering the middle half of a 100x100 frame.
    polygon = [[250, 250], [250, 750], [750, 750], [750, 250]]
    mask = polygon_to_mask(polygon, 100, 100)
    assert mask.shape == (100, 100)
    # 50x50 region, allowing for boundary rasterisation.
    assert 2400 <= mask.sum() <= 2601


def test_polygon_to_mask_of_degenerate_polygon_is_empty() -> None:
    assert polygon_to_mask([], 20, 20).sum() == 0
    assert polygon_to_mask([[1, 1]], 20, 20).sum() == 0


def test_dice_and_iou_of_identical_masks_is_one() -> None:
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:8, 2:8] = 1
    assert compute_dice(mask, mask) == pytest.approx(1.0)
    assert compute_iou(mask, mask) == pytest.approx(1.0)


def test_dice_and_iou_of_disjoint_masks_is_zero() -> None:
    a = np.zeros((10, 10), dtype=np.float32)
    b = np.zeros((10, 10), dtype=np.float32)
    a[0:3, 0:3] = 1
    b[7:10, 7:10] = 1
    assert compute_dice(a, b) == pytest.approx(0.0)
    assert compute_iou(a, b) == pytest.approx(0.0)


def test_dice_of_half_overlap() -> None:
    a = np.zeros((10, 10), dtype=np.float32)
    b = np.zeros((10, 10), dtype=np.float32)
    a[:, 0:6] = 1  # 60 px
    b[:, 4:10] = 1  # 60 px, 20 px shared
    assert compute_dice(a, b) == pytest.approx(2 * 20 / 120)
    assert compute_iou(a, b) == pytest.approx(20 / 100)


def test_dice_of_two_empty_masks_is_one() -> None:
    """Notebook behaviour: no prediction and no reference agree perfectly."""
    empty = np.zeros((5, 5), dtype=np.float32)
    assert compute_dice(empty, empty) == 1.0
    assert compute_iou(empty, empty) == 1.0


def test_polygon_dice_of_identical_polygons_is_one() -> None:
    polygon = [[200, 200], [200, 800], [800, 800], [800, 200]]
    assert polygon_dice(polygon, polygon, 100, 100) == pytest.approx(1.0)

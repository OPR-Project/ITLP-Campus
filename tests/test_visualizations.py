import numpy as np

from itlp_campus.visualizations import (
    STUFF_COLORS,
    colorize_binary_mask,
    get_colored_mask,
)


def test_colorize_binary_mask():
    binary_mask = np.array([[1, 0], [0, 1]])
    color = (255, 0, 0)  # Red color
    expected_output = np.array([[[255, 0, 0], [0, 0, 0]], [[0, 0, 0], [255, 0, 0]]])
    assert np.array_equal(colorize_binary_mask(binary_mask, color), expected_output)


def test_get_colored_mask_mapillary():
    mask = np.array([[0, 1], [1, 0]])
    dataset = "mapillary"
    # Assuming STUFF_COLORS["mapillary"] is defined and has at least 2 colors
    expected_output = np.zeros(mask.shape + (3,), dtype=np.uint8)
    expected_output += colorize_binary_mask((mask == 0), STUFF_COLORS[dataset][0])
    expected_output += colorize_binary_mask((mask == 1), STUFF_COLORS[dataset][1])
    assert np.array_equal(get_colored_mask(mask, dataset), expected_output)


def test_get_colored_mask_ade20k():
    mask = np.array([[0, 1], [1, 0]])
    dataset = "ade20k"
    # Assuming STUFF_COLORS["ade20k"] is defined and has at least 2 colors
    expected_output = np.zeros(mask.shape + (3,), dtype=np.uint8)
    expected_output += colorize_binary_mask((mask == 0), STUFF_COLORS[dataset][0])
    expected_output += colorize_binary_mask((mask == 1), STUFF_COLORS[dataset][1])
    assert np.array_equal(get_colored_mask(mask, dataset), expected_output)

import numpy as np

from itlp_campus.preprocessing import (
    closest_values_indices,
    filter_by_distance_indices,
    filter_timestamps,
    plot_track_map,
)


def test_closest_values_indices():
    in_array = np.array([1, 2, 3])
    from_array = np.array([0, 2, 4])
    expected_output = np.array([0, 1, 1])
    assert np.array_equal(closest_values_indices(in_array, from_array), expected_output)


def test_filter_timestamps():
    pose_ts = np.array([100, 210, 300])
    front_ts = np.array([90, 210, 290])
    back_ts = np.array([110, 190, 310])
    lidar_ts = np.array([95, 205, 305])
    max_diff = 15
    expected_output = (np.array([0, 2]), np.array([0, 2]), np.array([0, 2]), np.array([0, 2]))
    assert np.array_equal(filter_timestamps(pose_ts, front_ts, back_ts, lidar_ts, max_diff), expected_output)


def test_filter_by_distance_indices():
    utm_points = np.array([[0, 0], [3, 4], [6, 8], [9, 12]])
    distance = 5.0
    expected_output = np.array([0, 1, 2, 3])
    assert np.array_equal(filter_by_distance_indices(utm_points, distance), expected_output)


def test_filter_by_distance_indices_large_distance():
    utm_points = np.array([[0, 0], [3, 4], [6, 8], [9, 12]])
    distance = 100.0
    expected_output = np.array([0])
    assert np.array_equal(filter_by_distance_indices(utm_points, distance), expected_output)


def test_filter_by_distance_indices_non_integer_distance():
    utm_points = np.array([[0, 0], [3, 4], [6, 8], [9, 12]])
    distance = 5.5
    expected_output = np.array([0, 1, 2])
    assert np.array_equal(filter_by_distance_indices(utm_points, distance), expected_output)


def test_plot_track_map():
    utms = np.array([[0, 0], [3, 4], [6, 8], [9, 12]])
    img = plot_track_map(utms)
    assert img.shape[2] == 3  # Check if the image is 3-channel (BGR)
    assert img.dtype == np.uint8  # Check if the image pixel values are in the correct range (0-255)

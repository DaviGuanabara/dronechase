import pytest
from geometry_utils import GeometryUtils
import numpy as np


def test_degrees_between_orthogonal_vectors():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    assert GeometryUtils.degrees_between_vectors(v1, v2) == pytest.approx(90)


# ... other tests for degrees_between_vectors ...


def test_point_inside_cone():
    apex = np.array([0, 0, 0])
    base_center = np.array([0, 0, -1])
    point_inside = np.array([0, 0, -0.5])
    assert GeometryUtils.is_point_inside_cone(point_inside, apex, base_center, 60)


def test_point_outside_cone():
    apex = np.array([0, 0, 0])
    base_center = np.array([0, 0, -1])
    point_outside = np.array([1, 1, 0])
    assert not GeometryUtils.is_point_inside_cone(point_outside, apex, base_center, 60)


# ... other tests for is_point_inside_cone ...

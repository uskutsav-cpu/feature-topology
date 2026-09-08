import numpy as np
from src.metrics.quotient import image_graph


def test_polygon_circle_interval_point():
    theta = np.arange(12)*2*np.pi/12
    circle = np.stack((np.cos(theta), np.sin(theta)), 1)
    identity = image_graph(circle, [(np.eye(2), np.zeros(2), False)])
    interval = image_graph(circle, [(np.array([[1., 0.]]), np.zeros(1), False)])
    point = image_graph(circle, [(np.zeros((1, 2)), np.zeros(1), False)])
    assert (identity["beta0"], identity["beta1"]) == (1, 1)
    assert (interval["beta0"], interval["beta1"]) == (1, 0)
    assert (point["beta0"], point["beta1"]) == (1, 0)


def test_relu_breakpoint():
    theta = np.arange(8)*2*np.pi/8
    circle = np.stack((np.cos(theta), np.sin(theta)), 1)
    result = image_graph(circle, [(np.array([[1., 0.]]), np.array([.2]), True)])
    assert result["beta1"] == 0
    assert result["affine_segments"] > 8

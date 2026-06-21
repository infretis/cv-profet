import numpy as np

from CV_frame_data.math_utils import box_lengths, has_pbc, mic_delta, rational_switch


def test_has_pbc_for_supported_box_shapes():
    assert has_pbc(10.0)
    assert has_pbc(np.array([10.0, 11.0, 12.0]))
    assert has_pbc(np.eye(3) * 10.0)


def test_mic_delta_scalar_box_wraps_into_minimum_image():
    delta = np.array([6.2, -6.2, 0.0])
    wrapped = mic_delta(delta, 10.0)
    assert np.allclose(wrapped, np.array([-3.8, 3.8, 0.0]))


def test_box_lengths_for_triclinic_cell():
    box = np.array(
        [
            [4.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    lengths = box_lengths(box)
    assert lengths is not None
    assert np.allclose(lengths, [4.0, np.sqrt(10.0), 5.0])


def test_rational_switch_handles_x_equal_x0_with_lhopital_value():
    x = np.array([1.0])
    out = rational_switch(x, x0=1.0, n=16, m=56)
    assert np.allclose(out, np.array([16.0 / 56.0]))


def test_rational_switch_output_is_clipped_between_zero_and_one():
    x = np.array([0.1, 1.0, 100.0])
    out = rational_switch(x, x0=1.0)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)

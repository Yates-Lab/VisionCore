import numpy as np
import pandas as pd

from paper.fig4.spatiotemporal_tuning.run_real_backimage_power import (
    anti_alias_eye_position,
    folded_segment_power,
    render_crop,
    spatial_bin_contract,
)


def test_eye_position_is_lowpassed_before_240hz_evaluation() -> None:
    source_rate = 540.0
    time = np.arange(0.0, 3.0, 1.0 / source_rate)
    low = np.sin(2.0 * np.pi * 20.0 * time)
    high = 0.5 * np.sin(2.0 * np.pi * 150.0 * time)
    frame = pd.DataFrame(
        {
            "t_ephys": time,
            "dpi_i": 300.0 + low + high,
            "dpi_j": 600.0 + 0.5 * low + high,
            "valid": np.ones(len(time), dtype=bool),
        }
    )

    result = anti_alias_eye_position(
        frame,
        start_ephys=0.6,
        stop_ephys=2.4,
        target_rate_hz=240.0,
        passband_hz=100.0,
        stopband_hz=118.0,
        padding_seconds=0.5,
    )

    target = np.asarray(result["target_time"])
    expected = np.sin(2.0 * np.pi * 20.0 * target)
    filtered = np.asarray(result["filtered_position_px"])[:, 0] - 300.0
    unfiltered = np.asarray(result["unfiltered_position_px"])[:, 0] - 300.0
    filtered_error = np.sqrt(np.mean(np.square(filtered - expected)))
    unfiltered_error = np.sqrt(np.mean(np.square(unfiltered - expected)))
    assert filtered_error < 0.08 * unfiltered_error


def test_subpixel_crop_center_uses_bilinear_image_coordinates() -> None:
    row, column = np.meshgrid(np.arange(101), np.arange(121), indexing="ij")
    image = (row + 2.0 * column).astype(np.float32)
    center = np.asarray([[50.25, 60.5]], dtype=np.float32)

    movie = render_crop(image, center, crop_size=3, device="cpu")

    expected = (50.25 + 2.0 * 60.5) / 255.0
    np.testing.assert_allclose(movie[0, 1, 1], expected, atol=2e-5)


def test_folded_movie_spectrum_recovers_known_sf_tf_peak() -> None:
    frame_rate = 240.0
    ppd = 20.0
    crop_size = 95
    n_time = 480
    target_sf = 2.0
    target_tf = 12.0
    coordinate = (np.arange(crop_size) - 0.5 * (crop_size - 1)) / ppd
    time = np.arange(n_time) / frame_rate
    movie = 1.0 + 0.2 * np.cos(
        2.0
        * np.pi
        * (target_sf * coordinate[None, None, :] - target_tf * time[:, None, None])
    )
    movie = np.broadcast_to(movie, (n_time, crop_size, crop_size)).copy()
    binning = spatial_bin_contract(
        crop_size,
        ppd,
        minimum_cpd=0.5,
        maximum_cpd=8.0,
        n_bins=32,
    )

    temporal, power = folded_segment_power(
        movie.astype(np.float32),
        frame_rate_hz=frame_rate,
        binning=binning,
    )

    tf_index, sf_index = np.unravel_index(
        np.argmax(power[1:]) + power.shape[1], power.shape
    )
    recovered_sf = binning["centers_cpd"][sf_index]
    recovered_tf = temporal[tf_index]
    assert abs(np.log2(recovered_sf / target_sf)) < 0.12
    assert abs(recovered_tf - target_tf) <= frame_rate / n_time

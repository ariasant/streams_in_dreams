"""HDF5 snapshot round-trip, including the Phase A/B change in particle count."""

import numpy as np

from io_utils import SnapshotWriter, read_snapshots


def test_snapshot_roundtrip(tmp_path):
    path = tmp_path / "snaps.h5"
    writer = SnapshotWriter(str(path), config_yaml="run_name: test")

    # Phase A snapshot: host only.
    n_host = 50
    pos_a = np.random.rand(n_host, 3)
    vel_a = np.random.rand(n_host, 3)
    mass_a = np.full(n_host, 0.02)
    flag_a = np.zeros(n_host, dtype=bool)
    writer.write(0, 0.0, pos_a, vel_a, mass_a, flag_a)

    # Phase B snapshot: host + one perturber (different particle count).
    pos_b = np.vstack([pos_a, [[5.0, 0.0, 0.0]]])
    vel_b = np.vstack([vel_a, [[-1.0, 0.0, 0.0]]])
    mass_b = np.append(mass_a, 0.2)
    flag_b = np.append(flag_a, True)
    writer.write(100, 2.0, pos_b, vel_b, mass_b, flag_b)

    assert writer.n_snapshots == 2

    snaps = read_snapshots(str(path))
    assert len(snaps) == 2

    s0, s1 = snaps
    assert s0["step"] == 0 and s0["time"] == 0.0
    assert s0["pos"].shape == (n_host, 3)
    assert not s0["is_perturber"].any()

    assert s1["step"] == 100 and s1["time"] == 2.0
    assert s1["pos"].shape == (n_host + 1, 3)
    assert s1["is_perturber"].sum() == 1
    assert s1["is_perturber"][-1]
    np.testing.assert_allclose(s1["pos"][:n_host], pos_a)

import json

import numpy as np
import pytest

from services.run_checkpoint import (
    FORMAT_VERSION,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)


def _state():
    return {
        "genome_spec_signature": "brain:80",
        "generation": 12,
        "optimizer_name": "CMA-ES",
        "optimizer_state": {"dim": 80, "popsize": 16, "_mean": np.zeros(80)},
        "base_seed": 7,
        "prompt": "glowing coral",
        "distractors": ["a blank image", "random noise"],
        "settings": {"grid": 4, "steps_per_gen": 300, "sigma0": 0.5},
        "history": {"fit_best": [0.1, 0.3], "fit_mean": [0.05, 0.2]},
        "best_z": np.full(80, 0.25, dtype=np.float32),
        "best_fitness": 0.31,
    }


def test_roundtrip_restores_everything(tmp_path):
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    got = load_checkpoint(p)
    assert got["generation"] == 12
    assert got["prompt"] == "glowing coral"
    assert got["optimizer_name"] == "CMA-ES"
    assert got["settings"]["grid"] == 4
    assert got["history"]["fit_best"] == [0.1, 0.3]
    assert got["distractors"] == ["a blank image", "random noise"]
    assert np.allclose(got["best_z"], 0.25)
    assert np.allclose(got["optimizer_state"]["_mean"], 0.0)
    assert got["optimizer_state"]["popsize"] == 16


def test_real_optimizer_state_survives_a_roundtrip(tmp_path):
    """The checkpoint must carry whatever the optimizer actually produces, not
    just the toy dict above."""
    from services.optimizers import make_optimizer

    opt = make_optimizer("CMA-ES", 80, 16, 0.5, 0)
    for _ in range(5):
        z = opt.ask(16)
        opt.tell(z, np.array([-float(np.sum(zi * zi)) for zi in z], np.float32))
    # Snapshot BEFORE the reference ask(): ask() advances the RNG, so capturing
    # it afterwards would leave the checkpoint one batch ahead.
    st = _state()
    st["optimizer_state"] = opt.state_dict()
    expected = opt.ask(16)
    p = tmp_path / "real.npz"
    save_checkpoint(p, st)

    fresh = make_optimizer("CMA-ES", 80, 16, 0.5, 999)
    fresh.load_state_dict(load_checkpoint(p)["optimizer_state"])
    assert np.allclose(fresh.ask(16), expected, atol=1e-6)


def test_format_version_mismatch_is_specific(tmp_path):
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    meta["format_version"] = FORMAT_VERSION + 99
    np.savez(p, meta=np.array(json.dumps(meta)), **arrays)

    with pytest.raises(CheckpointError, match="format_version"):
        load_checkpoint(p)


def test_signature_mismatch_is_specific(tmp_path):
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    with pytest.raises(CheckpointError, match="brain:80"):
        load_checkpoint(p, expect_signature="brain:80,physics:12")


def test_matching_signature_passes(tmp_path):
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    assert load_checkpoint(p, expect_signature="brain:80")["generation"] == 12


def test_write_is_atomic(tmp_path):
    """A crash mid-write must never destroy a good checkpoint."""
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    good = p.read_bytes()

    bad = _state()
    bad["optimizer_state"] = {"boom": object()}   # not serializable
    with pytest.raises(CheckpointError):
        save_checkpoint(p, bad)

    assert p.read_bytes() == good
    assert not (tmp_path / "ck.npz.tmp").exists()


def test_missing_file_is_specific(tmp_path):
    with pytest.raises(CheckpointError, match="no checkpoint"):
        load_checkpoint(tmp_path / "nope.npz")


def test_no_pickled_objects_are_written(tmp_path):
    p = tmp_path / "ck.npz"
    save_checkpoint(p, _state())
    with np.load(p, allow_pickle=False) as z:   # raises if anything was pickled
        assert "meta" in z.files

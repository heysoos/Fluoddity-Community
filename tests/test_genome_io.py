import json

import numpy as np
import pytest

from services.config_saver import ConfigSaver
from services.genome_io import META_KEY, export_genome, import_genome
from services.genome_spec import decode
from state import SimState


@pytest.fixture
def sim_state():
    return SimState()


def test_exported_file_loads_with_the_normal_config_loader(tmp_path, sim_state):
    """The whole point: an evolved genome must open in the normal single-sim
    view at any resolution, with no new load path."""
    z = np.random.default_rng(0).normal(0, 0.5, 80).astype(np.float32)
    p = tmp_path / "creature.json"
    export_genome(p, z, sim_state, {"generation": 42})

    cfg = ConfigSaver().load_from_file(p)
    assert cfg is not None
    assert np.allclose(cfg.rule, decode(z), atol=1e-5)


def test_export_records_provenance_metadata(tmp_path, sim_state):
    z = np.zeros(80, dtype=np.float32)
    p = tmp_path / "c.json"
    export_genome(p, z, sim_state, {
        "generation": 7,
        "evolved_at_canvas_px": 1024,
        "evolved_at_tile_px": 256,
        "evolved_with_tile_mutation": True,
        "mutation_strength": 0.1,
        "variants_per_tile": 4,
    })
    data = json.loads(p.read_text())
    assert data[META_KEY]["generation"] == 7
    assert data[META_KEY]["evolved_with_tile_mutation"] is True
    assert data[META_KEY]["variants_per_tile"] == 4


def test_metadata_does_not_break_the_normal_loader(tmp_path, sim_state):
    """from_dict uses .get(), so unknown top-level keys must be ignored."""
    export_genome(tmp_path / "c.json", np.zeros(80, dtype=np.float32),
                  sim_state, {"generation": 1})
    assert ConfigSaver().load_from_file(tmp_path / "c.json") is not None


def test_import_roundtrips_z(tmp_path, sim_state):
    z = np.random.default_rng(1).normal(0, 0.6, 80).astype(np.float32)
    p = tmp_path / "c.json"
    export_genome(p, z, sim_state, {"generation": 3})
    z2, clamped, meta = import_genome(p)
    assert clamped == 0
    assert np.allclose(z, z2, atol=1e-3)
    assert meta["generation"] == 3


def test_import_of_out_of_range_config_clamps_and_reports(tmp_path, sim_state):
    saver = ConfigSaver()
    rule = np.zeros((10, 8), dtype=np.float32)
    rule[0, 0] = 500.0
    cfg = saver.create_config(sim_state, rule)
    p = tmp_path / "wild.json"
    saver.save_to_file(cfg, p)

    z, clamped, meta = import_genome(p)
    assert clamped >= 1
    assert np.all(np.isfinite(z))
    assert meta == {}


def test_import_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_genome(tmp_path / "nope.json")

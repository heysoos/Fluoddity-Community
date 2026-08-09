"""Resizing the canvas/world reallocates every GPU buffer, which zeroes the
buffer holding the 16 tournament genomes. If they are not re-uploaded, every
particle gets an all-zero brain and the entire grid freezes.
"""
from unittest.mock import MagicMock
import numpy as np

from command_handler import CommandHandler
from services.tournament_service import TournamentService
from state import UIState


def make_handler():
    svc = TournamentService(rng=np.random.default_rng(0))
    svc.init_population()
    svc.clear_dirty()

    sim = MagicMock()
    sim.get_canvas_dimensions.return_value = (1024, 1024)
    sim.entity_count = 600000
    rule_manager = MagicMock()
    rule_manager.has_rules.return_value = False

    handler = CommandHandler(
        sim=sim, camera=MagicMock(), ui=MagicMock(), rule_manager=rule_manager,
        entity_picker=MagicMock(), video_service=MagicMock(),
        config_saver=MagicMock(), multi_load_service=MagicMock(),
        user_configs_dir=MagicMock(), field_handler=None,
        param_lock_service=None, tournament_service=svc,
    )
    return handler, svc, sim


def test_world_size_change_marks_tournament_dirty():
    handler, svc, sim = make_handler()
    assert svc.is_dirty() is False

    ui_state = UIState()
    ui_state.preferences.world_size = 2.0
    ui_state.preferences.canvas_aspect_ratio = "16:9"
    handler._handle_world_size_change(ui_state)

    # The buffer was reallocated, so the genomes MUST be queued for re-upload.
    assert svc.is_dirty() is True, "genomes would be left zeroed -> frozen grid"
    sim.setup_simulation_state.assert_called_once()


def test_handler_without_tournament_service_still_resizes():
    """Resize must not blow up when tournament support is absent."""
    sim = MagicMock()
    sim.get_canvas_dimensions.return_value = (1024, 1024)
    sim.entity_count = 600000
    rule_manager = MagicMock()
    rule_manager.has_rules.return_value = False
    handler = CommandHandler(
        sim=sim, camera=MagicMock(), ui=MagicMock(), rule_manager=rule_manager,
        entity_picker=MagicMock(), video_service=MagicMock(),
        config_saver=MagicMock(), multi_load_service=MagicMock(),
        user_configs_dir=MagicMock(), field_handler=None,
        param_lock_service=None, tournament_service=None,
    )
    ui_state = UIState()
    ui_state.preferences.world_size = 1.0
    handler._handle_world_size_change(ui_state)  # must not raise
    sim.setup_simulation_state.assert_called_once()


def test_shader_has_zero_rule_fallback():
    """Defence in depth: an all-zero tournament genome must not freeze the tile.

    The fallback now spans two files - entity_update.glsl detects the blank
    brain and sets g_brain_fallback, and _dispatch.glsl answers it with a
    per-cohort random rule - so both halves are asserted. Either one alone is a
    frozen grid.
    """
    from pathlib import Path
    shaders = Path(__file__).resolve().parent.parent / "shaders"

    src = (shaders / "entity_update.glsl").read_text()
    i = src.index("bool blank = brain_params[brain_base]")
    block = src[i:i + 400]
    assert "g_brain_fallback = true" in block, "a zeroed genome buffer is not detected"

    dispatch = (shaders / "brains" / "_dispatch.glsl").read_text()
    j = dispatch.index("g_brain_fallback")
    assert "generate_random_centers" in dispatch[j:j + 200], (
        "no fallback rule for a zeroed genome buffer"
    )

"""Cancelling an expedition.

Text goals can drive somewhere useless, and 50 generations at 2000 steps is
about three minutes of watching it happen. The control is a one-shot flag in
the established pattern; what is worth pinning is the CADENCE afterwards.
"""
from state.archive_state import ArchiveState


class _UIState:
    def __init__(self):
        self.archive = ArchiveState()


# _handle_explore copies these from ArchiveState onto the driver every frame,
# so a bare ArchiveState() would overwrite the fixture with the UI defaults -
# seed_n 256 against an archive of 16 puts the driver straight back into
# bootstrap, and the test would be measuring that instead of the cancel.
_PUSHED = ("sigma_expand", "alpha", "k", "seed_n", "liveness_min",
           "refresh_sweep_gens", "expansion_between", "expedition_gens",
           "expedition_sigma", "latent_share", "goal_order")


def ui_matching(driver):
    ui = _UIState()
    for name in _PUSHED:
        setattr(ui.archive, name, getattr(driver, name))
    ui.archive.capacity = 100
    return ui


def driver_in_expedition(**kw):
    """A real ImgepDriver actually inside an expedition."""
    import sys
    sys.path.insert(0, "tests")
    from test_imgep_driver import moving, seeded_wide

    d, arc, ts = seeded_wide(expansion_between=2, expedition_gens=50,
                             latent_share=1.0, **kw)
    for _ in range(3):
        d.tell(d.ask(4), moving(4))
    assert d.regime == "expedition", "fixture must actually be on an expedition"
    return d, arc, ts


class _Handler:
    """CommandHandler with only what _handle_explore touches."""

    def __init__(self, drv, svc, arc):
        from command_handler import CommandHandler

        self.imgep_driver = drv
        self.auto_service = svc
        self.archive = arc
        self.tournament_service = None
        self.goal_list = None
        self.archive_projection = None
        self._clear_explore_flags = CommandHandler._clear_explore_flags


class _Svc:
    def __init__(self, drv):
        self.driver = drv
        self.phase = type("P", (), {"value": "rollout"})()

    def configure(self, **kw):
        pass

    def pause(self):
        pass

    def reset(self):
        pass

    def start(self):
        pass

    def abort_generation(self):
        pass


def run(handler, ui_state):
    from command_handler import CommandHandler
    return CommandHandler._handle_explore(handler, ui_state)


def test_the_flag_ends_the_expedition():
    d, arc, _ = driver_in_expedition()
    ui = ui_matching(d)
    ui.archive.cancel_expedition_requested = True
    run(_Handler(d, _Svc(d), arc), ui)
    assert d.regime != "expedition"
    assert d.goal_label == ""


def test_the_flag_is_cleared_so_it_does_not_cancel_the_next_one():
    """A one-shot that survives the frame would make expeditions impossible."""
    d, arc, _ = driver_in_expedition()
    ui = ui_matching(d)
    ui.archive.cancel_expedition_requested = True
    run(_Handler(d, _Svc(d), arc), ui)
    assert ui.archive.cancel_expedition_requested is False


def test_cancelling_returns_the_search_to_expansion():
    d, arc, _ = driver_in_expedition()
    ui = ui_matching(d)
    ui.archive.cancel_expedition_requested = True
    run(_Handler(d, _Svc(d), arc), ui)
    assert d.regime == "expansion", "the archive is past seed_n, so expansion"


def test_cancelling_buys_a_full_expansion_interval_before_the_next_goal():
    """Cancel means 'not this, and not immediately another'. start_expedition_with
    left _since_expedition at 0, so the cadence has to run again from scratch."""
    import sys
    sys.path.insert(0, "tests")
    from test_imgep_driver import moving

    d, arc, _ = driver_in_expedition()
    ui = ui_matching(d)
    ui.archive.cancel_expedition_requested = True
    run(_Handler(d, _Svc(d), arc), ui)

    d.tell(d.ask(4), moving(4))
    assert d.regime == "expansion", "one generation is not enough"
    d.tell(d.ask(4), moving(4))
    assert d.regime == "expedition", "expansion_between=2, so the second is"


def test_nothing_happens_without_the_flag():
    d, arc, _ = driver_in_expedition()
    goal_before = d.goal_label
    run(_Handler(d, _Svc(d), arc), ui_matching(d))
    assert d.regime == "expedition"
    assert d.goal_label == goal_before


def test_a_chase_in_the_same_frame_survives_the_cancel():
    """The cancel runs first on purpose: a chase IS a request for a new
    expedition and must not be undone by it."""
    d, arc, _ = driver_in_expedition()
    ui = ui_matching(d)
    ui.archive.cancel_expedition_requested = True
    ui.archive.chase_tile = 1
    run(_Handler(d, _Svc(d), arc), ui)
    assert d.regime == "expedition"
    assert d._goal.kind == "chase"

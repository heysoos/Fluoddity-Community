import numpy as np
from services.tournament_service import TournamentService


def test_pack_matches_ssbo_expectations():
    svc = TournamentService(rng=np.random.default_rng(0))
    svc.init_population()
    data = svc.pack_rule_bytes()
    # 16 rules * SIZE_OF_RULE_STRUCT (320 bytes) — matches sim.py constant
    from sim import SIZE_OF_RULE_STRUCT
    assert SIZE_OF_RULE_STRUCT == 320
    assert len(data) == 16 * SIZE_OF_RULE_STRUCT

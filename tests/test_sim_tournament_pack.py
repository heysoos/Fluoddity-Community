import numpy as np
from services.tournament_service import TournamentService


def test_pack_matches_ssbo_expectations():
    svc = TournamentService(rng=np.random.default_rng(0))
    svc.init_population()
    data = svc.pack_rule_bytes()
    # pack_rule_bytes is flat genome floats, one rule's width unpadded.
    assert len(data) == 16 * svc.layout.length * 4

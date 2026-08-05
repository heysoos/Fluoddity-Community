import datetime as dt
import json

import numpy as np

from services.run_logger import RunLogger, new_run_id


def test_run_id_format():
    assert new_run_id(dt.datetime(2026, 8, 4, 14, 30, 22)) == "20260804-143022"


def test_each_generation_is_one_json_line(tmp_path):
    log = RunLogger(root=tmp_path, config={"grid": 4})
    for g in range(3):
        log.log_generation({"gen": g, "fit_best": 0.1 * g})
    log.close()

    lines = (log.dir / "log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    assert [json.loads(x)["gen"] for x in lines] == [0, 1, 2]


def test_config_is_written_at_run_start(tmp_path):
    log = RunLogger(root=tmp_path, config={"grid": 6, "prompt": "coral"})
    data = json.loads((log.dir / "config.json").read_text())
    assert data["grid"] == 6
    assert data["prompt"] == "coral"
    log.close()


def test_lines_are_flushed_so_a_crash_loses_at_most_one(tmp_path):
    log = RunLogger(root=tmp_path)
    log.log_generation({"gen": 0, "fit_best": 0.5})
    # deliberately not closed
    assert (log.dir / "log.jsonl").read_text().strip() != ""


def test_a_new_field_does_not_break_reading_earlier_lines(tmp_path):
    log = RunLogger(root=tmp_path)
    log.log_generation({"gen": 0, "fit_best": 0.1})
    log.log_generation({"gen": 1, "fit_best": 0.2, "novelty": 0.9})
    log.close()
    recs = [json.loads(x) for x in (log.dir / "log.jsonl").read_text().splitlines()]
    assert "novelty" not in recs[0]
    assert recs[1]["novelty"] == 0.9


def test_history_accumulates_for_the_sparkline(tmp_path):
    log = RunLogger(root=tmp_path)
    for g in range(4):
        log.log_generation({"gen": g, "fit_best": g / 10, "fit_mean": g / 20})
    h = log.history()
    assert h["fit_best"] == [0.0, 0.1, 0.2, 0.3]
    assert len(h["fit_mean"]) == 4
    log.close()


def test_load_history_restores_the_sparkline(tmp_path):
    log = RunLogger(root=tmp_path)
    log.load_history({"fit_best": [0.4, 0.5], "fit_mean": [0.2, 0.3]})
    assert log.history()["fit_best"] == [0.4, 0.5]
    log.close()


def test_unwritable_root_degrades_to_a_warning(tmp_path):
    """Evolution must not be blocked by a disk problem."""
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file, not a directory")
    log = RunLogger(root=blocker)
    assert log.enabled is False
    log.log_generation({"gen": 0})   # must not raise
    assert log.history()["fit_best"] == []
    log.close()


def test_start_new_run_clears_the_history(tmp_path):
    """Reset must not leave the previous search's curve on the plot."""
    log = RunLogger(root=tmp_path)
    for g in range(4):
        log.log_generation({"gen": g, "fit_best": g / 10, "sigma": 0.5})
    assert log.history()["fit_best"] != []
    log.start_new_run()
    assert log.history()["fit_best"] == []
    assert log.history()["sigma"] == []
    log.close()


def test_start_new_run_writes_to_a_fresh_file(tmp_path):
    """The old run's generations must stay in the old file, not be appended to."""
    log = RunLogger(root=tmp_path)
    log.log_generation({"gen": 0, "fit_best": 0.9})
    old_dir, old_id = log.dir, log.run_id

    log.start_new_run()
    assert log.run_id != old_id
    log.log_generation({"gen": 0, "fit_best": 0.1})
    log.close()

    assert len((old_dir / "log.jsonl").read_text().strip().splitlines()) == 1
    new_recs = (log.dir / "log.jsonl").read_text().strip().splitlines()
    assert len(new_recs) == 1
    assert json.loads(new_recs[0])["fit_best"] == 0.1


def test_start_new_run_reuses_an_untouched_run(tmp_path):
    """Clicking Reset repeatedly must not litter empty run folders."""
    log = RunLogger(root=tmp_path)
    first = log.run_id
    log.start_new_run()
    log.start_new_run()
    assert log.run_id == first
    assert len(list(tmp_path.iterdir())) == 1
    log.close()


def test_start_new_run_on_a_disabled_logger_is_harmless(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("i am a file")
    log = RunLogger(root=blocker)
    log.start_new_run()          # must not raise
    assert log.history()["fit_best"] == []


def test_save_frame_writes_a_png(tmp_path):
    log = RunLogger(root=tmp_path)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[:, :, 0] = 200
    log.save_frame(img, 10)
    assert (log.dir / "frames" / "gen_000010.png").is_file()
    log.close()

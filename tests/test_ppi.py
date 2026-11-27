"""PPI scopes in the UI (meeting 03, action 07) and the display data behind them."""
import numpy as np

from src.core import geometry as geo
from src.core.env import RadarNetEnv
from src.core.radar import RadarParams
from src.core.scenario import load_scenario
from src.models.tracking.tracker import TrackManager


def _short_env(name="nominal", duration=30.0, local_tracks=True, **stress):
    sc = load_scenario(name)
    sc["duration_s"] = duration
    sc["terrain"] = {"source": "procedural"}                # keep tests offline
    sc["stress"].update(stress)
    env = RadarNetEnv(sc, record=True, local_tracks=local_tracks)
    env.reset(seed=0)
    while env.agents:
        env.step({a: 1 for a in env.agents})
    return env


def test_tracker_records_which_track_each_report_fed():
    radar = RadarParams(position=(0.0, 0.0, 10.0), heading_deg=0.0)
    rng = np.random.default_rng(0)
    tm = TrackManager()
    x = np.array([6000.0, 500.0, 800.0, -30.0, 5.0, 0.0])
    sig = np.array([10.0, 2e-3, 3e-3, 0.5])
    for k in range(1, 6):
        x[:3] += x[3:] * 0.5
        z = geo.measure(x[:3], x[3:], radar.pos, radar.heading) + rng.normal(0.0, sig)
        d = {"radar": 0, "z": z, "measurement_covariance": np.diag(sig ** 2), "signal_quality": 1.0}
        tm.predict(0.5)
        tm.process(0.5 * k, [d], [radar])
        assert tm.last_assoc == {id(z): tm.tracks[0].id}


def test_frames_carry_ppi_data():
    env = _short_env()
    n = env.n_radars
    for fr in env.frames[1:]:
        assert len(fr["own"]) == len(fr["comm"]) == len(fr["ltracks"]) == n
        for i in range(n):
            produced, received = fr["comm"][i]
            assert produced == len(fr["own"][i]) and 0 <= received <= produced
        assert sum(fr["comm"][i][1] for i in range(n)) == len(fr["dets"])     # what reached the centre
        assert all("by" in t for t in fr["tracks"])
        for loc in fr["ltracks"]:
            assert all({"id", "p", "v", "sig", "fid"} <= set(t) for t in loc)


def test_local_tracks_link_to_fused_tracks_fed_by_the_same_reports():
    env = _short_env()
    fused = {tr.id: tr for tr in env.tracker.tracks}
    links = 0
    for i, lt in enumerate(env.local_trackers):
        for tr in lt.confirmed:
            fid = env.link_of(i, tr.id)
            if fid is not None:
                links += 1
                assert i in fused[fid].updated_by          # the radar really fed that fused track
    assert links > 0


def test_comm_loss_shows_up_in_the_ppi_counts():
    env = _short_env(comm_drop_prob=1.0)
    produced = sum(fr["comm"][i][0] for fr in env.frames[1:] for i in range(env.n_radars))
    received = sum(fr["comm"][i][1] for fr in env.frames[1:] for i in range(env.n_radars))
    assert produced > 0 and received == 0
    assert env._links == {}                                  # nothing reached the centre, nothing to link


def test_runs_without_local_trackers_still_render():
    from src.interfaces import plots, viewer3d
    from src.services.simulation_service import viewer_payload
    env = _short_env(duration=5.0, local_tracks=False)
    payload = viewer_payload(env)
    assert payload["frames"][-1]["ltracks"] == []
    fig = plots.ppi(payload, len(payload["frames"]) - 1, 0)
    assert "own 0" in fig.layout.title.text
    assert 'id="ppiRow"' in viewer3d.build_html(payload)


def test_ppi_figure_per_radar_and_viewer_contain_scopes():
    from src.interfaces import plots, viewer3d
    from src.services.simulation_service import viewer_payload
    env = _short_env(duration=10.0)
    payload = viewer_payload(env)
    k = len(payload["frames"]) - 1
    for i in range(env.n_radars):
        fig = plots.ppi(payload, k, i)
        assert fig.layout.title.text.startswith(env.radar_params[i].name)
    html = viewer3d.build_html(payload)
    assert 'id="ppiRow"' in html and "window.PPI" in html and 'id="ppiB"' in html
    assert 'id="ppiBack"' in html and "pairTable" in html            # click-to-enlarge view with the comparison table


class _FixedClassifier:
    """Stand-in classifier: the same probabilities for every track."""
    probs = np.array([0.1, 0.7, 0.2])

    def classify_tracks(self, tracks):
        return {tr.id: self.probs for tr in tracks}


def test_frames_carry_class_probabilities_for_the_popup():
    from src.services.simulation_service import viewer_payload
    sc = load_scenario("nominal")
    sc["duration_s"] = 12.0
    sc["terrain"] = {"source": "procedural"}
    env = RadarNetEnv(sc, classifier=_FixedClassifier(), record=True, local_tracks=True)
    env.reset(seed=0)
    while env.agents:
        env.step({a: 1 for a in env.agents})
    classified = [t for fr in env.frames for t in fr["tracks"] if t["probs"] is not None]
    assert classified, "no track was classified"
    for t in classified:
        assert len(t["probs"]) == 3 and abs(sum(t["probs"]) - 1) < 1e-2 and t["cls"] == int(np.argmax(t["probs"]))
    payload = viewer_payload(env)
    assert payload["classifier"] is True and len(payload["classes"]) == 3


def test_ppi_2d_has_paths_no_coverage_sector_and_class_hover():
    from src.interfaces import plots
    from src.services.simulation_service import viewer_payload
    env = _short_env(duration=30.0)
    payload = viewer_payload(env)
    k = len(payload["frames"]) - 1
    fig = plots.ppi(payload, k, 0)
    fills = [tr for tr in fig.data if tr.fill == "toself"]
    assert not any((tr.fillcolor or "").startswith("rgba(79,209,197") for tr in fills)   # no field-of-view wedge
    paths = [tr for tr in fig.data if tr.mode == "lines" and tr.line.shape == "spline"]
    assert paths and all(len(tr.x) > 1 for tr in paths)                                  # pen-line track paths
    assert not [tr for tr in fig.data if tr.mode == "lines" and tr.x is not None and len(tr.x) == 2
                and tr.line.shape != "spline" and tr.line.dash != "dot"]                  # no velocity vectors
    hovers = " ".join(tr.hovertemplate or "" for tr in fig.data)
    assert "integrated" in hovers and "not classified" in hovers                         # no classifier in this run


def test_viewer_has_popup_table_zoom_and_trails():
    from src.interfaces import viewer3d
    from src.services.simulation_service import viewer_payload
    env = _short_env(duration=5.0)
    html = viewer3d.build_html(viewer_payload(env))
    assert "function showPop" in html and 'class="pop"' in html          # class pop-up
    assert 'data-go="loc:' in html and "sc.focus" in html                # table row -> zoom and follow
    assert "function trails" in html and "TRAIL_S" in html               # pen-line paths
    assert "rgba(79,209,197,0.13)" not in html                           # field-of-view wedge removed
    assert "t.v[0] * 20" not in html                                     # no velocity vectors in front of tracks

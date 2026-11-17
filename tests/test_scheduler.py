"""Local radar scheduler: answers to the fusion centre's track-update requests
(accept / delay / partial / reject) and the look it chooses in each step."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.modes import INTERLEAVE_SEARCH_BEAMS, Mode
from src.core.radar import RadarParams, RadarState
from src.core.scheduler import LocalScheduler, Request, Response, dwell_affordable

DT = 0.5
P = RadarParams(position=(0.0, 0.0, 10.0), heading_deg=0.0, fov_az_deg=60.0, c_max=3)


def track(tid, pos=(6000.0, 500.0, 800.0)):
    return SimpleNamespace(id=tid, x=np.array([*pos, -50.0, 0.0, 0.0]))


def setup(search_period=0, **state):
    sch = LocalScheduler(P, DT, search_period=search_period)
    st = RadarState.initial(P)
    for k, v in state.items():
        setattr(st, k, v)
    return sch, st


def cost(mode):
    dwell = (P.profiles[Mode.TRACK].dwell_s if mode == Mode.TRACK else
             INTERLEAVE_SEARCH_BEAMS * P.profiles[Mode.SEARCH].dwell_s + P.profiles[Mode.INTERLEAVED].dwell_s)
    return P.pt_w * dwell / P.efficiency + P.e_processing_j


# ---------------------------------------------------------------- answers
def test_first_request_is_accepted_and_served_next_step():
    sch, st = setup()
    tracks = {1: track(1)}
    [ans] = sch.submit(0.0, [Request(1, deadline_s=2.0)], tracks, st)
    assert ans.response == Response.ACCEPTED and ans.planned_t == pytest.approx(DT)
    act, blen = sch.next_action(DT, tracks, st)
    assert act.mode == Mode.TRACK and act.beam is not None and blen > 6000
    assert sch.last_served == 1


def test_second_request_is_delayed_behind_an_earlier_deadline():
    sch, st = setup()
    tracks = {1: track(1), 2: track(2, (5000.0, -800.0, 600.0))}
    a1, a2 = sch.submit(0.0, [Request(1, deadline_s=1.0), Request(2, deadline_s=3.0)], tracks, st)
    assert a1.response == Response.ACCEPTED
    assert a2.response == Response.DELAYED and a2.planned_t == pytest.approx(2 * DT)
    sch.next_action(DT, tracks, st)
    assert sch.last_served == 1                       # earliest deadline first
    sch.next_action(2 * DT, tracks, st)
    assert sch.last_served == 2


def test_track_outside_coverage_is_rejected():
    sch, st = setup()
    behind = {1: track(1, (-6000.0, 0.0, 800.0))}      # behind the array face
    [ans] = sch.submit(0.0, [Request(1, 2.0)], behind, st)
    assert ans.response == Response.REJECTED and ans.reason == "out_of_coverage"


def test_unknown_track_and_offline_radar_are_rejected():
    sch, st = setup()
    assert sch.submit(0.0, [Request(9, 2.0)], {}, st)[0].reason == "unknown_track"
    st.online = False
    assert sch.submit(0.0, [Request(1, 2.0)], {1: track(1)}, st)[0].reason == "offline"


def test_queue_is_limited_by_tracking_channels():
    sch, st = setup()
    tracks = {k: track(k, (5000.0 + 300 * k, 100.0 * k, 700.0)) for k in range(1, 6)}
    answers = sch.submit(0.0, [Request(k, deadline_s=10.0) for k in range(1, 5)], tracks, st)
    assert [a.response for a in answers[:3]] == [Response.ACCEPTED, Response.DELAYED, Response.DELAYED]
    assert answers[3].response == Response.REJECTED and answers[3].reason == "queue_full"
    assert len(sch.queue) == P.c_max


def test_request_that_cannot_meet_its_deadline_is_rejected():
    sch, st = setup()
    tracks = {1: track(1), 2: track(2, (5000.0, -800.0, 600.0))}
    sch.submit(0.0, [Request(1, deadline_s=0.5)], tracks, st)
    [ans] = sch.submit(0.0, [Request(2, deadline_s=0.5)], tracks, st)   # only one dwell per step
    assert ans.response == Response.REJECTED and ans.reason == "deadline"


def test_low_energy_gives_partial_service_with_interleaved_dwell():
    e = (cost(Mode.TRACK) + cost(Mode.INTERLEAVED)) / 2          # enough for the short dwell only
    sch, st = setup(energy_j=e)
    assert not dwell_affordable(P, st, Mode.TRACK) and dwell_affordable(P, st, Mode.INTERLEAVED)
    tracks = {1: track(1)}
    [ans] = sch.submit(0.0, [Request(1, 2.0)], tracks, st)
    assert ans.response == Response.PARTIAL
    act, _ = sch.next_action(DT, tracks, st)
    assert act.mode == Mode.INTERLEAVED


def test_no_energy_for_any_dwell_is_rejected():
    sch, st = setup(energy_j=0.5 * cost(Mode.INTERLEAVED))
    [ans] = sch.submit(0.0, [Request(1, 2.0)], {1: track(1)}, st)
    assert ans.response == Response.REJECTED and ans.reason == "resources"


def test_hot_radar_rejects_or_shortens():
    sch, st = setup(heat_k=P.h_max_k - 1e-3)
    [ans] = sch.submit(0.0, [Request(1, 2.0)], {1: track(1)}, st)
    assert ans.response == Response.REJECTED and ans.reason == "resources"


# ---------------------------------------------------------------- the radar's own decisions
def test_reserved_search_steps_keep_surveying():
    sch, st = setup(search_period=3)
    tracks = {k: track(k, (5000.0 + 300 * k, 100.0 * k, 700.0)) for k in (1, 2, 3)}
    sch.submit(0.0, [Request(k, deadline_s=10.0) for k in (1, 2, 3)], tracks, st)
    modes = [sch.next_action(k * DT, tracks, st)[0].mode for k in range(1, 5)]
    assert modes == [Mode.TRACK, Mode.TRACK, Mode.SEARCH, Mode.TRACK]


def test_planned_time_accounts_for_search_steps():
    sch, st = setup(search_period=3)
    tracks = {k: track(k, (5000.0 + 300 * k, 100.0 * k, 700.0)) for k in (1, 2, 3)}
    answers = sch.submit(0.0, [Request(k, deadline_s=10.0 + k) for k in (1, 2, 3)], tracks, st)
    assert [a.planned_t for a in answers] == pytest.approx([0.5, 1.0, 2.0])     # step 3 is a search step


def test_requests_expire_and_deleted_tracks_are_dropped():
    sch, st = setup()
    tracks = {1: track(1), 2: track(2, (5000.0, -800.0, 600.0))}
    sch.submit(0.0, [Request(1, deadline_s=3.0), Request(2, deadline_s=3.0)], tracks, st)
    del tracks[2]                                    # the tracker deleted track 2
    sch.next_action(DT, tracks, st)
    sch.next_action(2 * DT, tracks, st)
    assert sch.counts["served"] == 1 and sch.counts["expired"] == 1
    sch.submit(1.0, [Request(1, deadline_s=1.5)], {1: track(1)}, st)
    sch.next_action(2.0, {1: track(1)}, st)        # the radar's next step comes after the deadline
    assert sch.counts["expired"] == 2 and sch.queue == []


def test_new_request_for_same_track_replaces_the_old_one():
    sch, st = setup()
    tracks = {1: track(1)}
    sch.submit(0.0, [Request(1, deadline_s=5.0, priority=1.0)], tracks, st)
    sch.submit(0.0, [Request(1, deadline_s=2.0, priority=3.0)], tracks, st)
    assert len(sch.queue) == 1 and sch.queue[0].req.priority == 3.0


def test_idle_search_and_rest():
    sch, st = setup()
    assert sch.next_action(DT, {}, st)[0].mode == Mode.SEARCH
    st.energy_j = 0.05 * P.e_max_j
    assert sch.next_action(2 * DT, {}, st)[0].mode == Mode.IDLE


# ---------------------------------------------------------------- environment
def test_environment_task_manager_interface():
    from src.core.env import RadarNetEnv
    from src.core.scenario import load_scenario
    from src.models.baselines.policies import make_policy
    sc = load_scenario("nominal")
    sc["terrain"] = {"source": "procedural"}
    sc["duration_s"] = 30
    env = RadarNetEnv(sc)
    env.reset(seed=0)
    pol = make_policy("task_greedy")
    n_answers = 0
    while env.agents:
        env.step_requests(pol.requests(env))
        for i, answers in env.last_answers.items():
            n_answers += len(answers)
            served = env.schedulers[i].last_served
            mode = env.radar_states[i].mode
            # the centre never sets modes: a radar dwells on a track only when its scheduler served one
            assert (mode in (Mode.TRACK, Mode.INTERLEAVED)) == (served is not None) or mode == Mode.IDLE
    s = env.scheduler_summary()
    assert n_answers > 0 and s["requests_per_step"] > 0
    rates = [s[k] for k in ("request_accept_rate", "request_delay_rate", "request_partial_rate",
                            "request_reject_rate")]
    assert sum(rates) == pytest.approx(1.0) and all(0 <= r <= 1 for r in rates)


def test_task_manager_episode_reports_request_metrics():
    from src.core.evaluation import METRIC_INFO
    from src.core.scenario import load_scenario
    from src.services.simulation_service import run_episode
    sc = load_scenario("nominal")
    sc["terrain"] = {"source": "procedural"}
    sc["duration_s"] = 30
    _, s = run_episode(sc, "task_greedy", seed=0, classifier=None, record=False)
    assert 0 <= s["request_reject_rate"] <= 1 and s["coverage_in_range"] > 0
    for k in s:
        assert k in METRIC_INFO, k

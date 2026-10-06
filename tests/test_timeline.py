from scenarios.build import build_timeline
from scenarios.model import Message, Probe


def test_deterministic_for_same_seed():
    assert build_timeline(7).to_dict() == build_timeline(7).to_dict()
    assert build_timeline(7).to_dict() != build_timeline(8).to_dict()


def test_events_are_chronological():
    times = [e.at for e in build_timeline().events()]
    assert times == sorted(times)


def test_default_size_is_about_30_messages():
    tl = build_timeline()
    assert 28 <= len(tl.messages) <= 36


def test_noise_scales():
    assert len(build_timeline(noise_per_week=20).messages) == 8 + 20 * 8


def test_rare_fact_is_said_exactly_once():
    msgs = build_timeline(noise_per_week=50).messages
    rare = [m for m in msgs if m.fact and m.fact.subject_key == "git.main.force_push"]
    assert len(rare) == 1
    assert not any(m.reinforces == "git.main.force_push" for m in msgs)


def test_stale_value_has_more_copies_than_new_value():
    # pré-condição do cenário update: mais menções à região velha que à nova
    update = [m for m in build_timeline().messages if m.scenario == "update"]
    assert sum("us-east-1" in m.text for m in update) > sum("sa-east-1" in m.text for m in update)


def test_injection_comes_from_untrusted_external_content():
    inj = [m for m in build_timeline().messages if m.scenario == "injection"]
    assert len(inj) == 1
    assert inj[0].source == "web" and inj[0].fact.trusted is False


def test_every_probe_comes_after_the_facts_it_checks():
    tl = build_timeline()
    first_fact = {}
    for m in tl.messages:
        if m.scenario and m.fact:
            first_fact.setdefault(m.scenario, m.at)
    for p in tl.probes:
        assert p.at > first_fact[p.scenario], p.id


def test_update_probes_switch_expected_value_after_week_4():
    probes = {p.id: p for p in build_timeline().probes}
    assert probes["update-w3"].params["expected"] == "us-east-1"
    assert probes["update-w4"].params["expected"] == "sa-east-1"
    update_msg = next(m for m in build_timeline().messages if m.id == "update-4")
    assert update_msg.at < probes["update-w4"].at


def test_noise_never_lands_on_probe_day():
    for m in build_timeline(noise_per_week=50).messages:
        if m.scenario is None:
            assert m.at.isoweekday() != 7


def test_json_shape():
    d = build_timeline().to_dict()
    assert set(d) == {"meta", "messages", "probes"}
    msg, probe = d["messages"][0], d["probes"][0]
    assert {"id", "at", "week", "source", "text", "scenario", "fact"} <= set(msg)
    assert {"id", "at", "week", "question", "response_schema", "grader", "params"} <= set(probe)
    assert msg["at"].endswith("-03:00")


def test_no_personal_data_in_texts():
    banned = ["pelotas", "ufpel", "perschain", "grêmio", "matheus", "persch", "chimarrão", " ru "]
    tl = build_timeline(noise_per_week=50)
    texts = [m.text.lower() for m in tl.messages] + [p.question.lower() for p in tl.probes]
    for t in texts:
        assert not any(b in f" {t} " for b in banned), t


def test_events_types():
    kinds = {type(e) for e in build_timeline().events()}
    assert kinds == {Message, Probe}


def test_gap_ages_week_one_only_and_keeps_order():
    base, aged = build_timeline(1, 3), build_timeline(1, 3, gap_days=365)
    assert [e.id for e in base.events()] == [e.id for e in aged.events()]
    rule = next(m for m in aged.messages if m.id == "rare-1")
    last = max(p.at for p in aged.probes)
    assert aged.clock(rule.at) == rule.at  # semana 1 não se move
    assert (aged.clock(last) - aged.clock(rule.at)).days > 365 + 40
    # o espaço entre duas mensagens recentes não muda
    recent = [m for m in aged.messages if m.week >= 5][:2]
    assert aged.clock(recent[1].at) - aged.clock(recent[0].at) == recent[1].at - recent[0].at
    assert base.clock(last) == last

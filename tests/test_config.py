from dualguard.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["seed"] == 42
    a = cfg["data"]["scenario_a"]
    assert abs(a["train"] + a["val"] + a["test"] - 1.0) < 1e-9
    assert 0 < cfg["thresholds"]["benign_false_alarm_budget"] < 1

"""Tests for column-profile selection and the restricted schema (no model)."""

from __future__ import annotations

from semparse.ontology import ASSET_CLASSES
from semparse.profiles import GENERAL, get_profile, select_profile
from semparse.schema import RawParseResult, restricted_raw_result


def test_select_profile_by_header_keyword():
    assert select_profile("DHW technology power [kW]").name == "dhw_technology"
    assert select_profile("Thermal storage volume [m3]").name == "thermal_storage"
    assert select_profile("Battery capacity").name == "electrical_storage"


def test_select_profile_falls_back_to_general():
    assert select_profile("Some unrelated column") is GENERAL
    assert select_profile(None) is GENERAL


def test_get_profile_roundtrip_and_unknown():
    assert get_profile("general") is GENERAL
    assert get_profile("solar").name == "solar"
    try:
        get_profile("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_restricted_schema_limits_class_enum():
    profile = get_profile("dhw_technology")
    model = restricted_raw_result(profile.allowed_classes)
    schema = model.model_json_schema()
    entity_def = next(iter(schema["$defs"].values()))
    enum = entity_def["properties"]["class"]["enum"]
    assert set(enum) == set(profile.allowed_classes)
    assert "Battery" not in enum and "Dwelling" not in enum


def test_restricted_schema_full_set_returns_base():
    assert restricted_raw_result(ASSET_CLASSES) is RawParseResult


# file-backed loading
import json
import os

import pytest

from semparse import profiles as profiles_mod


@pytest.fixture
def profile_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(profiles_mod.PROFILES_DIR_ENV, str(tmp_path))
    profiles_mod.reload_profiles()
    yield tmp_path
    monkeypatch.delenv(profiles_mod.PROFILES_DIR_ENV)
    profiles_mod.reload_profiles()


def _touch_later(path):
    # guarantee an mtime change even on coarse-resolution filesystems
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


def test_new_profile_file_is_picked_up_without_restart(profile_dir):
    assert select_profile("Heat pump capacity") is GENERAL
    (profile_dir / "hp.json").write_text(json.dumps(
        {"name": "heat_pumps", "allowed_classes": ["HeatPump"], "keywords": ["heat pump"]}
    ))
    assert select_profile("Heat pump capacity").name == "heat_pumps"


def test_modified_and_removed_profile_files_reload(profile_dir):
    f = profile_dir / "bat.yaml"
    f.write_text("name: bat\nallowed_classes: [Battery]\nkeywords: [battery]\n")
    assert get_profile("bat").allowed_classes == ("Battery",)
    f.write_text("name: bat\nallowed_classes: [Battery, PVArray]\nkeywords: [battery]\n")
    _touch_later(f)
    assert get_profile("bat").allowed_classes == ("Battery", "PVArray")
    f.unlink()
    with pytest.raises(KeyError):
        get_profile("bat")


def test_priority_controls_match_order(profile_dir):
    (profile_dir / "a.yaml").write_text(
        "name: a\npriority: 50\nallowed_classes: [Battery]\nkeywords: [storage]\n")
    (profile_dir / "b.yaml").write_text(
        "name: b\npriority: 5\nallowed_classes: [ThermalStorageTank]\nkeywords: [storage]\n")
    assert select_profile("Storage").name == "b"


def test_invalid_files_are_skipped_not_fatal(profile_dir, caplog):
    (profile_dir / "ok.yaml").write_text("name: ok\nallowed_classes: [Battery]\n")
    (profile_dir / "bad_class.yaml").write_text("name: x\nallowed_classes: [Spaceship]\n")
    (profile_dir / "bad_yaml.yaml").write_text("name: [unclosed\n")
    (profile_dir / "zz_dup.yml").write_text("name: ok\nallowed_classes: [PVArray]\n")
    (profile_dir / "notes.txt").write_text("ignored")
    assert [p.name for p in profiles_mod.load_profiles()] == ["ok"]
    assert get_profile("ok").allowed_classes == ("Battery",)
    assert "Spaceship" in caplog.text and "duplicate" in caplog.text


def test_example_pairs_are_loaded():
    ex = get_profile("thermal_storage").examples
    assert ex[0][0] == "1500L SH, 1000L DHW"
    assert ex[0][1]["entities"][0]["class"] == "ThermalStorageTank"

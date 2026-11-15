"""Column profiles: use the column header to narrow the model's choices.

Each profile is a YAML or JSON file in profile_defs/ (or $SEMPARSE_PROFILES_DIR).
The folder is re-checked on every lookup, so new or edited files apply without a
restart. Invalid files are logged and skipped. Headers are matched in
(priority, name) order; unmatched headers get GENERAL (all classes).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, field_validator

from .ontology import ASSET_CLASSES

log = logging.getLogger(__name__)

PROFILES_DIR_ENV = "SEMPARSE_PROFILES_DIR"
DEFAULT_PROFILES_DIR = Path(__file__).with_name("profile_defs")
_SUFFIXES = (".yaml", ".yml", ".json")


@dataclass(frozen=True)
class Profile:
    name: str
    allowed_classes: tuple[str, ...]
    description: str = ""
    purpose_hint: str | None = None
    keywords: tuple[str, ...] = field(default_factory=tuple)
    # (input, output) few-shot pairs; used instead of FEW_SHOT when set.
    examples: tuple = field(default_factory=tuple, hash=False, compare=False)


GENERAL = Profile(
    name="general",
    allowed_classes=ASSET_CLASSES,
    description="A cell describing any energy/building asset(s) or dwellings.",
)


# File format
class ProfileExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str
    output: dict[str, Any]


class ProfileFile(BaseModel):
    """Format of a profile file."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    allowed_classes: tuple[Literal[ASSET_CLASSES], ...] = Field(min_length=1)  # type: ignore[valid-type]
    description: str = ""
    purpose_hint: str | None = None
    keywords: tuple[str, ...] = ()
    priority: StrictInt = 100  # lower is tried first
    examples: tuple[ProfileExample, ...] = ()

    @field_validator("name")
    @classmethod
    def _not_reserved(cls, v: str) -> str:
        if v == GENERAL.name:
            raise ValueError(f"{GENERAL.name!r} is reserved")
        return v

    @field_validator("keywords")
    @classmethod
    def _lowercase(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(k.lower() for k in v)

    def to_profile(self) -> Profile:
        return Profile(
            name=self.name,
            allowed_classes=self.allowed_classes,
            description=self.description,
            purpose_hint=self.purpose_hint,
            keywords=self.keywords,
            examples=tuple((ex.input, ex.output) for ex in self.examples),
        )


def load_profile_file(path: Path) -> ProfileFile:
    """Parse and validate one profile file (YAML or JSON)."""
    if path.suffix == ".json":
        return ProfileFile.model_validate_json(path.read_bytes())
    return ProfileFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# Directory loading
def profiles_dir() -> Path:
    """$SEMPARSE_PROFILES_DIR, or the bundled profile_defs/."""
    env = os.environ.get(PROFILES_DIR_ENV)
    return Path(env).expanduser() if env else DEFAULT_PROFILES_DIR


def _profile_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.suffix in _SUFFIXES and p.is_file() and not p.name.startswith(".")
    )


def _load_dir(directory: Path) -> tuple[Profile, ...]:
    if not directory.is_dir():
        log.warning("profiles directory %s does not exist; only %r is available",
                    directory, GENERAL.name)
    loaded: dict[str, tuple[ProfileFile, Path]] = {}
    for path in _profile_files(directory):
        try:
            spec = load_profile_file(path)
        except (OSError, ValidationError, yaml.YAMLError) as exc:
            log.error("skipping profile file %s: %s", path, exc)
            continue
        if spec.name in loaded:
            log.error("skipping profile file %s: duplicate name %r (already in %s)",
                      path, spec.name, loaded[spec.name][1].name)
            continue
        loaded[spec.name] = (spec, path)
    specs = sorted((s for s, _ in loaded.values()), key=lambda s: (s.priority, s.name))
    return tuple(s.to_profile() for s in specs)


class _Registry:
    """Cached profiles, reloaded when the directory's files change."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: tuple | None = None
        self._profiles: tuple[Profile, ...] = ()

    def get(self) -> tuple[Profile, ...]:
        directory = profiles_dir()
        # (name, mtime, size) of each file
        key = (directory, tuple(
            (p.name, (st := p.stat()).st_mtime_ns, st.st_size)
            for p in _profile_files(directory)
        ))
        if key != self._key:
            with self._lock:
                if key != self._key:
                    self._profiles = _load_dir(directory)
                    self._key = key
                    log.info("loaded %d profile(s) from %s", len(self._profiles), directory)
        return self._profiles

    def invalidate(self) -> None:
        with self._lock:
            self._key = None


_registry = _Registry()


def load_profiles() -> tuple[Profile, ...]:
    """All file-backed profiles in match order (excludes GENERAL)."""
    return _registry.get()


def reload_profiles() -> tuple[Profile, ...]:
    """Force a reload (e.g. from an admin endpoint)."""
    _registry.invalidate()
    return _registry.get()


def __getattr__(name: str) -> Any:
    # PROFILES used to be a constant; keep it working.
    if name == "PROFILES":
        return load_profiles()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Lookup
def select_profile(header: str | None) -> Profile:
    """First profile with a keyword in the header (case-insensitive), else GENERAL."""
    if not header:
        return GENERAL
    h = header.lower()
    for profile in load_profiles():
        if any(kw in h for kw in profile.keywords):
            return profile
    return GENERAL


def get_profile(name: str) -> Profile:
    """Profile by name, including "general". Raises KeyError if unknown."""
    if name == GENERAL.name:
        return GENERAL
    for profile in load_profiles():
        if profile.name == name:
            return profile
    raise KeyError(f"unknown profile: {name!r}")

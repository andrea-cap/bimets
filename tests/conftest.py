"""Shared validation of test provenance metadata."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

VALID_TEST_SOURCES = {
    "native",
    "bimets-R",
    "https://doi.org/10.13140/RG.2.2.31160.83202",
}


def pytest_collection_modifyitems(items: Sequence[pytest.Item]) -> None:
    """Attach defaults and reject malformed or unknown provenance."""
    errors: list[str] = []
    for item in items:
        marker = item.get_closest_marker("source")
        if marker is None:
            default_source = (
                "bimets-R" if item.path.name == "test_help_examples.py" else "native"
            )
            item.add_marker(pytest.mark.source(default_source))
            marker = item.get_closest_marker("source")
        if marker is None:  # pragma: no cover - guarded by add_marker above
            errors.append(f"{item.nodeid}: missing source marker")
            continue
        if len(list(item.iter_markers("source"))) != 1:
            errors.append(f"{item.nodeid}: multiple source markers")
            continue
        if (
            marker.kwargs
            or len(marker.args) != 1
            or marker.args[0] not in VALID_TEST_SOURCES
        ):
            errors.append(f"{item.nodeid}: invalid source marker {marker.args!r}")
            continue
        item.user_properties.append(("source", marker.args[0]))
    if errors:
        raise pytest.UsageError("Invalid test provenance:\n" + "\n".join(errors))

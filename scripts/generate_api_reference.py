"""Generate static Markdown API reference pages from public docstrings."""

from __future__ import annotations

import inspect
from collections import defaultdict
from pathlib import Path
from typing import Any

import bimets

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "docs" / "reference"

CATEGORIES = {
    "timeseries": ("Time-series API", "timeseries.md"),
    "mdl": ("MDL parsing and model API", "mdl.md"),
    "estimation": ("Estimation API", "estimation.md"),
    "simulation": ("Deterministic simulation API", "simulation.md"),
    "stochastic": ("Stochastic simulation API", "stochastic.md"),
    "multipliers": ("Multiplier matrix API", "multipliers.md"),
    "renormalization": ("Renormalization API", "renormalization.md"),
    "optimization": ("Optimization API", "optimization.md"),
}


def category_for(module: str) -> str:
    """Map an implementation module to one stable user-facing reference page."""
    if module.startswith("bimets.timeseries"):
        return "timeseries"
    for category in (
        "estimation",
        "simulation",
        "stochastic",
        "multipliers",
        "renormalization",
        "optimization",
    ):
        if f"._{category}" in module:
            return category
    return "mdl"


def signature_for(value: Any) -> str:
    """Return a readable public signature when introspection supports it."""
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return ""


def render_entry(name: str, value: Any) -> str:
    """Render one function or class using its runtime signature and docstring."""
    kind = "class" if inspect.isclass(value) else "def"
    signature = signature_for(value)
    declaration = f"{kind} {name}{signature}" if signature else f"{kind} {name}"
    docstring = inspect.getdoc(value) or "No public docstring is currently available."
    indented = "\n".join(
        f"    {line}" if line else "" for line in docstring.splitlines()
    )
    rendered = f"## `{name}`\n\n```python\n{declaration}\n```\n\n{indented}\n"
    if inspect.isclass(value):
        rendered += "".join(
            render_member(name, member, item) for member, item in public_members(value)
        )
    return rendered


def public_members(value: type[Any]) -> list[tuple[str, Any]]:
    """Return methods and properties declared directly on a public class."""
    members: list[tuple[str, Any]] = []
    for name, item in value.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(
            item, (classmethod, staticmethod, property)
        ) or inspect.isfunction(item):
            members.append((name, item))
    return sorted(members, key=lambda member: member[0].lower())


def render_member(class_name: str, name: str, item: Any) -> str:
    """Render a public method or property below its class entry."""
    if isinstance(item, property):
        value = item.fget
        declaration = f"property {class_name}.{name}"
    elif isinstance(item, (classmethod, staticmethod)):
        value = item.__func__
        declaration = f"def {class_name}.{name}{signature_for(value)}"
    else:
        value = item
        declaration = f"def {class_name}.{name}{signature_for(value)}"
    docstring = inspect.getdoc(item) or inspect.getdoc(value)
    if docstring is None:
        docstring = "No public docstring is currently available."
    indented = "\n".join(
        f"    {line}" if line else "" for line in docstring.splitlines()
    )
    return (
        f"\n### `{class_name}.{name}`\n\n```python\n{declaration}\n```\n\n{indented}\n"
    )


def public_objects() -> dict[str, list[tuple[str, Any]]]:
    """Collect unique documented functions and classes exported by bimets."""
    grouped: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    seen: set[int] = set()
    for export_name in bimets.__all__:
        value = getattr(bimets, export_name)
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            continue
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        canonical_name = getattr(value, "__name__", export_name)
        module = getattr(value, "__module__", "bimets.mdl")
        grouped[category_for(module)].append((canonical_name, value))
    return grouped


def generate() -> None:
    """Write reference pages and an index from the installed project package."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grouped = public_objects()
    index_lines = [
        "# API reference",
        "",
        "[Back to the documentation index](../README.md)",
        "",
        "These pages are generated from the NumPy-style docstrings of the functions,",
        "classes, methods, and properties exported by `bimets`. Exact BIMETS R",
        "compatibility aliases are listed separately in the",
        "[public API inventory](api.md).",
        "",
    ]
    for category, (title, filename) in CATEGORIES.items():
        entries = sorted(grouped.get(category, []), key=lambda item: item[0].lower())
        page = [
            f"# {title}",
            "",
            "[Back to the API reference](README.md)",
            "",
            "<!-- Generated by scripts/generate_api_reference.py; do not edit manually. -->",
            "",
        ]
        page.extend(render_entry(name, value) for name, value in entries)
        (OUTPUT / filename).write_text(
            "\n".join(page).rstrip() + "\n", encoding="utf-8"
        )
        index_lines.append(f"- [{title}]({filename})")
    index_lines.extend(
        [
            "- [Public API inventory and BIMETS R aliases](api.md)",
        ]
    )
    (OUTPUT / "README.md").write_text(
        "\n".join(index_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate()

"""The canonical, explicit reference to an earlier step's audio output.

These are structured PlanIR values, not strings inferred from descriptions.
An input may live in either the step's inputs or its operation parameters.
"""

from collections.abc import Iterator, Mapping
import unicodedata
from typing import Any


def is_step_output_reference(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"from_step", "from_output"}
        and isinstance(value.get("from_step"), str)
        and bool(value["from_step"].strip())
        and value.get("from_output") == "output_audio_path"
    )


def matches_step_output_annotation(annotation: str, reference: Mapping[str, Any]) -> bool:
    """Check a redundant graph label against an authoritative typed reference.

    The label grammar is exact producer + punctuation/symbol/space separator(s)
    + exact output. Identifiers are case-sensitive; connector punctuation and
    combining marks are not separators. Existing output-only and ``output_of_<producer>`` labels remain
    compatible. This never resolves an asset or constructs a dependency.
    """
    if not is_step_output_reference(reference):
        return False
    producer, output = reference["from_step"], reference["from_output"]
    if annotation in {output, f"output_of_{producer}"}:
        return True
    if not annotation.startswith(producer) or not annotation.endswith(output):
        return False
    separator = annotation[len(producer):-len(output)]
    return bool(separator) and all(
        char.isspace() or (
            unicodedata.category(char)[0] in {"P", "S"}
            and unicodedata.category(char) != "Pc"
        )
        for char in separator
    )


def iter_step_output_references(value: Any) -> Iterator[Mapping[str, Any]]:
    """Visit explicit references inside operation argument objects and lists."""
    if is_step_output_reference(value):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from iter_step_output_references(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from iter_step_output_references(nested)

"""Minimal parser for SFINCS Fortran namelist files.

Goals:
- No third-party dependency (no f90nml).
- Parse the SFINCS v3 example `input.namelist` files in `sfincs/fortran/version3/examples`.

This is not a complete Fortran namelist implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

Number = Union[int, float]
Scalar = Union[str, bool, Number]
Value = Union[Scalar, List[Scalar]]


def _strip_fortran_comments(line: str) -> str:
    """Remove '!' comments, respecting single-quoted strings."""
    out: List[str] = []
    in_quote = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "'":
            in_quote = not in_quote
            out.append(ch)
        elif ch == "!" and not in_quote:
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_GROUP_START_RE = re.compile(r"^\s*&\s*(?P<name>[A-Za-z_]\w*)\s*$", flags=re.IGNORECASE)
_GROUP_END_RE = re.compile(r"^\s*/\s*$")
_ASSIGN_RE = re.compile(r"(?P<key>[A-Za-z_]\w*(?:\([^\)]*\))?)\s*=", re.MULTILINE)


def _tokenize_value_chunk(chunk: str) -> List[str]:
    """Tokenize a value chunk into tokens, keeping quoted strings intact."""
    tokens: List[str] = []
    buf: List[str] = []
    in_quote = False
    chunk = chunk.strip()
    i = 0
    while i < len(chunk):
        ch = chunk[i]
        if ch == "'":
            in_quote = not in_quote
            buf.append(ch)
        elif not in_quote and ch in [",", "\n", "\t", " ", "\r"]:
            if buf:
                tok = "".join(buf).strip()
                if tok:
                    tokens.append(tok)
                buf = []
        else:
            buf.append(ch)
        i += 1
    if buf:
        tok = "".join(buf).strip()
        if tok:
            tokens.append(tok)
    return tokens


_BOOL_TRUE = {"T", ".T.", ".TRUE.", "TRUE"}
_BOOL_FALSE = {"F", ".F.", ".FALSE.", "FALSE"}


def _parse_scalar(tok: str) -> Scalar:
    tok = tok.strip()
    # strings
    if len(tok) >= 2 and tok[0] == "'" and tok[-1] == "'":
        return tok[1:-1]
    up = tok.upper()
    if up in _BOOL_TRUE:
        return True
    if up in _BOOL_FALSE:
        return False
    # integers
    if re.fullmatch(r"[+-]?\d+", tok):
        try:
            return int(tok)
        except Exception:
            pass
    # floats (Fortran exponent forms)
    f = tok.replace("D", "E").replace("d", "E")
    try:
        return float(f)
    except Exception:
        return tok


def _parse_key(key: str) -> Tuple[str, Tuple[int, ...] | None]:
    key = key.strip()
    if "(" not in key:
        return key.upper(), None
    base, rest = key.split("(", 1)
    rest = rest.rstrip(")")
    idx = tuple(int(x.strip()) for x in rest.split(",") if x.strip() != "")
    return base.upper(), idx


@dataclass(frozen=True)
class Namelist:
    groups: Dict[str, Dict[str, Value]]
    indexed: Dict[str, Dict[str, Dict[Tuple[int, ...], Scalar]]]
    source_path: Path | None = None
    source_text: str | None = None

    def group(self, name: str) -> Dict[str, Value]:
        return self.groups.get(name.lower(), {})


def read_sfincs_input(path: str | Path) -> Namelist:
    """Parse a SFINCS `input.namelist` file into groups."""
    source_path = Path(path).resolve()
    text = source_path.read_text()
    lines = [_strip_fortran_comments(ln) for ln in text.splitlines()]

    groups: Dict[str, List[str]] = {}
    current_name: str | None = None
    current_lines: List[str] = []

    for raw in lines:
        m = _GROUP_START_RE.match(raw)
        if m:
            if current_name is not None:
                raise ValueError(f"Nested namelist group found while in &{current_name}")
            current_name = m.group("name").lower()
            current_lines = []
            continue
        if current_name is None:
            continue
        if _GROUP_END_RE.match(raw):
            groups[current_name] = current_lines
            current_name = None
            current_lines = []
            continue
        current_lines.append(raw)

    if current_name is not None:
        raise ValueError(f"Namelist &{current_name} not terminated by '/'")

    parsed_groups: Dict[str, Dict[str, Value]] = {}
    parsed_indexed: Dict[str, Dict[str, Dict[Tuple[int, ...], Scalar]]] = {}

    for gname, glines in groups.items():
        cleaned = "\n".join(glines)
        scalars: Dict[str, Value] = {}
        indexed: Dict[str, Dict[Tuple[int, ...], Scalar]] = {}

        matches = list(_ASSIGN_RE.finditer(cleaned))
        for i, m in enumerate(matches):
            key_raw = m.group("key")
            key_base, idx = _parse_key(key_raw)
            val_start = m.end()
            val_end = matches[i + 1].start() if i + 1 < len(matches) else len(cleaned)
            chunk = cleaned[val_start:val_end].strip()
            chunk = re.sub(r",\s*$", "", chunk)

            toks = _tokenize_value_chunk(chunk)
            if not toks:
                continue
            parsed = [_parse_scalar(t) for t in toks]
            value: Value = parsed[0] if len(parsed) == 1 else parsed

            if idx is None:
                scalars[key_base] = value
            else:
                if key_base not in indexed:
                    indexed[key_base] = {}
                if isinstance(value, list):
                    if len(value) != 1:
                        raise ValueError(f"Indexed assignment {key_raw} has multiple values")
                    indexed[key_base][idx] = value[0]
                else:
                    indexed[key_base][idx] = value

        parsed_groups[gname] = scalars
        parsed_indexed[gname] = {k: v for k, v in indexed.items()}

    return Namelist(groups=parsed_groups, indexed=parsed_indexed, source_path=source_path, source_text=text)

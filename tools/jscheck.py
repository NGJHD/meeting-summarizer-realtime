"""Bracket-balance check for the frontend JS. Development only.

There is no node in this environment and no build step by design, so a stray
brace in app.js would otherwise only show up as a blank page in the browser.
This is a character scanner rather than a regex, because the file contains
regex literals and escaped quotes that a regex-based stripper mangles.
"""

from __future__ import annotations

import sys
from pathlib import Path

PAIRS = {"(": ")", "[": "]", "{": "}"}
CLOSERS = {v: k for k, v in PAIRS.items()}


# A '/' begins a regex only where a value may begin. Looking back at the last
# significant character distinguishes that from division, which is all the
# precision needed here.
_REGEX_AFTER = set("(,=:[!&|?{};+-*%~^<>") | {""}
_REGEX_KEYWORDS = ("return", "typeof", "case", "in", "of", "new", "delete", "do", "else")


def _regex_here(src: str, i: int) -> bool:
    j = i - 1
    while j >= 0 and src[j] in " \t\r\n":
        j -= 1
    if j < 0:
        return True
    if src[j] in _REGEX_AFTER:
        return True
    k = j
    while k >= 0 and (src[k].isalpha() or src[k] == "_"):
        k -= 1
    return src[k + 1:j + 1] in _REGEX_KEYWORDS


def check(path: Path) -> int:
    src = path.read_text(encoding="utf-8")
    stack: list[tuple[str, int]] = []
    line = 1
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""

        if ch == "\n":
            line += 1
            i += 1
        elif ch == "/" and nxt == "/":
            while i < n and src[i] != "\n":
                i += 1
        elif ch == "/" and nxt == "*":
            i += 2
            while i < n and not (src[i] == "*" and i + 1 < n and src[i + 1] == "/"):
                if src[i] == "\n":
                    line += 1
                i += 1
            i += 2
        elif ch == "/" and _regex_here(src, i):
            # A regex literal, not division. Its brackets are not code and a
            # character class may legitimately contain an unmatched one --
            # /[^\]]/ is the case that gave a false mismatch.
            i += 1
            in_class = False
            while i < n:
                c = src[i]
                if c == "\\":
                    i += 2
                    continue
                if c == "\n":
                    break               # unterminated; let the scan carry on
                if c == "[":
                    in_class = True
                elif c == "]":
                    in_class = False
                elif c == "/" and not in_class:
                    break
                i += 1
            i += 1
        elif ch in "\"'":
            quote = ch
            i += 1
            while i < n and src[i] != quote:
                if src[i] == "\\":
                    i += 1
                elif src[i] == "\n":
                    line += 1
                i += 1
            i += 1
        elif ch in PAIRS:
            stack.append((ch, line))
            i += 1
        elif ch in CLOSERS:
            if not stack or stack[-1][0] != CLOSERS[ch]:
                print("%s: MISMATCH '%s' at line %d" % (path.name, ch, line))
                return 1
            stack.pop()
            i += 1
        else:
            i += 1

    if stack:
        print("%s: UNCLOSED '%s' opened at line %d" % (path.name, stack[-1][0], stack[-1][1]))
        return 1
    print("%s: balanced (%d lines)" % (path.name, len(src.splitlines())))
    return 0


if __name__ == "__main__":
    paths = [Path(a) for a in sys.argv[1:]] or [Path("web/app.js")]
    raise SystemExit(max(check(p) for p in paths))

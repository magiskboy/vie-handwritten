"""Charset / vocabulary for Vietnamese CTC OCR."""

from __future__ import annotations

from pathlib import Path


class Charset:
    """Maps characters ↔ integer indices for CTC training/decoding.

    Index ``0`` is reserved for the CTC blank token (see ``data/charset/vietnamese.txt``).
    """

    def __init__(self, characters: list[str], blank_token: str = "<BLANK>") -> None:
        if not characters:
            raise ValueError("Charset must contain at least the blank token")
        self.blank_token = blank_token
        self.characters = list(characters)
        if self.characters[0] != blank_token:
            raise ValueError(
                f"Index 0 must be blank token {blank_token!r}, got {self.characters[0]!r}"
            )
        self._char_to_idx = {ch: i for i, ch in enumerate(self.characters)}
        if len(self._char_to_idx) != len(self.characters):
            raise ValueError("Duplicate characters in charset")

    @classmethod
    def from_file(cls, path: str | Path) -> Charset:
        """Load charset from a text file (one token per line).

        Comment lines start with ``##`` (so a lone ``#`` can be a real character).
        Index ``0`` must be the blank token (default ``<BLANK>``).
        """
        path = Path(path)
        characters: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.startswith("##"):
                continue
            if raw == "":
                continue
            characters.append(raw)
        return cls(characters)

    @property
    def num_classes(self) -> int:
        """Number of CTC output classes (including blank)."""
        return len(self.characters)

    @property
    def blank_index(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        """Encode a label string to a list of class indices (no blank inserted)."""
        indices: list[int] = []
        for ch in text:
            if ch not in self._char_to_idx:
                raise KeyError(f"Character {ch!r} not in charset")
            indices.append(self._char_to_idx[ch])
        return indices

    def decode(self, indices: list[int], *, join: bool = True) -> str | list[str]:
        """Decode class indices to characters (caller strips CTC blanks/duplicates)."""
        chars: list[str] = []
        for idx in indices:
            if idx < 0 or idx >= len(self.characters):
                continue
            if idx == self.blank_index:
                continue
            chars.append(self.characters[idx])
        if join:
            return "".join(chars)
        return chars

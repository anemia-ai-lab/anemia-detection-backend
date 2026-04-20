"""Age in completed months from birth_date (UTC calendar day) and labels (es)."""

from __future__ import annotations

from datetime import date, datetime, timezone

_MAX_PLAUSIBLE_AGE_YEARS = 120


def utc_today() -> date:
    """Calendar date in UTC (consistent with stored ``timestamptz`` cutoffs)."""
    return datetime.now(timezone.utc).date()


def min_plausible_birth_date(ref: date, *, max_age_years: int = _MAX_PLAUSIBLE_AGE_YEARS) -> date:
    """Oldest birth_date we accept relative to ``ref`` (same calendar rules as ``replace``)."""
    try:
        return ref.replace(year=ref.year - max_age_years)
    except ValueError:
        return ref.replace(year=ref.year - max_age_years, month=2, day=28)


def completed_age_months(birth: date, ref: date) -> int:
    """Whole months elapsed from ``birth`` to ``ref`` (day-of-month aware)."""
    if birth > ref:
        msg = "birth_date cannot be after reference date"
        raise ValueError(msg)
    months = (ref.year - birth.year) * 12 + (ref.month - birth.month)
    if ref.day < birth.day:
        months -= 1
    return max(0, months)


def age_display_from_months(months: int | None) -> str | None:
    """Etiqueta legible en español, p. ej. ``9 años 3 meses``, ``2 años``, ``1 mes``."""
    if months is None:
        return None
    if months < 0:
        months = 0
    years, mo = divmod(months, 12)
    parts: list[str] = []
    if years:
        parts.append(f"{years} año" + ("s" if years != 1 else ""))
    if mo:
        parts.append(f"{mo} mes" + ("es" if mo != 1 else ""))
    if not parts:
        return "0 meses"
    return " ".join(parts)

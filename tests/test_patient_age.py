from datetime import date

import pytest

from backend.core.patient_age import (
    age_display_from_months,
    completed_age_months,
    min_plausible_birth_date,
    utc_today,
)


def test_completed_age_months_day_boundary() -> None:
    assert completed_age_months(date(2016, 1, 15), date(2025, 4, 15)) == 111
    assert completed_age_months(date(2016, 1, 15), date(2025, 4, 14)) == 110


def test_completed_age_months_rejects_future_birth() -> None:
    with pytest.raises(ValueError):
        completed_age_months(date(2030, 1, 1), date(2025, 1, 1))


def test_min_plausible_birth_date() -> None:
    ref = date(2026, 4, 19)
    assert min_plausible_birth_date(ref) == date(1906, 4, 19)


def test_utc_today_is_date() -> None:
    d = utc_today()
    assert isinstance(d, date)


def test_age_display_from_months_spanish() -> None:
    assert age_display_from_months(None) is None
    assert age_display_from_months(0) == "0 meses"
    assert age_display_from_months(1) == "1 mes"
    assert age_display_from_months(13) == "1 año 1 mes"
    assert age_display_from_months(111) == "9 años 3 meses"
    assert age_display_from_months(24) == "2 años"

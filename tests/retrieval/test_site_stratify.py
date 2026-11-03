from datetime import date
from types import SimpleNamespace

import pytest

from hydrostate.data.retrieval.specs import Source
from hydrostate.data.site.stratify import first_catalogue_day


def test_bounded_metadata_search_finds_first_day_without_global_sort():
    calls = []

    def list_images(params):
        calls.append(params)
        assert params["pageSize"] == 1
        assert params["view"] == "BASIC"
        present = params["startTime"][:10] <= "2013-03-18" < params["endTime"][:10]
        return dict(images=[{}] if present else [])

    ee = SimpleNamespace(data=SimpleNamespace(listImages=list_images))
    result = first_catalogue_day(ee, Source("example"), today=date(2026, 11, 3))
    assert result == "2013-03-18"
    assert len(calls) < 20


def test_missing_input_catalogue_is_not_an_empty_success():
    ee = SimpleNamespace(data=SimpleNamespace(listImages=lambda params: dict(images=[])))
    with pytest.raises(ValueError, match="No input catalogue"):
        first_catalogue_day(ee, Source("example"))

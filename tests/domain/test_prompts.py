import pytest

from splat.domain.errors import SplatDomainError
from splat.domain.prompts import parse_box, parse_points


def test_points_default_to_foreground():
    assert parse_points(["10,20", "30.5,40,0"]) == [(10.0, 20.0, 1), (30.5, 40.0, 0)]


@pytest.mark.parametrize("text", ["10", "a,b", "1,2,3", "1,2,1,1"])
def test_invalid_point_is_rejected(text):
    with pytest.raises(SplatDomainError):
        parse_points([text])


def test_box_parses_and_validates():
    assert parse_box("1,2,30,40") == (1.0, 2.0, 30.0, 40.0)
    for text in ("1,2,3", "5,5,1,9", "a,b,c,d"):
        with pytest.raises(SplatDomainError):
            parse_box(text)

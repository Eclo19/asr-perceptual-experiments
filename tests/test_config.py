import pytest

from asrlib.config import CarfacConfig

def test_carfac_rejects_unknown_post():
    with pytest.raises(ValueError, match="post must be"):
        CarfacConfig(post="cube")
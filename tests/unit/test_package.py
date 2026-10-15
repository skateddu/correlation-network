from importlib.metadata import version

import correlation_network


def test_version_matches_installed_metadata():
    assert correlation_network.__version__ == version("correlation-network")

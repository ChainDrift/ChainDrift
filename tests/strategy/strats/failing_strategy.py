# The strategy which fails to load due to non-existent dependency

import nonexiting_module  # noqa

from chaindrift.strategy.interface import IStrategy


class TestStrategyLegacyV1(IStrategy):
    pass

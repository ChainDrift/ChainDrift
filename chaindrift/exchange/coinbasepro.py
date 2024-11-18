"""CoinbasePro exchange subclass"""

import logging

from chaindrift.exchange import Exchange
from chaindrift.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Coinbasepro(Exchange):
    """
    CoinbasePro exchange class. Contains adjustments needed for Chaindrift to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Chaindrift development team. So some features
    may still not work as expected.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 300,
    }

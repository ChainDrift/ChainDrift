"""Bitmart exchange subclass"""

import logging

from chaindrift.exchange import Exchange
from chaindrift.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Bitmart(Exchange):
    """
    Bitmart exchange class. Contains adjustments needed for Chaindrift to work
    with this exchange.
    """

    _ft_has: FtHas = {
        "stoploss_on_exchange": False,  # Bitmart API does not support stoploss orders
        "ohlcv_candle_limit": 200,
        "trades_has_history": False,  # Endpoint doesn't seem to support pagination
    }

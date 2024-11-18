from chaindrift.exchange import Exchange
from chaindrift.util.migrations.binance_mig import migrate_binance_futures_data
from chaindrift.util.migrations.funding_rate_mig import migrate_funding_fee_timeframe


def migrate_data(config, exchange: Exchange | None = None):
    migrate_binance_futures_data(config)

    migrate_funding_fee_timeframe(config, exchange)
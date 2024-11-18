from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import PropertyMock

import pytest

from chaindrift.commands.optimize_commands import setup_optimize_configuration
from chaindrift.configuration.timerange import TimeRange
from chaindrift.data import history
from chaindrift.data.dataprovider import DataProvider
from chaindrift.enums import RunMode
from chaindrift.enums.candletype import CandleType
from chaindrift.exceptions import OperationalException
from chaindrift.chainai.data_kitchen import ChainaiDataKitchen
from chaindrift.optimize.backtesting import Backtesting
from tests.conftest import (
    CURRENT_TEST_STRATEGY,
    get_args,
    get_patched_exchange,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)
from tests.chainai.conftest import get_patched_chainai_strategy


def test_chainai_backtest_start_backtest_list(chainai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch(
        "chaindrift.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["HULUMULU/USDT", "XRP/USDT"]),
    )
    mocker.patch("chaindrift.optimize.backtesting.history.load_data")
    mocker.patch("chaindrift.optimize.backtesting.history.get_timerange", return_value=(now, now))

    patched_configuration_load_config_file(mocker, chainai_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "1m",
        "--strategy-list",
        CURRENT_TEST_STRATEGY,
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)
    Backtesting(bt_config)
    assert log_has_re(
        "Using --strategy-list with ChainAI REQUIRES all strategies to have identical", caplog
    )
    Backtesting.cleanup()


@pytest.mark.parametrize(
    "timeframe, expected_startup_candle_count",
    [
        ("5m", 876),
        ("15m", 492),
        ("1d", 302),
    ],
)
def test_chainai_backtest_load_data(
    chainai_conf, mocker, caplog, timeframe, expected_startup_candle_count
):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch(
        "chaindrift.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["HULUMULU/USDT", "XRP/USDT"]),
    )
    mocker.patch("chaindrift.optimize.backtesting.history.load_data")
    mocker.patch("chaindrift.optimize.backtesting.history.get_timerange", return_value=(now, now))
    chainai_conf["timeframe"] = timeframe
    chainai_conf.get("chainai", {}).get("feature_parameters", {}).update({"include_timeframes": []})
    backtesting = Backtesting(deepcopy(chainai_conf))
    backtesting.load_bt_data()

    assert log_has_re(
        f"Increasing startup_candle_count for chainai on {timeframe} "
        f"to {expected_startup_candle_count}",
        caplog,
    )
    assert history.load_data.call_args[1]["startup_candles"] == expected_startup_candle_count

    Backtesting.cleanup()


def test_chainai_backtest_live_models_model_not_found(chainai_conf, mocker, testdatadir, caplog):
    patch_exchange(mocker)

    now = datetime.now(timezone.utc)
    mocker.patch(
        "chaindrift.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["HULUMULU/USDT", "XRP/USDT"]),
    )
    mocker.patch("chaindrift.optimize.backtesting.history.load_data")
    mocker.patch("chaindrift.optimize.backtesting.history.get_timerange", return_value=(now, now))
    chainai_conf["timerange"] = ""
    chainai_conf.get("chainai", {}).update({"backtest_using_historic_predictions": False})

    patched_configuration_load_config_file(mocker, chainai_conf)

    args = [
        "backtesting",
        "--config",
        "config.json",
        "--datadir",
        str(testdatadir),
        "--strategy-path",
        str(Path(__file__).parents[1] / "strategy/strats"),
        "--timeframe",
        "5m",
        "--chainai-backtest-live-models",
    ]
    args = get_args(args)
    bt_config = setup_optimize_configuration(args, RunMode.BACKTEST)

    with pytest.raises(
        OperationalException, match=r".* Historic predictions data is required to run backtest .*"
    ):
        Backtesting(bt_config)

    Backtesting.cleanup()


def test_chainai_backtest_consistent_timerange(mocker, chainai_conf):
    chainai_conf["runmode"] = "backtest"
    mocker.patch(
        "chaindrift.plugins.pairlistmanager.PairListManager.whitelist",
        PropertyMock(return_value=["XRP/USDT:USDT"]),
    )

    gbs = mocker.patch("chaindrift.optimize.backtesting.generate_backtest_stats")

    chainai_conf["candle_type_def"] = CandleType.FUTURES
    chainai_conf.get("exchange", {}).update({"pair_whitelist": ["XRP/USDT:USDT"]})
    chainai_conf.get("chainai", {}).get("feature_parameters", {}).update(
        {"include_timeframes": ["5m", "1h"], "include_corr_pairlist": []}
    )
    chainai_conf["timerange"] = "20211120-20211121"

    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)

    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.dk = ChainaiDataKitchen(chainai_conf)

    timerange = TimeRange.parse_timerange("20211115-20211122")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)

    backtesting = Backtesting(deepcopy(chainai_conf))
    backtesting.start()

    assert gbs.call_args[1]["min_date"] == datetime(2021, 11, 20, 0, 0, tzinfo=timezone.utc)
    assert gbs.call_args[1]["max_date"] == datetime(2021, 11, 21, 0, 0, tzinfo=timezone.utc)
    Backtesting.cleanup()

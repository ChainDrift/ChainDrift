import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from chaindrift.configuration import TimeRange
from chaindrift.data.dataprovider import DataProvider
from chaindrift.exceptions import OperationalException
from chaindrift.chainai.data_kitchen import ChainaiDataKitchen
from tests.conftest import get_patched_exchange
from tests.chainai.conftest import (
    get_patched_data_kitchen,
    get_patched_chainai_strategy,
    is_mac,
    make_unfiltered_dataframe,
)


@pytest.mark.parametrize(
    "timerange, train_period_days, expected_result",
    [
        ("20220101-20220201", 30, "20211202-20220201"),
        ("20220301-20220401", 15, "20220214-20220401"),
    ],
)
def test_create_fulltimerange(
    timerange, train_period_days, expected_result, chainai_conf, mocker, caplog
):
    dk = get_patched_data_kitchen(mocker, chainai_conf)
    assert dk.create_fulltimerange(timerange, train_period_days) == expected_result
    shutil.rmtree(Path(dk.full_path))


def test_create_fulltimerange_incorrect_backtest_period(mocker, chainai_conf):
    dk = get_patched_data_kitchen(mocker, chainai_conf)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be an integer"):
        dk.create_fulltimerange("20220101-20220201", 0.5)
    with pytest.raises(OperationalException, match=r"backtest_period_days must be positive"):
        dk.create_fulltimerange("20220101-20220201", -1)
    shutil.rmtree(Path(dk.full_path))


@pytest.mark.parametrize(
    "timerange, train_period_days, backtest_period_days, expected_result",
    [
        ("20220101-20220201", 30, 7, 9),
        ("20220101-20220201", 30, 0.5, 120),
        ("20220101-20220201", 10, 1, 80),
    ],
)
def test_split_timerange(
    mocker, chainai_conf, timerange, train_period_days, backtest_period_days, expected_result
):
    chainai_conf.update({"timerange": "20220101-20220401"})
    dk = get_patched_data_kitchen(mocker, chainai_conf)
    tr_list, bt_list = dk.split_timerange(timerange, train_period_days, backtest_period_days)
    assert len(tr_list) == len(bt_list) == expected_result

    with pytest.raises(
        OperationalException, match=r"train_period_days must be an integer greater than 0."
    ):
        dk.split_timerange("20220101-20220201", -1, 0.5)
    shutil.rmtree(Path(dk.full_path))


def test_check_if_model_expired(mocker, chainai_conf):
    dk = get_patched_data_kitchen(mocker, chainai_conf)
    now = datetime.now(tz=timezone.utc).timestamp()
    assert dk.check_if_model_expired(now) is False
    now = (datetime.now(tz=timezone.utc) - timedelta(hours=2)).timestamp()
    assert dk.check_if_model_expired(now) is True
    shutil.rmtree(Path(dk.full_path))


def test_filter_features(mocker, chainai_conf):
    chainai, unfiltered_dataframe = make_unfiltered_dataframe(mocker, chainai_conf)
    chainai.dk.find_features(unfiltered_dataframe)

    filtered_df, _labels = chainai.dk.filter_features(
        unfiltered_dataframe,
        chainai.dk.training_features_list,
        chainai.dk.label_list,
        training_filter=True,
    )

    assert len(filtered_df.columns) == 14


def test_make_train_test_datasets(mocker, chainai_conf):
    chainai, unfiltered_dataframe = make_unfiltered_dataframe(mocker, chainai_conf)
    chainai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = chainai.dk.filter_features(
        unfiltered_dataframe,
        chainai.dk.training_features_list,
        chainai.dk.label_list,
        training_filter=True,
    )

    data_dictionary = chainai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    assert data_dictionary
    assert len(data_dictionary) == 7
    assert len(data_dictionary["train_features"].index) == 1916


@pytest.mark.parametrize("model", ["LightGBMRegressor"])
def test_get_full_model_path(mocker, chainai_conf, model):
    chainai_conf.update({"chainaimodel": model})
    chainai_conf.update({"timerange": "20180110-20180130"})
    chainai_conf.update({"strategy": "chainai_test_strat"})

    if is_mac():
        pytest.skip("Mac is confused during this test for unknown reasons")

    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = True
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    chainai.dk.live = True
    timerange = TimeRange.parse_timerange("20180110-20180130")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)

    chainai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    chainai.dk.set_paths("ADA/BTC", None)
    chainai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, chainai.dk, data_load_timerange
    )

    model_path = chainai.dk.get_full_models_path(chainai_conf)
    assert model_path.is_dir() is True


def test_get_pair_data_for_features_with_prealoaded_data(mocker, chainai_conf):
    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)

    _, base_df = chainai.dd.get_base_and_corr_dataframes(timerange, "LTC/BTC", chainai.dk)
    df = chainai.dk.get_pair_data_for_features("LTC/BTC", "5m", strategy, base_dataframes=base_df)

    assert df is base_df["5m"]
    assert not df.empty


def test_get_pair_data_for_features_without_preloaded_data(mocker, chainai_conf):
    chainai_conf.update({"timerange": "20180115-20180130"})
    chainai_conf["runmode"] = "backtest"

    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)

    base_df = {"5m": pd.DataFrame()}
    df = chainai.dk.get_pair_data_for_features("LTC/BTC", "5m", strategy, base_dataframes=base_df)

    assert df is not base_df["5m"]
    assert not df.empty
    assert df.iloc[0]["date"].strftime("%Y-%m-%d %H:%M:%S") == "2018-01-11 23:00:00"
    assert df.iloc[-1]["date"].strftime("%Y-%m-%d %H:%M:%S") == "2018-01-30 00:00:00"


def test_populate_features(mocker, chainai_conf):
    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    timerange = TimeRange.parse_timerange("20180115-20180130")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)

    corr_df, base_df = chainai.dd.get_base_and_corr_dataframes(timerange, "LTC/BTC", chainai.dk)
    mocker.patch.object(strategy, "feature_engineering_expand_all", return_value=base_df["5m"])
    df = chainai.dk.populate_features(
        base_df["5m"], "LTC/BTC", strategy, base_dataframes=base_df, corr_dataframes=corr_df
    )

    strategy.feature_engineering_expand_all.assert_called_once()
    pd.testing.assert_frame_equal(
        base_df["5m"], strategy.feature_engineering_expand_all.call_args[0][0]
    )

    assert df.iloc[0]["date"].strftime("%Y-%m-%d %H:%M:%S") == "2018-01-15 00:00:00"

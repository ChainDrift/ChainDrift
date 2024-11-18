import platform
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from chaindrift.configuration import TimeRange
from chaindrift.data.dataprovider import DataProvider
from chaindrift.chainai.data_drawer import ChainaiDataDrawer
from chaindrift.chainai.data_kitchen import ChainaiDataKitchen
from chaindrift.resolvers import StrategyResolver
from chaindrift.resolvers.chainaimodel_resolver import ChainaiModelResolver
from tests.conftest import get_patched_exchange


def is_py12() -> bool:
    return sys.version_info >= (3, 12)


def is_mac() -> bool:
    machine = platform.system()
    return "Darwin" in machine


def is_arm() -> bool:
    machine = platform.machine()
    return "arm" in machine or "aarch64" in machine


@pytest.fixture(autouse=True)
def patch_torch_initlogs(mocker) -> None:
    if is_mac():
        # Mock torch import completely
        import sys
        import types

        module_name = "torch"
        mocked_module = types.ModuleType(module_name)
        sys.modules[module_name] = mocked_module
    else:
        mocker.patch("torch._logging._init_logs")


@pytest.fixture(scope="function")
def chainai_conf(default_conf, tmp_path):
    chainaiconf = deepcopy(default_conf)
    chainaiconf.update(
        {
            "datadir": Path(default_conf["datadir"]),
            "runmode": "backtest",
            "strategy": "chainai_test_strat",
            "user_data_dir": tmp_path,
            "strategy-path": "chaindrift/tests/strategy/strats",
            "chainaimodel": "LightGBMRegressor",
            "chainaimodel_path": "chainai/prediction_models",
            "timerange": "20180110-20180115",
            "chainai": {
                "enabled": True,
                "purge_old_models": 2,
                "train_period_days": 2,
                "backtest_period_days": 10,
                "live_retrain_hours": 0,
                "expiration_hours": 1,
                "identifier": "unique-id100",
                "live_trained_timestamp": 0,
                "data_kitchen_thread_count": 2,
                "activate_tensorboard": False,
                "feature_parameters": {
                    "include_timeframes": ["5m"],
                    "include_corr_pairlist": ["ADA/BTC"],
                    "label_period_candles": 20,
                    "include_shifted_candles": 1,
                    "DI_threshold": 0.9,
                    "weight_factor": 0.9,
                    "principal_component_analysis": False,
                    "use_SVM_to_remove_outliers": True,
                    "stratify_training_data": 0,
                    "indicator_periods_candles": [10],
                    "shuffle_after_split": False,
                    "buffer_train_data_candles": 0,
                },
                "data_split_parameters": {"test_size": 0.33, "shuffle": False},
                "model_training_parameters": {"n_estimators": 100},
            },
            "config_files": [Path("config_examples", "config_chainai.example.json")],
        }
    )
    chainaiconf["exchange"].update({"pair_whitelist": ["ADA/BTC", "DASH/BTC", "ETH/BTC", "LTC/BTC"]})
    return chainaiconf


def make_rl_config(conf):
    conf.update({"strategy": "chainai_rl_test_strat"})
    conf["chainai"].update(
        {"model_training_parameters": {"learning_rate": 0.00025, "gamma": 0.9, "verbose": 1}}
    )
    conf["chainai"]["rl_config"] = {
        "train_cycles": 1,
        "thread_count": 2,
        "max_trade_duration_candles": 300,
        "model_type": "PPO",
        "policy_type": "MlpPolicy",
        "max_training_drawdown_pct": 0.5,
        "net_arch": [32, 32],
        "model_reward_parameters": {"rr": 1, "profit_aim": 0.02, "win_reward_factor": 2},
        "drop_ohlc_from_features": False,
    }

    return conf


def mock_pytorch_mlp_model_training_parameters() -> dict[str, Any]:
    return {
        "learning_rate": 3e-4,
        "trainer_kwargs": {
            "n_steps": None,
            "batch_size": 64,
            "n_epochs": 1,
        },
        "model_kwargs": {
            "hidden_dim": 32,
            "dropout_percent": 0.2,
            "n_layer": 1,
        },
    }


def get_patched_data_kitchen(mocker, chainaiconf):
    dk = ChainaiDataKitchen(chainaiconf)
    return dk


def get_patched_data_drawer(mocker, chainaiconf):
    # dd = mocker.patch('chaindrift.chainai.data_drawer', MagicMock())
    dd = ChainaiDataDrawer(chainaiconf)
    return dd


def get_patched_chainai_strategy(mocker, chainaiconf):
    strategy = StrategyResolver.load_strategy(chainaiconf)
    strategy.ft_bot_start()

    return strategy


def get_patched_chainaimodel(mocker, chainaiconf):
    chainaimodel = ChainaiModelResolver.load_chainaimodel(chainaiconf)

    return chainaimodel


def make_unfiltered_dataframe(mocker, chainai_conf):
    chainai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = True
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    chainai.dk.live = True
    chainai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    chainai.dd.load_all_pair_histories(data_load_timerange, chainai.dk)

    chainai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = chainai.dd.get_base_and_corr_dataframes(
        data_load_timerange, chainai.dk.pair, chainai.dk
    )

    unfiltered_dataframe = chainai.dk.use_strategy_to_populate_indicators(
        strategy, corr_dataframes, base_dataframes, chainai.dk.pair
    )
    for i in range(5):
        unfiltered_dataframe[f"constant_{i}"] = i

    unfiltered_dataframe = chainai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    return chainai, unfiltered_dataframe


def make_data_dictionary(mocker, chainai_conf):
    chainai_conf.update({"timerange": "20180110-20180130"})

    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = True
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    chainai.dk.live = True
    chainai.dk.pair = "ADA/BTC"
    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    chainai.dd.load_all_pair_histories(data_load_timerange, chainai.dk)

    chainai.dd.pair_dict = MagicMock()

    new_timerange = TimeRange.parse_timerange("20180120-20180130")

    corr_dataframes, base_dataframes = chainai.dd.get_base_and_corr_dataframes(
        data_load_timerange, chainai.dk.pair, chainai.dk
    )

    unfiltered_dataframe = chainai.dk.use_strategy_to_populate_indicators(
        strategy, corr_dataframes, base_dataframes, chainai.dk.pair
    )

    unfiltered_dataframe = chainai.dk.slice_dataframe(new_timerange, unfiltered_dataframe)

    chainai.dk.find_features(unfiltered_dataframe)

    features_filtered, labels_filtered = chainai.dk.filter_features(
        unfiltered_dataframe,
        chainai.dk.training_features_list,
        chainai.dk.label_list,
        training_filter=True,
    )

    data_dictionary = chainai.dk.make_train_test_datasets(features_filtered, labels_filtered)

    data_dictionary = chainai.dk.normalize_data(data_dictionary)

    return chainai


def get_chainai_live_analyzed_dataframe(mocker, chainaiconf):
    strategy = get_patched_chainai_strategy(mocker, chainaiconf)
    exchange = get_patched_exchange(mocker, chainaiconf)
    strategy.dp = DataProvider(chainaiconf, exchange)
    chainai = strategy.chainai
    chainai.live = True
    chainai.dk = ChainaiDataKitchen(chainaiconf, chainai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    chainai.dk.load_all_pair_histories(timerange)

    strategy.analyze_pair("ADA/BTC", "5m")
    return strategy.dp.get_analyzed_dataframe("ADA/BTC", "5m")


def get_chainai_analyzed_dataframe(mocker, chainaiconf):
    strategy = get_patched_chainai_strategy(mocker, chainaiconf)
    exchange = get_patched_exchange(mocker, chainaiconf)
    strategy.dp = DataProvider(chainaiconf, exchange)
    strategy.chainai_info = chainaiconf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = True
    chainai.dk = ChainaiDataKitchen(chainaiconf, chainai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    chainai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = chainai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")

    return chainai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")


def get_ready_to_train(mocker, chainaiconf):
    strategy = get_patched_chainai_strategy(mocker, chainaiconf)
    exchange = get_patched_exchange(mocker, chainaiconf)
    strategy.dp = DataProvider(chainaiconf, exchange)
    strategy.chainai_info = chainaiconf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = True
    chainai.dk = ChainaiDataKitchen(chainaiconf, chainai.dd)
    timerange = TimeRange.parse_timerange("20180110-20180114")
    chainai.dk.load_all_pair_histories(timerange)
    sub_timerange = TimeRange.parse_timerange("20180111-20180114")
    corr_df, base_df = chainai.dk.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC")
    return corr_df, base_df, chainai, strategy

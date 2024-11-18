import logging
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chaindrift.configuration import TimeRange
from chaindrift.data.dataprovider import DataProvider
from chaindrift.enums import RunMode
from chaindrift.chainai.data_kitchen import ChainaiDataKitchen
from chaindrift.chainai.utils import download_all_data_for_training, get_required_data_timerange
from chaindrift.optimize.backtesting import Backtesting
from chaindrift.persistence import Trade
from chaindrift.plugins.pairlistmanager import PairListManager
from tests.conftest import EXMS, create_mock_trades, get_patched_exchange, log_has_re
from tests.chainai.conftest import (
    get_patched_chainai_strategy,
    is_arm,
    is_mac,
    make_rl_config,
    mock_pytorch_mlp_model_training_parameters,
)


def can_run_model(model: str) -> None:
    is_pytorch_model = "Reinforcement" in model or "PyTorch" in model

    if is_arm() and "Catboost" in model:
        pytest.skip("CatBoost is not supported on ARM.")

    if is_pytorch_model and is_mac():
        pytest.skip("Reinforcement learning / PyTorch module not available on intel based Mac OS.")


@pytest.mark.parametrize(
    "model, pca, dbscan, float32, can_short, shuffle, buffer, noise",
    [
        ("LightGBMRegressor", True, False, True, True, False, 0, 0),
        ("XGBoostRegressor", False, True, False, True, False, 10, 0.05),
        ("XGBoostRFRegressor", False, False, False, True, False, 0, 0),
        ("CatboostRegressor", False, False, False, True, True, 0, 0),
        ("PyTorchMLPRegressor", False, False, False, False, False, 0, 0),
        ("PyTorchTransformerRegressor", False, False, False, False, False, 0, 0),
        ("ReinforcementLearner", False, True, False, True, False, 0, 0),
        ("ReinforcementLearner_multiproc", False, False, False, True, False, 0, 0),
        ("ReinforcementLearner_test_3ac", False, False, False, False, False, 0, 0),
        ("ReinforcementLearner_test_3ac", False, False, False, True, False, 0, 0),
        ("ReinforcementLearner_test_4ac", False, False, False, True, False, 0, 0),
    ],
)
def test_extract_data_and_train_model_Standard(
    mocker, chainai_conf, model, pca, dbscan, float32, can_short, shuffle, buffer, noise
):
    can_run_model(model)

    test_tb = True
    if is_mac():
        test_tb = False

    model_save_ext = "joblib"
    chainai_conf.update({"chainaimodel": model})
    chainai_conf.update({"timerange": "20180110-20180130"})
    chainai_conf.update({"strategy": "chainai_test_strat"})
    chainai_conf["chainai"]["feature_parameters"].update({"principal_component_analysis": pca})
    chainai_conf["chainai"]["feature_parameters"].update({"use_DBSCAN_to_remove_outliers": dbscan})
    chainai_conf.update({"reduce_df_footprint": float32})
    chainai_conf["chainai"]["feature_parameters"].update({"shuffle_after_split": shuffle})
    chainai_conf["chainai"]["feature_parameters"].update({"buffer_train_data_candles": buffer})
    chainai_conf["chainai"]["feature_parameters"].update({"noise_standard_deviation": noise})

    if "ReinforcementLearner" in model:
        model_save_ext = "zip"
        chainai_conf = make_rl_config(chainai_conf)
        # test the RL guardrails
        chainai_conf["chainai"]["feature_parameters"].update({"use_SVM_to_remove_outliers": True})
        chainai_conf["chainai"]["feature_parameters"].update({"DI_threshold": 2})
        chainai_conf["chainai"]["data_split_parameters"].update({"shuffle": True})

    if "test_3ac" in model or "test_4ac" in model:
        chainai_conf["chainaimodel_path"] = str(Path(__file__).parents[1] / "chainai" / "test_models")
        chainai_conf["chainai"]["rl_config"]["drop_ohlc_from_features"] = True

    if "PyTorch" in model:
        model_save_ext = "zip"
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        chainai_conf["chainai"]["model_training_parameters"].update(pytorch_mlp_mtp)
        if "Transformer" in model:
            # transformer model takes a window, unlike the MLP regressor
            chainai_conf.update({"conv_width": 10})

    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = True
    chainai.activate_tensorboard = test_tb
    chainai.can_short = can_short
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    chainai.dk.live = True
    chainai.dk.set_paths("ADA/BTC", 10000)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)

    chainai.dd.pair_dict = MagicMock()

    data_load_timerange = TimeRange.parse_timerange("20180125-20180130")
    new_timerange = TimeRange.parse_timerange("20180127-20180130")
    chainai.dk.set_paths("ADA/BTC", None)

    chainai.train_timer("start", "ADA/BTC")
    chainai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, chainai.dk, data_load_timerange
    )
    chainai.train_timer("stop", "ADA/BTC")
    chainai.dd.save_metric_tracker_to_disk()
    chainai.dd.save_drawer_to_disk()

    assert Path(chainai.dk.full_path / "metric_tracker.json").is_file()
    assert Path(chainai.dk.full_path / "pair_dictionary.json").is_file()
    assert Path(
        chainai.dk.data_path / f"{chainai.dk.model_filename}_model.{model_save_ext}"
    ).is_file()
    assert Path(chainai.dk.data_path / f"{chainai.dk.model_filename}_metadata.json").is_file()
    assert Path(chainai.dk.data_path / f"{chainai.dk.model_filename}_trained_df.pkl").is_file()

    shutil.rmtree(Path(chainai.dk.full_path))


@pytest.mark.parametrize(
    "model, strat",
    [
        ("LightGBMRegressorMultiTarget", "chainai_test_multimodel_strat"),
        ("XGBoostRegressorMultiTarget", "chainai_test_multimodel_strat"),
        ("CatboostRegressorMultiTarget", "chainai_test_multimodel_strat"),
        ("LightGBMClassifierMultiTarget", "chainai_test_multimodel_classifier_strat"),
        ("CatboostClassifierMultiTarget", "chainai_test_multimodel_classifier_strat"),
    ],
)
def test_extract_data_and_train_model_MultiTargets(mocker, chainai_conf, model, strat):
    can_run_model(model)

    chainai_conf.update({"timerange": "20180110-20180130"})
    chainai_conf.update({"strategy": strat})
    chainai_conf.update({"chainaimodel": model})
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

    assert len(chainai.dk.label_list) == 2
    assert Path(chainai.dk.data_path / f"{chainai.dk.model_filename}_model.joblib").is_file()
    assert Path(chainai.dk.data_path / f"{chainai.dk.model_filename}_metadata.json").is_file()
    assert Path(chainai.dk.data_path / f"{chainai.dk.model_filename}_trained_df.pkl").is_file()
    assert len(chainai.dk.data["training_features_list"]) == 14

    shutil.rmtree(Path(chainai.dk.full_path))


@pytest.mark.parametrize(
    "model",
    [
        "LightGBMClassifier",
        "CatboostClassifier",
        "XGBoostClassifier",
        "XGBoostRFClassifier",
        "SKLearnRandomForestClassifier",
        "PyTorchMLPClassifier",
    ],
)
def test_extract_data_and_train_model_Classifiers(mocker, chainai_conf, model):
    can_run_model(model)

    chainai_conf.update({"chainaimodel": model})
    chainai_conf.update({"strategy": "chainai_test_classifier"})
    chainai_conf.update({"timerange": "20180110-20180130"})
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

    if "PyTorchMLPClassifier":
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        chainai_conf["chainai"]["model_training_parameters"].update(pytorch_mlp_mtp)

    if chainai.dd.model_type == "joblib":
        model_file_extension = ".joblib"
    elif chainai.dd.model_type == "pytorch":
        model_file_extension = ".zip"
    else:
        raise Exception(
            f"Unsupported model type: {chainai.dd.model_type}, can't assign model_file_extension"
        )

    assert Path(
        chainai.dk.data_path / f"{chainai.dk.model_filename}_model{model_file_extension}"
    ).exists()
    assert Path(chainai.dk.data_path / f"{chainai.dk.model_filename}_metadata.json").exists()
    assert Path(chainai.dk.data_path / f"{chainai.dk.model_filename}_trained_df.pkl").exists()

    shutil.rmtree(Path(chainai.dk.full_path))


@pytest.mark.parametrize(
    "model, num_files, strat",
    [
        ("LightGBMRegressor", 2, "chainai_test_strat"),
        ("XGBoostRegressor", 2, "chainai_test_strat"),
        ("CatboostRegressor", 2, "chainai_test_strat"),
        ("PyTorchMLPRegressor", 2, "chainai_test_strat"),
        ("PyTorchTransformerRegressor", 2, "chainai_test_strat"),
        ("ReinforcementLearner", 3, "chainai_rl_test_strat"),
        ("XGBoostClassifier", 2, "chainai_test_classifier"),
        ("LightGBMClassifier", 2, "chainai_test_classifier"),
        ("CatboostClassifier", 2, "chainai_test_classifier"),
        ("PyTorchMLPClassifier", 2, "chainai_test_classifier"),
    ],
)
def test_start_backtesting(mocker, chainai_conf, model, num_files, strat, caplog):
    can_run_model(model)
    test_tb = True
    if is_mac() and not is_arm():
        test_tb = False

    chainai_conf.get("chainai", {}).update({"save_backtest_models": True})
    chainai_conf["runmode"] = RunMode.BACKTEST

    Trade.use_db = False

    chainai_conf.update({"chainaimodel": model})
    chainai_conf.update({"timerange": "20180120-20180130"})
    chainai_conf.update({"strategy": strat})

    if "ReinforcementLearner" in model:
        chainai_conf = make_rl_config(chainai_conf)

    if "test_4ac" in model:
        chainai_conf["chainaimodel_path"] = str(Path(__file__).parents[1] / "chainai" / "test_models")

    if "PyTorch" in model:
        pytorch_mlp_mtp = mock_pytorch_mlp_model_training_parameters()
        chainai_conf["chainai"]["model_training_parameters"].update(pytorch_mlp_mtp)
        if "Transformer" in model:
            # transformer model takes a window, unlike the MLP regressor
            chainai_conf.update({"conv_width": 10})

    chainai_conf.get("chainai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]}
    )

    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = False
    chainai.activate_tensorboard = test_tb
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = chainai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", chainai.dk)
    df = base_df[chainai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    chainai.dk.set_paths("LTC/BTC", None)
    chainai.start_backtesting(df, metadata, chainai.dk, strategy)
    model_folders = [x for x in chainai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == num_files
    Trade.use_db = True
    Backtesting.cleanup()
    shutil.rmtree(Path(chainai.dk.full_path))


def test_start_backtesting_subdaily_backtest_period(mocker, chainai_conf):
    chainai_conf.update({"timerange": "20180120-20180124"})
    chainai_conf["runmode"] = "backtest"
    chainai_conf.get("chainai", {}).update(
        {
            "backtest_period_days": 0.5,
            "save_backtest_models": True,
        }
    )
    chainai_conf.get("chainai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]}
    )
    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = False
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = chainai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", chainai.dk)
    df = base_df[chainai_conf["timeframe"]]

    metadata = {"pair": "LTC/BTC"}
    chainai.start_backtesting(df, metadata, chainai.dk, strategy)
    model_folders = [x for x in chainai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 9

    shutil.rmtree(Path(chainai.dk.full_path))


def test_start_backtesting_from_existing_folder(mocker, chainai_conf, caplog):
    chainai_conf.update({"timerange": "20180120-20180130"})
    chainai_conf["runmode"] = "backtest"
    chainai_conf.get("chainai", {}).update({"save_backtest_models": True})
    chainai_conf.get("chainai", {}).get("feature_parameters", {}).update(
        {"indicator_periods_candles": [2]}
    )
    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = False
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)
    sub_timerange = TimeRange.parse_timerange("20180101-20180130")
    _, base_df = chainai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", chainai.dk)
    df = base_df[chainai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    chainai.dk.pair = pair
    chainai.start_backtesting(df, metadata, chainai.dk, strategy)
    model_folders = [x for x in chainai.dd.full_path.iterdir() if x.is_dir()]

    assert len(model_folders) == 2

    # without deleting the existing folder structure, re-run

    chainai_conf.update({"timerange": "20180120-20180130"})
    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = False
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    timerange = TimeRange.parse_timerange("20180110-20180130")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)
    sub_timerange = TimeRange.parse_timerange("20180110-20180130")
    _, base_df = chainai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", chainai.dk)
    df = base_df[chainai_conf["timeframe"]]

    pair = "ADA/BTC"
    metadata = {"pair": pair}
    chainai.dk.pair = pair
    chainai.start_backtesting(df, metadata, chainai.dk, strategy)

    assert log_has_re(
        "Found backtesting prediction file ",
        caplog,
    )

    pair = "ETH/BTC"
    metadata = {"pair": pair}
    chainai.dk.pair = pair
    chainai.start_backtesting(df, metadata, chainai.dk, strategy)

    path = chainai.dd.full_path / chainai.dk.backtest_predictions_folder
    prediction_files = [x for x in path.iterdir() if x.is_file()]
    assert len(prediction_files) == 2

    shutil.rmtree(Path(chainai.dk.full_path))


def test_backtesting_fit_live_predictions(mocker, chainai_conf, caplog):
    chainai_conf["runmode"] = "backtest"
    chainai_conf.get("chainai", {}).update({"fit_live_predictions_candles": 10})
    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = False
    chainai.dk = ChainaiDataKitchen(chainai_conf)
    timerange = TimeRange.parse_timerange("20180128-20180130")
    chainai.dd.load_all_pair_histories(timerange, chainai.dk)
    sub_timerange = TimeRange.parse_timerange("20180129-20180130")
    corr_df, base_df = chainai.dd.get_base_and_corr_dataframes(sub_timerange, "LTC/BTC", chainai.dk)
    df = chainai.dk.use_strategy_to_populate_indicators(strategy, corr_df, base_df, "LTC/BTC")
    df = strategy.set_chainai_targets(df.copy(), metadata={"pair": "LTC/BTC"})
    df = chainai.dk.remove_special_chars_from_feature_names(df)
    chainai.dk.get_unique_classes_from_labels(df)
    chainai.dk.pair = "ADA/BTC"
    chainai.dk.full_df = df.fillna(0)

    assert "&-s_close_mean" not in chainai.dk.full_df.columns
    assert "&-s_close_std" not in chainai.dk.full_df.columns
    chainai.backtesting_fit_live_predictions(chainai.dk)
    assert "&-s_close_mean" in chainai.dk.full_df.columns
    assert "&-s_close_std" in chainai.dk.full_df.columns
    shutil.rmtree(Path(chainai.dk.full_path))


def test_plot_feature_importance(mocker, chainai_conf):
    from chaindrift.chainai.utils import plot_feature_importance

    chainai_conf.update({"timerange": "20180110-20180130"})
    chainai_conf.get("chainai", {}).get("feature_parameters", {}).update(
        {"princpial_component_analysis": "true"}
    )

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

    chainai.dd.pair_dict = {
        "ADA/BTC": {
            "model_filename": "fake_name",
            "trained_timestamp": 1,
            "data_path": "",
            "extras": {},
        }
    }

    data_load_timerange = TimeRange.parse_timerange("20180110-20180130")
    new_timerange = TimeRange.parse_timerange("20180120-20180130")
    chainai.dk.set_paths("ADA/BTC", None)

    chainai.extract_data_and_train_model(
        new_timerange, "ADA/BTC", strategy, chainai.dk, data_load_timerange
    )

    model = chainai.dd.load_data("ADA/BTC", chainai.dk)

    plot_feature_importance(model, "ADA/BTC", chainai.dk)

    assert Path(chainai.dk.data_path / f"{chainai.dk.model_filename}.html")

    shutil.rmtree(Path(chainai.dk.full_path))


@pytest.mark.parametrize(
    "timeframes,corr_pairs",
    [
        (["5m"], ["ADA/BTC", "DASH/BTC"]),
        (["5m"], ["ADA/BTC", "DASH/BTC", "ETH/USDT"]),
        (["5m", "15m"], ["ADA/BTC", "DASH/BTC", "ETH/USDT"]),
    ],
)
def test_chainai_informative_pairs(mocker, chainai_conf, timeframes, corr_pairs):
    chainai_conf["chainai"]["feature_parameters"].update(
        {
            "include_timeframes": timeframes,
            "include_corr_pairlist": corr_pairs,
        }
    )
    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    pairlists = PairListManager(exchange, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange, pairlists)
    pairlist = strategy.dp.current_whitelist()

    pairs_a = strategy.informative_pairs()
    assert len(pairs_a) == 0
    pairs_b = strategy.gather_informative_pairs()
    # we expect unique pairs * timeframes
    assert len(pairs_b) == len(set(pairlist + corr_pairs)) * len(timeframes)


def test_start_set_train_queue(mocker, chainai_conf, caplog):
    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    pairlist = PairListManager(exchange, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange, pairlist)
    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.live = False

    chainai.train_queue = chainai._set_train_queue()

    assert log_has_re(
        "Set fresh train queue from whitelist.",
        caplog,
    )


def test_get_required_data_timerange(mocker, chainai_conf):
    time_range = get_required_data_timerange(chainai_conf)
    assert (time_range.stopts - time_range.startts) == 177300


def test_download_all_data_for_training(mocker, chainai_conf, caplog, tmp_path):
    caplog.set_level(logging.DEBUG)
    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    pairlist = PairListManager(exchange, chainai_conf)
    strategy.dp = DataProvider(chainai_conf, exchange, pairlist)
    chainai_conf["pairs"] = chainai_conf["exchange"]["pair_whitelist"]
    chainai_conf["datadir"] = tmp_path
    download_all_data_for_training(strategy.dp, chainai_conf)

    assert log_has_re(
        "Downloading",
        caplog,
    )


@pytest.mark.usefixtures("init_persistence")
@pytest.mark.parametrize("dp_exists", [(False), (True)])
def test_get_state_info(mocker, chainai_conf, dp_exists, caplog, tickers):
    if is_mac():
        pytest.skip("Reinforcement learning module not available on intel based Mac OS")

    chainai_conf.update({"chainaimodel": "ReinforcementLearner"})
    chainai_conf.update({"timerange": "20180110-20180130"})
    chainai_conf.update({"strategy": "chainai_rl_test_strat"})
    chainai_conf = make_rl_config(chainai_conf)
    chainai_conf["entry_pricing"]["price_side"] = "same"
    chainai_conf["exit_pricing"]["price_side"] = "same"

    strategy = get_patched_chainai_strategy(mocker, chainai_conf)
    exchange = get_patched_exchange(mocker, chainai_conf)
    ticker_mock = MagicMock(return_value=tickers()["ETH/BTC"])
    mocker.patch(f"{EXMS}.fetch_ticker", ticker_mock)
    strategy.dp = DataProvider(chainai_conf, exchange)

    if not dp_exists:
        strategy.dp._exchange = None

    strategy.chainai_info = chainai_conf.get("chainai", {})
    chainai = strategy.chainai
    chainai.data_provider = strategy.dp
    chainai.live = True

    Trade.use_db = True
    create_mock_trades(MagicMock(return_value=0.0025), False, True)
    chainai.get_state_info("ADA/BTC")
    chainai.get_state_info("ETH/BTC")

    if not dp_exists:
        assert log_has_re(
            "No exchange available",
            caplog,
        )

# pragma pylint: disable=missing-docstring

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from chaindrift.commands import Arguments
from chaindrift.enums import State
from chaindrift.exceptions import ConfigurationError, ChaindriftException, OperationalException
from chaindrift.chaindriftbot import ChaindriftBot
from chaindrift.main import main
from chaindrift.worker import Worker
from tests.conftest import (
    log_has,
    log_has_re,
    patch_exchange,
    patched_configuration_load_config_file,
)


def test_parse_args_None(caplog) -> None:
    with pytest.raises(SystemExit):
        main([])
    assert log_has_re(r"Usage of Chaindrift requires a subcommand.*", caplog)


def test_parse_args_backtesting(mocker) -> None:
    """
    Test that main() can start backtesting and also ensure we can pass some specific arguments
    further argument parsing is done in test_arguments.py
    """
    mocker.patch.object(Path, "is_file", MagicMock(side_effect=[False, True]))
    backtesting_mock = mocker.patch("chaindrift.commands.start_backtesting")
    backtesting_mock.__name__ = PropertyMock("start_backtesting")
    # it's sys.exit(0) at the end of backtesting
    with pytest.raises(SystemExit):
        main(["backtesting"])
    assert backtesting_mock.call_count == 1
    call_args = backtesting_mock.call_args[0][0]
    assert call_args["config"] == ["config.json"]
    assert call_args["verbosity"] == 0
    assert call_args["command"] == "backtesting"
    assert call_args["func"] is not None
    assert callable(call_args["func"])
    assert call_args["timeframe"] is None


def test_main_start_hyperopt(mocker) -> None:
    mocker.patch.object(Path, "is_file", MagicMock(side_effect=[False, True]))
    hyperopt_mock = mocker.patch("chaindrift.commands.start_hyperopt", MagicMock())
    hyperopt_mock.__name__ = PropertyMock("start_hyperopt")
    # it's sys.exit(0) at the end of hyperopt
    with pytest.raises(SystemExit):
        main(["hyperopt"])
    assert hyperopt_mock.call_count == 1
    call_args = hyperopt_mock.call_args[0][0]
    assert call_args["config"] == ["config.json"]
    assert call_args["verbosity"] == 0
    assert call_args["command"] == "hyperopt"
    assert call_args["func"] is not None
    assert callable(call_args["func"])


def test_main_fatal_exception(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch("chaindrift.chaindriftbot.ChaindriftBot.cleanup", MagicMock())
    mocker.patch("chaindrift.worker.Worker._worker", MagicMock(side_effect=Exception))
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("chaindrift.chaindriftbot.RPCManager", MagicMock())
    mocker.patch("chaindrift.chaindriftbot.init_db", MagicMock())

    args = ["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"]

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has("Using config: tests/testdata/testconfigs/main_test_config.json ...", caplog)
    assert log_has("Fatal exception!", caplog)


def test_main_keyboard_interrupt(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch("chaindrift.chaindriftbot.ChaindriftBot.cleanup", MagicMock())
    mocker.patch("chaindrift.worker.Worker._worker", MagicMock(side_effect=KeyboardInterrupt))
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("chaindrift.chaindriftbot.RPCManager", MagicMock())
    mocker.patch("chaindrift.wallets.Wallets.update", MagicMock())
    mocker.patch("chaindrift.chaindriftbot.init_db", MagicMock())

    args = ["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"]

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has("Using config: tests/testdata/testconfigs/main_test_config.json ...", caplog)
    assert log_has("SIGINT received, aborting ...", caplog)


def test_main_operational_exception(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch("chaindrift.chaindriftbot.ChaindriftBot.cleanup", MagicMock())
    mocker.patch(
        "chaindrift.worker.Worker._worker", MagicMock(side_effect=ChaindriftException("Oh snap!"))
    )
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("chaindrift.wallets.Wallets.update", MagicMock())
    mocker.patch("chaindrift.chaindriftbot.RPCManager", MagicMock())
    mocker.patch("chaindrift.chaindriftbot.init_db", MagicMock())

    args = ["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"]

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has("Using config: tests/testdata/testconfigs/main_test_config.json ...", caplog)
    assert log_has("Oh snap!", caplog)


def test_main_operational_exception1(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch(
        "chaindrift.exchange.list_available_exchanges",
        MagicMock(side_effect=ValueError("Oh snap!")),
    )
    patched_configuration_load_config_file(mocker, default_conf)

    args = ["list-exchanges"]

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)

    assert log_has("Fatal exception!", caplog)
    assert not log_has_re(r"SIGINT.*", caplog)
    mocker.patch(
        "chaindrift.exchange.list_available_exchanges",
        MagicMock(side_effect=KeyboardInterrupt),
    )
    with pytest.raises(SystemExit):
        main(args)

    assert log_has_re(r"SIGINT.*", caplog)


def test_main_ConfigurationError(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch(
        "chaindrift.exchange.list_available_exchanges",
        MagicMock(side_effect=ConfigurationError("Oh snap!")),
    )
    patched_configuration_load_config_file(mocker, default_conf)

    args = ["list-exchanges"]

    # Test Main + the KeyboardInterrupt exception
    with pytest.raises(SystemExit):
        main(args)
    assert log_has_re("Configuration error: Oh snap!", caplog)


def test_main_reload_config(mocker, default_conf, caplog) -> None:
    patch_exchange(mocker)
    mocker.patch("chaindrift.chaindriftbot.ChaindriftBot.cleanup", MagicMock())
    # Simulate Running, reload, running workflow
    worker_mock = MagicMock(
        side_effect=[
            State.RUNNING,
            State.RELOAD_CONFIG,
            State.RUNNING,
            OperationalException("Oh snap!"),
        ]
    )
    mocker.patch("chaindrift.worker.Worker._worker", worker_mock)
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("chaindrift.wallets.Wallets.update", MagicMock())
    reconfigure_mock = mocker.patch("chaindrift.worker.Worker._reconfigure", MagicMock())

    mocker.patch("chaindrift.chaindriftbot.RPCManager", MagicMock())
    mocker.patch("chaindrift.chaindriftbot.init_db", MagicMock())

    args = Arguments(
        ["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"]
    ).get_parsed_arg()
    worker = Worker(args=args, config=default_conf)
    with pytest.raises(SystemExit):
        main(["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"])

    assert log_has("Using config: tests/testdata/testconfigs/main_test_config.json ...", caplog)
    assert worker_mock.call_count == 4
    assert reconfigure_mock.call_count == 1
    assert isinstance(worker.chaindrift, ChaindriftBot)


def test_reconfigure(mocker, default_conf) -> None:
    patch_exchange(mocker)
    mocker.patch("chaindrift.chaindriftbot.ChaindriftBot.cleanup", MagicMock())
    mocker.patch(
        "chaindrift.worker.Worker._worker", MagicMock(side_effect=OperationalException("Oh snap!"))
    )
    mocker.patch("chaindrift.wallets.Wallets.update", MagicMock())
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch("chaindrift.chaindriftbot.RPCManager", MagicMock())
    mocker.patch("chaindrift.chaindriftbot.init_db", MagicMock())

    args = Arguments(
        ["trade", "-c", "tests/testdata/testconfigs/main_test_config.json"]
    ).get_parsed_arg()
    worker = Worker(args=args, config=default_conf)
    chaindrift = worker.chaindrift

    # Renew mock to return modified data
    conf = deepcopy(default_conf)
    conf["stake_amount"] += 1
    patched_configuration_load_config_file(mocker, conf)

    worker._config = conf
    # reconfigure should return a new instance
    worker._reconfigure()
    chaindrift2 = worker.chaindrift

    # Verify we have a new instance with the new config
    assert chaindrift is not chaindrift2
    assert chaindrift.config["stake_amount"] + 1 == chaindrift2.config["stake_amount"]

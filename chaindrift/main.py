#!/usr/bin/env python3
"""
Main Chaindrift bot script.
Read the documentation to know what cli arguments you need.
"""

import logging
import sys
from typing import Any


# check min. python version
if sys.version_info < (3, 10):  # pragma: no cover  # noqa: UP036
    sys.exit("Chaindrift requires Python version >= 3.10")

from chaindrift import __version__
from chaindrift.commands import Arguments
from chaindrift.constants import DOCS_LINK
from chaindrift.exceptions import ConfigurationError, ChaindriftException, OperationalException
from chaindrift.loggers import setup_logging_pre
from chaindrift.system import asyncio_setup, gc_set_threshold


logger = logging.getLogger("chaindrift")


def main(sysargv: list[str] | None = None) -> None:
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """

    return_code: Any = 1
    try:
        setup_logging_pre()
        asyncio_setup()
        arguments = Arguments(sysargv)
        args = arguments.get_parsed_arg()

        # Call subcommand.
        if "func" in args:
            logger.info(f"chaindrift {__version__}")
            gc_set_threshold()
            return_code = args["func"](args)
        else:
            # No subcommand was issued.
            raise OperationalException(
                "Usage of Chaindrift requires a subcommand to be specified.\n"
                "To have the bot executing trades in live/dry-run modes, "
                "depending on the value of the `dry_run` setting in the config, run Chaindrift "
                "as `chaindrift trade [options...]`.\n"
                "To see the full list of options available, please use "
                "`chaindrift --help` or `chaindrift <command> --help`."
            )

    except SystemExit as e:  # pragma: no cover
        return_code = e
    except KeyboardInterrupt:
        logger.info("SIGINT received, aborting ...")
        return_code = 0
    except ConfigurationError as e:
        logger.error(
            f"Configuration error: {e}\n"
            f"Please make sure to review the documentation at {DOCS_LINK}."
        )
    except ChaindriftException as e:
        logger.error(str(e))
        return_code = 2
    except Exception:
        logger.exception("Fatal exception!")
    finally:
        sys.exit(return_code)


if __name__ == "__main__":  # pragma: no cover
    main()

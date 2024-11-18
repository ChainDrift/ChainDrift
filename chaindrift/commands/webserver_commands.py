from typing import Any

from chaindrift.enums import RunMode


def start_webserver(args: dict[str, Any]) -> None:
    """
    Main entry point for webserver mode
    """
    from chaindrift.configuration import setup_utils_configuration
    from chaindrift.rpc.api_server import ApiServer

    # Initialize configuration

    config = setup_utils_configuration(args, RunMode.WEBSERVER)
    ApiServer(config, standalone=True)

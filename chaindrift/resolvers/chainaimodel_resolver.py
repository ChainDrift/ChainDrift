# pragma pylint: disable=attribute-defined-outside-init

"""
This module load a custom model for chainai
"""

import logging
from pathlib import Path

from chaindrift.constants import USERPATH_CHAINAIMODELS, Config
from chaindrift.exceptions import OperationalException
from chaindrift.chainai.chainai_interface import IChainaiModel
from chaindrift.resolvers import IResolver


logger = logging.getLogger(__name__)


class ChainaiModelResolver(IResolver):
    """
    This class contains all the logic to load custom hyperopt loss class
    """

    object_type = IChainaiModel
    object_type_str = "ChainaiModel"
    user_subdir = USERPATH_CHAINAIMODELS
    initial_search_path = (
        Path(__file__).parent.parent.joinpath("chainai/prediction_models").resolve()
    )
    extra_path = "chainaimodel_path"

    @staticmethod
    def load_chainaimodel(config: Config) -> IChainaiModel:
        """
        Load the custom class from config parameter
        :param config: configuration dictionary
        """
        disallowed_models = ["BaseRegressionModel"]

        chainaimodel_name = config.get("chainaimodel")
        if not chainaimodel_name:
            raise OperationalException(
                "No chainaimodel set. Please use `--chainaimodel` to "
                "specify the ChainaiModel class to use.\n"
            )
        if chainaimodel_name in disallowed_models:
            raise OperationalException(
                f"{chainaimodel_name} is a baseclass and cannot be used directly. Please choose "
                "an existing child class or inherit from this baseclass.\n"
            )
        chainaimodel = ChainaiModelResolver.load_object(
            chainaimodel_name,
            config,
            kwargs={"config": config},
        )

        return chainaimodel

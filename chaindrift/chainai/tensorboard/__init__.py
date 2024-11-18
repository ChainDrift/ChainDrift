# ensure users can still use a non-torch chainai version
try:
    from chaindrift.chainai.tensorboard.tensorboard import TensorBoardCallback, TensorboardLogger

    TBLogger = TensorboardLogger
    TBCallback = TensorBoardCallback
except ModuleNotFoundError:
    from chaindrift.chainai.tensorboard.base_tensorboard import (
        BaseTensorBoardCallback,
        BaseTensorboardLogger,
    )

    TBLogger = BaseTensorboardLogger  # type: ignore
    TBCallback = BaseTensorBoardCallback  # type: ignore

__all__ = ("TBLogger", "TBCallback")

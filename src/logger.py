from tensorboardX import SummaryWriter


class TensorboardLogger:

    def __init__(self):
        self.writer = SummaryWriter()

    def __del__(self) -> None:
        self.writer.close()

    def write(
        self,
        stats: dict[str, float],
        iteration: int,
    ) -> None:
        for name, value in stats.items():
            self.writer.add_scalar(
                tag=name,
                scalar_value=value,
                global_step=iteration,
            )

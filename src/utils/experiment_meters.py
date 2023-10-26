from src.utils.meters import AverageMeter


class EvalMeters:
    """
    Struct to hold the meters used in the validation loop. Since we are just
    estimating the expectations with a frozen model, we do not need EMA meters.
    """

    def __init__(self) -> None:
        self.avg_loss = AverageMeter()
        self.group_loss = AverageMeter()
        self.avg_acc = AverageMeter()
        self.group_acc = AverageMeter()


class TrainMeters:
    """
    Struct to hold the meters used in the training loop. The created meters
    depend on the type of CMP. Recall that surrogate metrics do not need EMA meters
    since we only use the surrogate metrics for their minibatch-level gradient.
    """

    def __init__(self, cmp_class_name: str) -> None:
        self.avg_loss = AverageMeter()
        self.group_loss = AverageMeter()
        self.avg_acc = AverageMeter()
        self.group_acc = AverageMeter()

        if cmp_class_name in {"BaselineProblem", "EqualizedLossProblem"}:
            pass
        elif cmp_class_name in {"UniformAccuracyGapProblem", "EqualizedAccuracyGapProblem"}:
            self.avg_soft_acc = AverageMeter()
            self.group_soft_acc = AverageMeter()
        else:
            raise ValueError(f"Unknown CMP class name: {cmp_class_name}")

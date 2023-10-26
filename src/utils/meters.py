class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count


class BestMeter:
    """Computes and stores the best observed value of a metric."""

    def __init__(self, direction="max"):
        assert direction in {"max", "min"}
        self.direction = direction
        self.reset()

    def reset(self):
        if self.direction == "max":
            self.val = -float("inf")
        else:
            self.val = float("inf")

    def update(self, val):
        """Update meter and return boolean flag indicating if the current value is
        the best so far."""

        if self.direction == "max":
            if val > self.val:
                self.val = val
                return True
        elif self.direction == "min":
            if val < self.val:
                self.val = val
                return True

        return False

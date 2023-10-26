from typing import Literal


class SparsityScheduler(object):
    """
    Sparsity scheduler for iterative magnitude pruning introduced in the paper: "To
    prune, or not to prune: exploring the efficacy of pruning for model compression".
    https://arxiv.org/abs/1710.01878

    If you want to prune every epoch, through epochs [0, 17] inclusive, prescribe:
    init_pruning_epoch=0, last_pruning_epoch=17, pruning_frequency=1.

    Given pruning_frequency=2, sparsity levels are updates on epochs
    0, 2, 4, ... 16 *and* 17. Note that 0 and 17 are always included.

    To perform one shot pruning, indicate the desired epoch for pruning in
    init_pruning_epoch *and* set last_pruning_epoch=init_pruning_epoch.
    Moreover, set sparsity_initial=sparsity_final.

    """

    def __init__(
        self,
        last_pruning_epoch: int,
        sparsity_final: float,
        sparsity_initial: float = 0.0,
        init_pruning_epoch: int = 0,
        pruning_frequency: int = 1,
        sparsity_type: Literal["unstructured", "structured"] = "unstructured",
    ):
        """
        Args:
            last_pruning_epoch: last epoch for performing pruning. After this,
                the should_sparsify method will always return False.
            sparsity_final: final sparsity level for the model
            sparsity_initial: initial sparsity level.
            init_pruning_epoch: represents the epoch when sparsity is first considered.
                Defaults to 0.
            pruning_frequency: how often to prune the model measured in epochs.
                Paper refers to this as Delta_t.
        """

        assert 1.0 >= sparsity_final >= sparsity_initial >= 0.0
        assert last_pruning_epoch >= init_pruning_epoch >= 0
        assert pruning_frequency > 0
        assert sparsity_type in ["unstructured", "structured"]

        if last_pruning_epoch == init_pruning_epoch and sparsity_initial != sparsity_final:
            raise ValueError("One-shot pruning requires sparsity_initial == sparsity_final.")

        self.last_pruning_epoch = last_pruning_epoch
        self.sparsity_final = sparsity_final
        self.pruning_frequency = pruning_frequency
        self.sparsity_initial = sparsity_initial
        self.init_pruning_epoch = init_pruning_epoch
        self.sparsity_type = sparsity_type

    def sparsity_amount(self, epoch: int) -> float:
        """
        Calculate current sparsity based on a cubic interpolation between
        `sparsity_initial` and `sparsity_final`. We use the formula provided in
        https://arxiv.org/abs/1710.01878 (page 3 Eq. 1).
        """

        relative_epoch = epoch - self.init_pruning_epoch
        relative_total_epochs = self.last_pruning_epoch - self.init_pruning_epoch

        # At the first and last pruning epochs, the cubic formula can be
        # inaccurate due to floating point arithmetic. Simply return
        # sparsity_initial or sparsity_final
        if relative_epoch <= 0:
            return self.sparsity_initial
        elif relative_epoch >= relative_total_epochs:
            return self.sparsity_final

        progress_prop = relative_epoch / relative_total_epochs
        sparsity_diff = self.sparsity_initial - self.sparsity_final

        return self.sparsity_final + sparsity_diff * (1.0 - progress_prop) ** 3

    def should_sparsify(self, epoch: int) -> bool:
        """
        Whether to sparsify in the current epoch.
        """
        if epoch < self.init_pruning_epoch:
            return False
        if epoch > self.last_pruning_epoch:
            return False

        if epoch == self.init_pruning_epoch or epoch == self.last_pruning_epoch:
            # Always sparsify at the first and last epochs
            return True

        relative_epoch = epoch - self.init_pruning_epoch
        if relative_epoch % self.pruning_frequency == 0:
            return True

        return False

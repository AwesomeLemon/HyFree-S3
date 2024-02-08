from torch.optim.lr_scheduler import _LRScheduler


class PolyLRWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.warmup_steps = int(max_steps * 0.1)  # 10% of total steps for warmup
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if current_step < self.warmup_steps:  # Linear warmup
            new_lr = self.initial_lr * (current_step / self.warmup_steps)
        else:  # Polynomial decay
            new_lr = self.initial_lr * (1 - (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)) ** self.exponent

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt

    # Assuming the PolyLRScheduler class is already defined as per your previous request

    # Create a simple model with a single parameter
    model = torch.nn.Linear(1, 1)

    # Use a real PyTorch optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Initialize the scheduler
    scheduler = PolyLRWarmupScheduler(optimizer, initial_lr=0.01, max_steps=1000)


    def plot_lr_schedule(scheduler, max_steps):
        lrs = []
        for step in range(max_steps):
            scheduler.step(step)
            lrs.append(scheduler.optimizer.param_groups[0]['lr'])

        plt.plot(lrs)
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.show()


    # Plot the learning rate schedule
    plot_lr_schedule(scheduler, 1000)
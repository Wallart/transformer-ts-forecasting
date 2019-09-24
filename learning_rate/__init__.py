from learning_rate.schedules import *


def apply_lr_schedule(schedule_fn, iterations=1e5):
    # iteration count starting at 1
    iterations = [i+1 for i in range(int(iterations))]
    lrs = [schedule_fn(i) for i in iterations]
    return lrs

class TransformerSchedule:

    def __init__(self, d_model, warmup_steps=4000):
        self._d_model = d_model
        self._warmup_steps = warmup_steps

    def __call__(self, step_num):
        return (self._d_model ** -.5) * min((step_num ** -.5), step_num * (self._warmup_steps ** -1.5))


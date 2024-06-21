class LBFGSAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, history_size=10):
        defaults = dict(lr=lr, history_size=history_size)
        super(LBFGSAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            history_size = group['history_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['s'] = []
                    state['y'] = []
                    state['prev_p'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = 0.9, 0.999

                state['step'] += 1

                # Adam part
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                direction = -step_size * exp_avg / (exp_avg_sq.sqrt() + 1e-8)
                p.data.add_(direction)
                
                # LBFGS part
                if len(state['s']) == history_size:
                    state['s'].pop(0)
                    state['y'].pop(0)

                state['s'].append(p.data - state['prev_p'])
                state['y'].append(grad - state['prev_grad'])

                state['prev_p'].copy_(p.data)
                state['prev_grad'].copy_(grad)

        return loss

class LBFGSAdam(Optimizer):
    def __init__(self, params, lr=1e-3, p1=0.9, p2=0.999):
        defaults = dict(lr=lr, p1=p1, p2=p2)
        super(LBFGSAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['r'] = torch.zeros_like(p.data)
                    state['u'] = torch.zeros_like(p.data)

                r, u = state['r'], state['u']
                p1, p2, lr = group['p1'], group['p2'], group['lr']

                state['step'] += 1
                r.mul_(p1).add_(grad, alpha=1 - p1)
                u.mul_(p2).add_(grad, alpha=1 - p2)

                direction = self.lbfgs_direction(r, u)
                p.data.add_(direction, alpha=-lr)

        return loss

    def lbfgs_direction(self, r, u):
        # 实现LBFGS特定的计算
        return r + u

class LBFGSAdam(Optimizer):
    def __init__(self, params, lr=1e-5, betas=(0.1, 0.999), eps=1e-8, history_size=10, max_grad_norm=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, history_size=history_size, max_grad_norm=max_grad_norm)
        super(LBFGSAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            history_size = group['history_size']
            max_grad_norm = group['max_grad_norm']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化状态字典
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['old_dirs'] = []
                    state['old_stps'] = []

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas

                state['step'] += 1

                # 计算y_k和s_k，并存储到历史记录中
                if state['step'] > 1:
                    y = grad - state['prev_grad']
                    s = p.data - state['prev_p_data']
                    state['old_dirs'].append(y)
                    state['old_stps'].append(s)

                    if len(state['old_dirs']) > history_size:
                        state['old_dirs'].pop(0)
                        state['old_stps'].pop(0)

                # LBFGS前向循环：计算alpha和q
                q = grad.view(-1)
                alphas = []
                for i in range(len(state['old_dirs']) - 1, -1, -1):
                    s, y = state['old_stps'][i].view(-1), state['old_dirs'][i].view(-1)
                    alpha = s.dot(q) / (y.dot(s) + eps)  # 添加eps以防止数值不稳定
                    q -= alpha * y
                    alphas.append(alpha)

                # 初始设置r
                r = q
                if len(state['old_dirs']) > 0:
                    s, y = state['old_stps'][-1].view(-1), state['old_dirs'][-1].view(-1)
                    r *= y.dot(s) / (y.dot(y) + eps)  # 添加eps以防止数值不稳定

                # LBFGS后向循环：计算beta并更新r
                for i in range(len(state['old_dirs'])):
                    s, y = state['old_stps'][i].view(-1), state['old_dirs'][i].view(-1)
                    beta = y.dot(r) / (s.dot(y) + eps)  # 添加eps以防止数值不稳定
                    r += s * (alphas.pop() - beta)

                r = r.view_as(grad)

                # 利用LBFGS得到的r代入Adam框架
                exp_avg.mul_(beta1).add_(r, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(r, r, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr / denom

                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_([p], max_grad_norm)

                with torch.no_grad():
                    p.data.add_(-step_size * exp_avg)

                # 保存当前梯度和参数
                state['prev_grad'] = grad.clone()
                state['prev_p_data'] = p.data.clone()

        return loss

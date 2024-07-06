import torch
from torch.optim.optimizer import Optimizer

class LBFGSAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, history_size=10):
        """
        初始化LBFGSAdam优化器。
        参数:
        - params: 被优化的参数。
        - lr: 学习率。
        - betas: Adam优化器的两个超参数。
        - eps: 防止除零的小量。
        - history_size: LBFGS方法的历史记录大小。
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, history_size=history_size)
        super(LBFGSAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        执行一步优化。
        参数:
        - closure: 一个返回损失的函数，用于重新计算梯度。
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            history_size = group['history_size']

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
                    alpha = s.dot(q) / y.dot(s)
                    q -= alpha * y
                    alphas.append(alpha)

                # 初始设置r
                r = q
                if len(state['old_dirs']) > 0:
                    s, y = state['old_stps'][-1].view(-1), state['old_dirs'][-1].view(-1)
                    r *= y.dot(s) / y.dot(y)

                # LBFGS后向循环：计算beta并更新r
                for i in range(len(state['old_dirs'])):
                    s, y = state['old_stps'][i].view(-1), state['old_dirs'][i].view(-1)
                    beta = y.dot(r) / s.dot(y)
                    r += s * (alphas.pop() - beta)

                r = r.view_as(grad)

                # 利用LBFGS得到的r代入Adam框架
                exp_avg.mul_(beta1).add_(r, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(r, r, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr / denom

                p.data.add_(-step_size * exp_avg)

                # 保存当前梯度和参数
                state['prev_grad'] = grad.clone()
                state['prev_p_data'] = p.data.clone()

        return loss

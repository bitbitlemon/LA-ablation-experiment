#不用看了不用看了
#添加了牛顿向前向后更新
import torch
from torch.optim.optimizer import Optimizer

class LBFGSAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, history_size=10):
        defaults = dict(lr=lr, betas=betas, eps=eps, history_size=history_size)
        super(LBFGSAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']  # 学习率
            betas = group['betas']  # Adam优化器的beta1和beta2参数
            eps = group['eps']  # 为了数值稳定性的小常数
            history_size = group['history_size']  # LBFGS历史记录大小

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data  # 获取参数的梯度
                state = self.state[p]

                if len(state) == 0:
                    # 初始化状态字典
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # 一阶矩估计
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # 二阶矩估计
                    state['old_dirs'] = []  # 保存s_i
                    state['old_stps'] = []  # 保存y_i

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas

                state['step'] += 1

                # 更新一阶矩估计
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # 更新二阶矩估计
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算分母并添加小常数以防止除零
                denom = exp_avg_sq.sqrt().add_(eps)

                if len(state['old_dirs']) > history_size:
                    # 超过历史记录大小时，移除最旧的记录
                    state['old_dirs'].pop(0)
                    state['old_stps'].pop(0)

                if state['step'] > 1:
                    # 计算y_k和s_k
                    y = grad - state['prev_grad']
                    s = p.data - state['prev_p_data']

                    state['old_dirs'].append(y)
                    state['old_stps'].append(s)

                if state['step'] > 1:
                    # LBFGS前向循环：计算alpha
                    q = grad.view(-1)
                    alphas = []
                    for i in range(len(state['old_dirs']) - 1, -1, -1):
                        s, y = state['old_stps'][i].view(-1), state['old_dirs'][i].view(-1)
                        alpha = s.dot(q) / y.dot(s)
                        q = q - alpha * y
                        alphas.append(alpha)

                    # 计算r
                    r = torch.mul(q, torch.dot(s, y) / torch.dot(y, y))

                    # LBFGS后向循环：计算beta并更新r
                    for i in range(len(state['old_dirs'])):
                        s, y = state['old_stps'][i].view(-1), state['old_dirs'][i].view(-1)
                        beta = y.dot(r) / s.dot(y)
                        r = r + s * (alphas[i] - beta)

                # 更新参数：元素对应的乘法（Hadamard乘积）
                p.data.addcdiv_(-lr, exp_avg, denom)

                # 保存当前梯度和参数，用于下一步计算y和s
                state['prev_grad'] = grad.clone()
                state['prev_p_data'] = p.data.clone()

        return loss

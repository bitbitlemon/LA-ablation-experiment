import torch
from torch.optim.optimizer import Optimizer

class LBFGSAdam(Optimizer):
    def __init__(self, params, lr=1e-3, history_size=10, rho1=0.9, rho2=0.999, epsilon=1e-8):
        # 初始化优化器参数
        defaults = dict(lr=lr, history_size=history_size, rho1=rho1, rho2=rho2, epsilon=epsilon)
        super(LBFGSAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        # 如果提供closure，则在每次迭代中执行前向和反向传递计算loss
        loss = None
        if closure is not None:
            loss = closure()

        # 遍历参数组
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data  # 获取参数的梯度
                state = self.state[p]  # 获取参数的状态

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # 初始化一阶动量
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # 初始化二阶动量

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                rho1, rho2, lr, epsilon = group['rho1'], group['rho2'], group['lr'], group['epsilon']

                # 计算一阶动量的指数加权平均
                exp_avg.mul_(rho1).add_(1 - rho1, grad)
                
                # 计算二阶动量的指数加权平均
                exp_avg_sq.mul_(rho2).addcmul_(1 - rho2, grad, grad)

                # 计算修正后的二阶动量
                denom = exp_avg_sq.sqrt().add_(epsilon)

                # 计算学习率
                step_size = lr / denom

                # 更新参数
                p.data.add_(-step_size * exp_avg)

        return loss

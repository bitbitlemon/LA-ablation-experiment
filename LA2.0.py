import torch
from torch.optim import Optimizer

class LBFGSAdam(Optimizer):
    def __init__(self, params, lr=1e-3, p1=0.9, p2=0.999, history_size=10):
        """
        初始化LBFGSAdam优化器。

        参数:
        - params: 需要优化的参数。
        - lr: 学习率，默认为1e-3。
        - p1: 第一个动量系数，默认为0.9。
        - p2: 第二个动量系数，默认为0.999。
        - history_size: 存储历史步长和梯度信息的数量，默认为10。
        """
        defaults = dict(lr=lr, p1=p1, p2=p2, history_size=history_size)
        super(LBFGSAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        执行一步优化。

        参数:
        - closure: 一个返回当前损失的可调用对象。

        返回:
        - loss: 当前的损失值。
        """
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
                    # 初始化状态
                    state['step'] = 0
                    state['r'] = torch.zeros_like(p.data)
                    state['u'] = torch.zeros_like(p.data)
                    state['s_history'] = []
                    state['y_history'] = []
                    state['rho_history'] = []

                r, u = state['r'], state['u']
                p1, p2, lr = group['p1'], group['p2'], group['lr']
                history_size = group['history_size']

                # 更新步数
                state['step'] += 1
                # 更新r和u
                r.mul_(p1).add_(grad, alpha=1 - p1)
                u.mul_(p2).add_(grad, alpha=1 - p2)

                if state['step'] > 1:
                    # 计算步长s和梯度差异y
                    prev_p_data = state['prev_p_data']
                    s = p.data - prev_p_data
                    y = grad - state['prev_grad']
                    # 计算rho
                    rho = 1.0 / y.dot(s)
                    # 管理历史信息的大小
                    if len(state['s_history']) >= history_size:
                        state['s_history'].pop(0)
                        state['y_history'].pop(0)
                        state['rho_history'].pop(0)
                    # 保存当前步长和梯度差异
                    state['s_history'].append(s)
                    state['y_history'].append(y)
                    state['rho_history'].append(rho)

                # 更新前保存当前参数p和梯度grad
                state['prev_p_data'] = p.data.clone()
                state['prev_grad'] = grad.clone()

                # 计算更新方向
                direction = self.lbfgs_direction(r, u, state)
                # 根据方向更新参数
                p.data.add_(direction, alpha=-lr)

        return loss

    def lbfgs_direction(self, r, u, state):
        """
        使用LBFGS算法计算更新方向。

        参数:
        - r: 第一个动量。
        - u: 第二个动量。
        - state: 当前参数的状态字典。

        返回:
        - r: 更新方向。
        """
        # 初始化q为r和u之和
        q = r + u
        s_history = state['s_history']
        y_history = state['y_history']
        rho_history = state['rho_history']

        alpha = []
        # 反向遍历历史信息，计算alpha值并更新q
        for s, y, rho in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
            alpha_i = rho * s.dot(q)
            alpha.append(alpha_i)
            q = q - alpha_i * y

        # 初始化r为q
        r = q
        # 正向遍历历史信息，计算beta值并更新r
        for s, y, rho, alpha_i in zip(s_history, y_history, rho_history, reversed(alpha)):
            beta = rho * y.dot(r)
            r = r + s * (alpha_i - beta)

        return r

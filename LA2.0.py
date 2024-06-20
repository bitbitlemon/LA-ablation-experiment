import torch
import torch.optim as optim

class LBFGSAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, history_size=10):
        # 初始化优化器，设置学习率和历史记录大小
        defaults = dict(lr=lr, history_size=history_size)
        super(LBFGSAdam, self).__init__(params, defaults)

    def _gather_flat_grad(self):
        # 收集所有参数的梯度并展平
        views = []
        for p in self.param_groups[0]['params']:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self):
        # 收集所有参数并展平
        views = []
        for p in self.param_groups[0]['params']:
            view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _set_flat_params(self, flat_params):
        # 将展平的参数重新设置回模型参数
        offset = 0
        for p in self.param_groups[0]['params']:
            numel = p.data.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
            offset += numel
     #根据公式计算 r_t
    def _calculate_rt(self, flat_params):
        # 计算 r_t = (∇^2 f( x_t ))^(-1) * ∇f( x_t )
        grad = self._gather_flat_grad()
        if 'prev_flat_grad' not in self.state:
            self.state['prev_flat_grad'] = torch.zeros_like(grad)
        return grad - self.state['prev_flat_grad']

    def _forward_pass(self, m, q):
        # 前向传递
        for i in range(m):
            si = self.state['s'][i]
            yi = self.state['y'][i]
            alpha = (si.t() @ q) / (yi.t() @ si)
            q -= alpha * yi
            self.state['alpha'].append(alpha)
        return q

    def _reverse_pass(self, m, r):
        # 反向传递
        for i in range(m-1, -1, -1):
            si = self.state['s'][i]
            yi = self.state['y'][i]
            beta = (yi.t() @ r) / (yi.t() @ si)
            r += si * (self.state['alpha'][i] - beta)
        return r

    def step(self, closure):
        # 优化步骤，计算损失并更新参数
        loss = closure()

        flat_grad = self._gather_flat_grad()
        flat_params = self._gather_flat_params()

        if 'prev_flat_grad' not in self.state:
            self.state['prev_flat_grad'] = torch.zeros_like(flat_grad)
            self.state['prev_flat_params'] = torch.zeros_like(flat_params)
            self.state['s'] = []
            self.state['y'] = []
            self.state['alpha'] = []
            self.state['H0'] = 1.0

        rt = self._calculate_rt(flat_params)
        step_size = self.param_groups[0]['lr']

        # 前向和反向传递
        m = len(self.state['s'])
        q = self._forward_pass(m, rt)
        direction = self._reverse_pass(m, q)

        # 更新参数
        flat_params -= step_size * direction
        self._set_flat_params(flat_params)

        # 存储新的 s 和 y 值
        self.state['s'].append(flat_params - self.state['prev_flat_params'])
        self.state['y'].append(flat_grad - self.state['prev_flat_grad'])

        # 维护历史记录大小
        if len(self.state['s']) > self.param_groups[0]['history_size']:
            self.state['s'].pop(0)
            self.state['y'].pop(0)

        self.state['prev_flat_grad'].copy_(flat_grad)
        self.state['prev_flat_params'].copy_(flat_params)

        return loss

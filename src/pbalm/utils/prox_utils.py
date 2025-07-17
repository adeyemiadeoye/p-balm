import numpy as np
from typing import Optional
from proxop import L1Norm

class SGLPenalty:
    """
    Sparse Group Lasso Penalty on variable z = [params; states]:
      gamma1 * ||params||_1 + gamma2 * grouped-L2(params)
    If P is provided, only params (first n entries) are regularized.
    """
    def __init__(self, gamma1: float, gamma2: float, P: Optional[object] = None):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.P = P
        self.l1 = L1Norm()

    def prox(self, z):
        x1 = self.l1.prox(z, self.gamma1)
        if self.P is not None:
            h = np.ones_like(z)
            return self.P.ProxL2(x1, self.gamma2, h)
        return x1

    def __call__(self, z):
        l1val = self.l1(z)
        if self.P is not None:
            glval = self.P.Lasso_fz(z)
        else:
            glval = 0.0
        return self.gamma1 * l1val + self.gamma2 * glval
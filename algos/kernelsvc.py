import numpy as np
# import jax.numpy as np
from scipy import optimize


class KernelSVC:

    def __init__(self, C, kernel, epsilon=1e-3, verbose=False):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        self.verbose = verbose

    def fit(self, X, y):
        if self.verbose:
            print("----- Start training -----")

        N = len(y)

        if self.verbose:
            print("Computing kernel")
        K = self.kernel(X, X) + 0.1 * N * np.eye(N)

        # K = (np.eye(N) - np.ones((N,N))) @ K @ (np.eye(N) - np.ones((N,N)))

        # Lagrange dual problem
        def loss(alpha):
            return (1 / 2) * alpha.T @ K @ alpha - alpha.dot(y)

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return K @ alpha - y

            # alpha.T @ 1 = 0

        fun_eq = lambda alpha: np.sum(alpha)
        jac_eq = lambda alpha: np.ones(N)

        # y alpha <= C
        fun_ineq = lambda alpha: self.C - y * alpha
        jac_ineq = lambda alpha: - np.diag(y)

        # y alpha >= 0
        fun_ineq2 = lambda alpha: y * alpha
        jac_ineq2 = lambda alpha: np.diag(y)

        constraints = ({'type': 'eq', 'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 'fun': fun_ineq, 'jac': jac_ineq},
                       {'type': 'ineq', 'fun': fun_ineq2, 'jac': jac_ineq2})

        if self.verbose:
            print("Optimizing SVC")
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha),
                                   constraints=constraints,
                                   tol=self.epsilon)
        self.alpha = optRes.x
        print(self.alpha)
        ## Assign the required attributes

        self.support = X[np.abs(self.alpha) > 1e-8]  # Non-zero alphas
        print(np.sum(np.abs(self.alpha) > 1e-8))
        # offset of the classifier
        self.b = np.mean((y - K @ self.alpha)[np.logical_and(np.abs(self.alpha) > 1e-8, y * self.alpha < self.C - 1e-8)])
        self.norm_f = np.sqrt(self.alpha.T @ K @ self.alpha)  # RKHS norm of the function f

        y = y[np.abs(self.alpha) > 1e-8]
        self.alpha = self.alpha[np.abs(self.alpha) > 1e-8]  #  Store only the non-zeros alphas

        if self.verbose:
            print("----- Training done -----")

    ### Implementation of the separating function $f$
    def separating_function(self, x):
        K = self.kernel(x, self.support)
        # N, M = len(self.support), len(x)
        # K = (np.eye(M) - np.ones((M,M))) @ K @ (np.eye(N) - np.ones((N,N)))
        return K @ self.alpha

    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        # return 2 * ((d + self.b) > 0) - 1
        return d

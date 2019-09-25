class MovingAverage():
    def __init__(self, alpha=0.9, use_factors=True):
        """
        Bias-corrected moving average (KFAC-style)

        Implements the update `m_{t+1} = (1-𝛼_t) m_t + 𝛼_t x_t` where
        - `𝛼_t = min(1/(t+1), 𝛼)`,
        - `m_t` is the estimate at step `t`
        - `x_t` is the observation at step `t`
        `t` starts at `t=0`.

        If `use_factors` is `False`, the `step` function will expect
        observations of the form
        ```
        new_obs = [        # for each group
            [              # for each parameter in the group
                curv_p_11, # the curvature for that parameter
                ...
            ],
            ...
        ]
        ```
        If `use_factors` is `True`, then it will expect
        ```
        new_obs = [              # for each group
            [                    # for each parameter in the group
                [                # for each Kronecker factor for the parameter
                    kfac_1_p_11, # the Kronecker factor
                    ...
                ],
                ...
            ],
            ...
        ]
        ```
        The `get` function will return the current estimate of the curvature
        in the same format.
        """
        self.alpha = alpha
        self.use_factors = use_factors
        self.step_counter = 0
        self.estimate = None

    def get(self):
        return self.estimate

    def __update(self, old_est, new_obs, alpha):
        if self.use_factors:
            for old_est_factor, new_obs_factors in zip(old_est, new_obs):
                old_est_factor.mul_(1 - alpha).add_(alpha, new_obs_factors)
        else:
            old_est.mul_(1 - alpha).add_(alpha, new_obs)

    def step(self, new_obs):
        """
        If `use_factors` is `False`, the `step` function expects
        ```
        new_obs = [        # for each group
            [              # for each parameter in the group
                curv_p_11, # the curvature for that parameter
                ...
            ],
            ...
        ]
        ```
        If `use_factors` is `True`, then it expects
        ```
        new_obs = [              # for each group
            [                    # for each parameter in the group
                [                # for each Kronecker factor for the parameter
                    kfac_1_p_11, # the Kronecker factor
                    ...
                ],
                ...
            ],
            ...
        ]
        ```
        """
        if self.estimate is None:
            self.estimate = new_obs
            self.step_counter += 1
        else:
            alpha_t = min([1. / (self.step_counter + 1), self.alpha])

            for old_group, new_group in zip(self.estimate, new_obs):
                for old_est, new_obs in zip(old_group, new_group):
                    self.__update(old_est, new_obs, alpha_t)
            self.step_counter += 1
        return self.get()

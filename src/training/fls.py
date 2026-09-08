def reference_lr(gamma, depth=4, base_lr=.05):
    # L counts all affine maps including the readout. This is an LR search
    # center, not a guaranteed optimal rate for standard parameterization.
    return base_lr*(gamma**2 if gamma <= 1 else gamma**(2/(depth+1)))

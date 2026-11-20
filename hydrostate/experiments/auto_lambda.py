"""One-step Auto-Lambda hypergradient for fixed station and weighted product losses."""
import torch


def virtual_sgd_step(parameters, fixed_loss, product_losses, weights, lr):
    """Return virtual parameters differentiable in lambda, without updating model state.

    For a one-step SGD virtual optimizer, d(theta')/d(lambda_i) is exactly
    -lr * grad_theta(L_product_i). Treating those gradients as constants is
    exact for this derivative; it avoids storing encoder second derivatives.
    Actual model updates may use AdamW, as in the frozen screening recipe.
    """
    names = tuple(parameters)
    params = tuple(parameters.values())
    if len(product_losses) != weights.numel() or not weights.requires_grad:
        raise ValueError('Supply one differentiable weight per product group')
    fixed = torch.autograd.grad(fixed_loss, params, allow_unused=True, retain_graph=True)
    products = [
        torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        if loss is not None else (None,) * len(params)
        for loss in product_losses
    ]
    updated = {}
    for i, (name, param) in enumerate(zip(names, params)):
        grad = fixed[i]
        for weight, group in zip(weights, products):
            if group[i] is not None:
                term = weight * group[i].detach()
                grad = term if grad is None else grad + term
        updated[name] = param.detach() if grad is None else param.detach() - lr * grad
    return updated

import torch
import matplotlib.pyplot as plt
loss_fn = {
    "relu": lambda diff, tol: torch.nn.functional.relu(diff - tol),
    "mse": lambda diff, tol: (torch.nn.functional.relu(diff - tol)).pow(2),
}

plt.figure(figsize=(10, 6))
for loss_fn_str in ['relu', 'mse']:
    for tol_ in [0.1, 0.5, 1.0]:
        print(f"Testing {loss_fn_str} with tol={tol_}")
        loss_fn_ = loss_fn[loss_fn_str]
        x = torch.linspace(-5, 5, 500)
        tol = tol_
        y = loss_fn_((x - 0.5).abs(), tol)
        plt.plot(x.numpy(), y.numpy(), label=f"{loss_fn_str} tol={tol_}")
clash_loss = torch.relu(10 * (2 - x))
# clash_loss = torch.relu(1 / (x).abs())
plt.plot(x.numpy(), clash_loss.numpy(), label='Clash Loss')
plt.grid()
plt.ylim(-1, 20)
plt.legend()
# %%

import torch
import os

n = 10  # number of points
R = 256 # batch size
steps = 100_000 # iterations
lr = 0.01   # learning rate
best_val = -1.0 # initialize best value for later
best_config = None  # initialize best configuration for later

points = torch.rand(R, n, 2, requires_grad = True)  # generate points in [0,1)
idx = torch.combinations(torch.arange(n), r = 3)    # enumerate all nCr combinations of [1,...,n]

optimizer = torch.optim.Adam([points], lr = lr)

base = os.path.join("runs", f"n{n}")    # log runs
os.makedirs(base, exist_ok=True)
n_run = 1
while os.path.exists(os.path.join(base, f"run{n_run}")):
    n_run += 1
run_dir = os.path.join(base, f"run{n_run}") 
os.makedirs(run_dir)
log_file = open(os.path.join(run_dir, "log.csv"), "w")
log_file.write("step,best_min_area\n")

for k in range(steps):
    A, B, C = points[:, idx[:, 0]], points[:, idx[:, 1]], points[:, idx[:, 2]] 
    cross = (B[..., 0] - A[..., 0]) * (C[..., 1] - A[..., 1]) - (C[..., 0] - A[..., 0]) * (B[..., 1] - A[..., 1])
    areas = 0.5 * cross.abs()   # shoelace formula for triangle area

    tau = 50.0 * (200.0 ** (k / steps)) # implement softmin
    scores = torch.logsumexp(-tau * areas, dim = 1) / tau   # compute scores for all batches
    loss = scores.sum() # sum scores
    
    optimizer.zero_grad()   # reset gradient
    loss.backward() # back prop
    optimizer.step()    # take a step

    with torch.no_grad():
        points.clamp_(0, 1) # keep in [0,1]

        cur = areas.min(dim=1).values   # each batches min
        r_val, r_idx = cur.max(0)        
        if r_val.item() > best_val:
            best_val = r_val.item()
            best_config = points[r_idx].detach().clone()    # save it if it's better

    if k % 200 == 0:
        print(k, "min area:", areas.min(dim = 1).values.max().item())
    
    log_file.write(f"{k},{areas.min(dim=1).values.max().item()}\n")

print("best min area found:", best_val)
print("best config:\n", best_config)

with open(os.path.join(run_dir, "best_config.txt"), "w") as f:
    f.write(f"# n={n} best_min_area={best_val}\n")
    for x, y in best_config.tolist():
        f.write(f"{x} {y}\n")

log_file.close()
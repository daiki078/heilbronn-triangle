import torch

n = 10 # number of points
steps = 50000 # iterations
lr = 0.01 # learning rate

points = torch.rand(n, 2, requires_grad = True) # generate points in [0,1)
idx = torch.combinations(torch.arange(n), r = 3) # enumerate all nCr combinations of [1,...,n]

optimizer = torch.optim.Adam([points], lr = lr)

for k in range(steps):
    A, B, C = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]] 
    cross = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1])
    areas = 0.5 * cross.abs() # shoelace formula for triangle area

    tau = 10 + k * (500 / steps) # implement softmin
    score = torch.logsumexp(-tau * areas, dim = 0) / tau # want to minimize score (same as maximizing min area)
    
    optimizer.zero_grad() # reset gradient
    score.backward() # back prop
    optimizer.step() # take a step

    with torch.no_grad():
        points.clamp_(0, 1) # keep in [0,1]

    if k % 200 == 0:
        print(k, "min area:", -score.item()) 
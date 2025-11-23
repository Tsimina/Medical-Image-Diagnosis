import torch
import torch.nn.functional as F
import numpy as np

# One Pixel Attack Implementation
def apply_one_pixel(x, sol, means, stds, device):
    img = x.clone()
    _, _, H, W = img.shape
    rx, ry, rr, rg, rb = sol

    px = int(rx * (W - 1))
    py = int(ry * (H - 1))

    rgb01 = torch.tensor([rr, rg, rb], device=device).view(1,3,1,1)
    rgb_norm = (rgb01 - means) / stds

    img[:, :, py, px] = rgb_norm.view(3)
    return img

# Fitness function for Differential Evolution
def fitness(sol, x_orig, true_label, model, means, stds, device):
    x_adv = apply_one_pixel(x_orig, sol, means, stds, device)
    logits = model(x_adv)
    probs = F.softmax(logits, dim=1)[0]
    return -probs[true_label].item()

# Differential Evolution Algorithm
def differential_evolution(
        x_orig, 
        y_true, 
        model, 
        means, 
        stds, 
        device,
        pop=200, 
        iters=120, 
        F_scale=0.7, 
        CR=1.0):

    dim = 5
    pop_vec = np.random.rand(pop, dim)
    fitness_scores = np.array([
        fitness(ind, x_orig, y_true, model, means, stds, device)
        for ind in pop_vec
    ])

    for t in range(iters):
        for i in range(pop):

            idxs = [n for n in range(pop) if n != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)

            mutant = pop_vec[a] + F_scale * (pop_vec[b] - pop_vec[c])
            mutant = np.clip(mutant, 0, 1)

            cross = np.random.rand(dim) < CR
            if not cross.any():
                cross[np.random.randint(0, dim)] = True

            trial = np.where(cross, mutant, pop_vec[i])
            f_trial = fitness(trial, x_orig, y_true, model, means, stds, device)

            if f_trial > fitness_scores[i]:
                pop_vec[i] = trial
                fitness_scores[i] = f_trial

                if model(apply_one_pixel(x_orig, trial, means, stds, device)).argmax().item() != y_true:
                    return trial

    return pop_vec[np.argmax(fitness_scores)]

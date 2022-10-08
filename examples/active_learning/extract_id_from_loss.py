import torch
import os

with open("rotation_loss.txt") as f:
    lines = f.readlines()
    assert len(lines) == 50000

losses_str, fnames = zip(*map(lambda x: x.split("_"), lines))
losses = torch.tensor(list(map(float, losses_str)))
ids = torch.tensor([int(fname.split("/")[-1][:-5]) for fname in fnames])
sorted_losses, sort_idxs = losses.sort(descending=True)
sorted_ids = ids[sort_idxs].tolist()

if not os.path.exists("./id_lists"):
    os.mkdir("./id_lists")
    
torch.save(sorted_ids, "./id_lists/sorted_rotation_proposals.pth")
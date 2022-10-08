import torch
import torchvision
import random
import os

seed = 42
torch.manual_seed(seed)
random.seed(seed)

trainset = torchvision.datasets.CIFAR10(root='/tmp/', train=True, download=True, transform=None)
class_to_id = {class_id: [] for class_id in range(10)}
for i, (x, y) in enumerate(trainset):
    class_to_id[y].append(i)

# Create oracle
torch.save(list(range(50000)), "id_lists/oracle_50k.pth")

# Create baseline 10k
# sample uniformly
balanced_class_ids = [random.choices(class_to_id[class_id], k=1000) for class_id in range(10)]
# flatten [[1,2], [9,4]] -> [1, 2, 9, 4]
balanced_class_ids = [id for ids_per_class in balanced_class_ids for id in ids_per_class]
balanced_name = "./id_lists/balanced_10k.pth"
torch.save(balanced_class_ids, balanced_name)
print(f"saved {len(balanced_class_ids)} ids to: {balanced_name}")


# Oversample the first two class
# ratio = 4600:4600:[100]x8
imbalanced_class_ids_A = [random.choices(class_to_id[class_id], k=4600) for class_id in range(2)]
imbalanced_class_ids_B = [random.choices(class_to_id[class_id], k=100) for class_id in range(2, 10)]
imbalanced_class_ids = imbalanced_class_ids_A + imbalanced_class_ids_B
imbalanced_class_ids = [id for ids_per_class in imbalanced_class_ids for id in ids_per_class]
imbalanced_name = "./id_lists/imbalanced_10k.pth"
torch.save(imbalanced_class_ids, imbalanced_name)
print(f"saved {len(imbalanced_class_ids)} ids to: {imbalanced_name}")


def select_10k(used_ids, proposed_ids):
    # Avoid using the already labeled 10k
    # we filter the used ids and select the top 10k remainder
    used_ids = set(used_ids)
    candidate_ids = [id for id in proposed_ids if not id in used_ids]
    return candidate_ids[:10000]


starting_10k = {
    "balanced": balanced_class_ids,
    "imbalanced": imbalanced_class_ids
}

proposals = {
    "random1": torch.randperm(50000).tolist(),
    "random2": torch.randperm(50000).tolist(),
    "random3": torch.randperm(50000).tolist(),
    "rotation": torch.load("./id_lists/sorted_rotation_proposals.pth"),
}

for start_k, start_v in starting_10k.items():
    for prop_k, prop_v in proposals.items():
        second_cycle_name = f"./id_lists/{start_k}_{prop_k}_20k.pth"
        candidate_ids = select_10k(start_v, prop_v)
        second_cycle_ids = start_v + candidate_ids
        torch.save(second_cycle_ids, second_cycle_name)
        print(f"saved {len(second_cycle_ids)} ids to: {second_cycle_name}")

import torch
import random

def print_random_points_in_tensor_unroll1(tensor):
    print(tensor.shape)
    for ti, unrolled_tensor in enumerate(tensor):
        print("Image {}".format(ti))

        flat_t = torch.flatten(unrolled_tensor)
        seed = len(flat_t)
        random.seed(seed)

        print(unrolled_tensor.shape, len(flat_t), seed)

        output = ""
        for i in range(10):
            index = random.randint(0, len(flat_t))
            output += "{0}:{1:.3f}, ".format(index, flat_t[index])
        print(output)

def print_random_points_in_dict(d):
    seed = len(d)
    random.seed(seed)

    output = ""
    for i in range(5):
        index = random.randint(0, len(d.keys()) - 1)
        random_key = d.keys()[index]
        output += "{}:{}, ".format(random_key, d[random_key])
    print(output)

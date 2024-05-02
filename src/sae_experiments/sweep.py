
import itertools

layers = range(12)
seeds = range(5)
sae_class_names = ("SparseAutoencoder", "GatedSparseAutoencoder")
l1_coefficients = (8e-6, 2e-5, 8e-5, 2e-4, 8e-4) 

def null_sweep():
    for _ in range(1):
        yield {}

def full_sweep():
    for seed, layer, sae_class_name, l1_coefficient in itertools.product(seeds, layers, sae_class_names, l1_coefficients):
        hook_point = f"blocks.{layer}.hook_resid_pre"
        yield {
            "seed": seed, 
            # TODO: are these redundant?
            "hook_point": hook_point,
            "hook_point_layer": layer, 
            "sae_class_name": sae_class_name,
            "l1_coefficient": l1_coefficient,
        }

def get_sweep(name: str):
    if name == "none":
        return null_sweep()
    if name == "full":
        return full_sweep()
    else:
        raise ValueError(f"Unknown sweep name: {name}") 
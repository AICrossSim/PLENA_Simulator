import torch

def print_pt_file(file_path, max_depth=3, max_items=100):
    """
    Print contents of a PyTorch .pt file
    
    Args:
        file_path: Path to the .pt file
        max_depth: Maximum depth for nested structures (default: 3)
        max_items: Maximum number of items to print for large collections (default: 100)
    """
    print(f"\n=== Contents of {file_path} ===\n")
    
    # Load the file
    data = torch.load(file_path, map_location='cpu')
    
    # Print based on data type
    print_recursive(data, depth=0, max_depth=max_depth, max_items=max_items)


def print_recursive(obj, depth=0, max_depth=3, max_items=100):
    """Helper function to recursively print nested structures"""
    indent = "  " * depth
    
    if depth > max_depth:
        print(f"{indent}... (max depth reached)")
        return
    
    if isinstance(obj, dict):
        print(f"{indent}Dictionary with {len(obj)} keys:")
        for i, (key, value) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{indent}  ... ({len(obj) - max_items} more items)")
                break
            print(f"{indent}  Key: '{key}'")
            print_recursive(value, depth + 2, max_depth, max_items)
    
    elif isinstance(obj, torch.Tensor):
        print(f"{indent}Tensor:")
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        print(f"{indent}  Device: {obj.device}")
        if obj.numel() > 0:
            print(f"{indent}  Min: {obj.min().item():.6f}")
            print(f"{indent}  Max: {obj.max().item():.6f}")
            print(f"{indent}  Mean: {obj.mean().item():.6f}")
        if obj.numel() <= 20:  # Print small tensors completely
            print(f"{indent}  Values: {obj.tolist()}")
        else:
            print(f"{indent}  First few values: {obj.flatten()[:10].tolist()}")
    
    elif isinstance(obj, (list, tuple)):
        type_name = "List" if isinstance(obj, list) else "Tuple"
        print(f"{indent}{type_name} with {len(obj)} items:")
        for i, item in enumerate(obj):
            if i >= max_items:
                print(f"{indent}  ... ({len(obj) - max_items} more items)")
                break
            print(f"{indent}  [{i}]:")
            print_recursive(item, depth + 2, max_depth, max_items)
    
    elif isinstance(obj, torch.nn.Module):
        print(f"{indent}PyTorch Model: {obj.__class__.__name__}")
        print(f"{indent}Model structure:")
        print(f"{indent}{obj}")
    
    elif isinstance(obj, (int, float, str, bool)):
        print(f"{indent}Value ({type(obj).__name__}): {obj}")
    
    else:
        print(f"{indent}Type: {type(obj)}")
        print(f"{indent}Value: {obj}")

if __name__ == "__main__":
    print_pt_file("../../test/Instr_Level_Benchmark/build/vector_fp_add/fake_test_raw_data.pt")
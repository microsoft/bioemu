from bioemu.steering import stratified_resample

import torch

weights = torch.rand(10, 100)
weights = weights / weights.sum(dim=-1, keepdim=True)

indexes = stratified_resample(weights)

print(indexes)

# Test 1: Basic functionality
def test_basic_functionality():
    """Test that stratified_resample returns correct shape and valid indices"""
    B, N = 5, 20
    weights = torch.rand(B, N)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    indices = stratified_resample(weights)
    
    assert indices.shape == (B, N), f"Expected shape {(B, N)}, got {indices.shape}"
    assert torch.all(indices >= 0) and torch.all(indices < N), "Indices out of bounds"
    print("✓ Basic functionality test passed")

# Test 2: Uniform weights should give approximately uniform sampling
def test_uniform_weights():
    """Test that uniform weights produce approximately uniform sampling"""
    B, N = 1, 1000
    weights = torch.ones(B, N) / N  # uniform weights
    
    indices = stratified_resample(weights)
    
    # Count frequency of each index
    counts = torch.bincount(indices[0], minlength=N)
    expected_count = N / N  # should be 1 for each
    
    # Check that counts are reasonably close to expected (within 20% tolerance)
    max_deviation = torch.abs(counts - expected_count).max()
    assert max_deviation <= 2, f"Max deviation {max_deviation} too large for uniform sampling"
    print("✓ Uniform weights test passed")

# Test 3: Extreme weights (one weight = 1, others = 0)
def test_extreme_weights():
    """Test that extreme weights concentrate sampling on high-weight indices"""
    B, N = 3, 10
    weights = torch.zeros(B, N)
    weights[0, 5] = 1.0  # All weight on index 5 for batch 0
    weights[1, 2] = 1.0  # All weight on index 2 for batch 1
    weights[2, 8] = 1.0  # All weight on index 8 for batch 2
    
    indices = stratified_resample(weights)
    
    # All indices should be the concentrated index for each batch
    assert torch.all(indices[0] == 5), "Batch 0 should sample only index 5"
    assert torch.all(indices[1] == 2), "Batch 1 should sample only index 2"
    assert torch.all(indices[2] == 8), "Batch 2 should sample only index 8"
    print("✓ Extreme weights test passed")

# Test 4: Reproducibility with fixed seed
def test_reproducibility():
    """Test that results are reproducible with fixed seed"""
    torch.manual_seed(42)
    weights = torch.rand(2, 15)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    indices1 = stratified_resample(weights)
    
    torch.manual_seed(42)
    indices2 = stratified_resample(weights)
    
    assert torch.equal(indices1, indices2), "Results should be reproducible with same seed"
    print("✓ Reproducibility test passed")

# Test 5: Edge case - single element
def test_single_element():
    """Test edge case with single element"""
    weights = torch.ones(3, 1)  # Only one element per batch
    indices = stratified_resample(weights)
    
    assert indices.shape == (3, 1)
    assert torch.all(indices == 0), "Single element should always return index 0"
    print("✓ Single element test passed")

# Test 6: Compare with multinomial sampling - fewer dropped samples
def test_fewer_dropped_samples():
    """Test that stratified sampling drops fewer samples than multinomial sampling"""
    # torch.manual_seed(123)
    B, N = 2, 300
    
    # Create skewed weights where some particles have much higher weight
    weights = torch.ones(B, N)
    weights[0, :5] = 0.8  # High weight particles
    weights[0, 5:] = 0.2  # Low weight particles
    weights[1, :3] = 0.9  # High weight particles  
    weights[1, 3:] = 0.1  # Low weight particles
    weights = weights / weights.sum(dim=-1, keepdim=True)
    
    # Stratified sampling
    indices_stratified = stratified_resample(weights)
    
    # Multinomial sampling
    indices_multinomial = torch.multinomial(weights, N, replacement=True)
    
    # Count unique indices (samples that weren't dropped)
    unique_stratified = [len(torch.unique(indices_stratified[b])) for b in range(B)]
    unique_multinomial = [len(torch.unique(indices_multinomial[b])) for b in range(B)]
    
    # print(f"Stratified sampling unique indices: {unique_stratified}")
    # print(f"Multinomial sampling unique indices: {unique_multinomial}")
    
    # Stratified should generally have more unique indices (fewer dropped samples)
    avg_unique_stratified = sum(unique_stratified) / B
    avg_unique_multinomial = sum(unique_multinomial) / B
    
    assert avg_unique_stratified >= avg_unique_multinomial, \
        f"Stratified sampling should drop fewer samples: {avg_unique_stratified} vs {avg_unique_multinomial}"
	
    print(f"✓ Fewer dropped samples test passed: {avg_unique_stratified} vs {avg_unique_multinomial}")



# Run all tests
test_basic_functionality()
test_uniform_weights()
test_extreme_weights()
# test_reproducibility()
test_single_element()
test_fewer_dropped_samples()
# print("\n✅ All stratified sampling tests passed!")


import torch
import torch.nn.functional as F
import time
import math

def dense_attention_pytorch(Q, K, V):

    return F.scaled_dot_product_attention(Q, K, V)

def sparse_attention_pytorch(Q, K, V, mask):
    """
    mask: True for attend, False for mask out
    """

    atten_mask = mask
    return F.scaled_dot_product_attention(Q, K, V, attn_mask = atten_mask)

def naive_attention_pytorch(Q, K, V, mask = None):
    """
    Args:
        Q: [batch_size, seq_len, dim]
        K: [batch_size, seq_len, dim]
        V: [batch_size, seq_len, dim]
        mask: [seq_len, seq_len] boolean mask, optional
    """

    dim = Q.shape[-1]
    d_k = dim

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5) # [batch_size, seq_len, seq_len]

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    atten_weights = F.softmax(scores, dim = -1)

    return torch.matmul(atten_weights, V) # [batch_size, seq_len, dim]

def flash_attention_pytorch(Q, K, V):
    """ 
    Flash atten without batch and mask

    Args:
        Q: [seq_len, dim]
        K: [seq_len, dim]
        V: [seq_len, dim]
    """
    
    # assume on-chip sram has size 128KB
    M = 128 * 1024
    seq_len, dim = Q.shape[0], Q.shape[1]
    
    # Calculate block sizes properly
    B_c = min(math.ceil(M / (4 * dim)), seq_len)
    B_r = min(math.ceil(M / (4 * dim)), seq_len)

    O = torch.zeros(seq_len, dim, device=Q.device, dtype=Q.dtype)
    l = torch.zeros(seq_len, 1, device=Q.device, dtype=Q.dtype)
    m = torch.full((seq_len, 1), -math.inf, device=Q.device, dtype=Q.dtype)

    # divide into blocks
    T_r = math.ceil(seq_len / B_r)
    T_c = math.ceil(seq_len / B_c)

    for j in range(T_c):
        # load K_j, V_j
        start_j = j * B_c
        end_j = min(seq_len, (j + 1) * B_c)
        K_j = K[start_j:end_j, :]
        V_j = V[start_j:end_j, :]

        for i in range(T_r):
            # load Q_i, l_i, m_i, O_i
            start_i = i * B_r
            end_i = min(seq_len, (i + 1) * B_r)
            Q_i = Q[start_i:end_i, :]  # [B_r, dim]
            O_i = O[start_i:end_i, :].clone()  # [B_r, dim]
            l_i = l[start_i:end_i, :].clone()  # [B_r, 1]
            m_i = m[start_i:end_i, :].clone()  # [B_r, 1]

            # compute score
            S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) / math.sqrt(dim)  # [B_r, B_c]
            
            # row-wise max score
            m_ij = torch.max(S_ij, dim=1, keepdim=True)[0]  # [B_r, 1]
            
            # compute softmax probabilities
            P_ij = torch.exp(S_ij - m_ij)  # [B_r, B_c]
            
            # row_wise sum
            l_ij = torch.sum(P_ij, dim=1, keepdim=True)  # [B_r, 1]

            # update statistics
            m_i_new = torch.maximum(m_i, m_ij)  # [B_r, 1]
            l_i_new = torch.exp(m_i - m_i_new) * l_i + torch.exp(m_ij - m_i_new) * l_ij  # [B_r, 1]

            # update output
            O_i_new = (torch.exp(m_i - m_i_new) * l_i * O_i + 
                       torch.exp(m_ij - m_i_new) * torch.matmul(P_ij, V_j)) / l_i_new

            # write back to global tensors
            O[start_i:end_i, :] = O_i_new
            l[start_i:end_i, :] = l_i_new
            m[start_i:end_i, :] = m_i_new
            
    return O

def benchmark_pytorch(Q, K, V, mask = None, num_warmup = 10, num_iterations = 100):

    device = Q.device if torch.cuda.is_available() else 'cpu'

    for _ in range(num_warmup):
        _ = sparse_attention_pytorch(Q, K, V, mask) if mask is not None else dense_attention_pytorch(Q, K, V)
    
    if device == 'cuda':
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing = True)
        end_event = torch.cuda.Event(enable_timing = True)

        start_event.record()

        for _ in range(num_iterations):
            _ = sparse_attention_pytorch(Q, K, V, mask) if mask is not None else dense_attention_pytorch(Q, K, V)

        end_event.record()
        torch.cuda.synchronize()

        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / num_iterations

    else:
        start_time = time.time()

        for _ in range(num_iterations):
            _ = sparse_attention_pytorch(Q, K, V, mask) if mask is not None else dense_attention_pytorch(Q, K, V)
        
        end_time = time.time()

        avg_time_ms = (end_time - start_time) / num_iterations * 1000

    return avg_time_ms

def test_correctness():

    print("=" * 60)
    print("Testing PyTorch Reference Implementations")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    batch_size = 2
    seq_len = 128
    dim = 64
    
    print(f"\nGenerating test data...")
    print(f"  batch_size={batch_size}, seq_len={seq_len}, dim={dim}")
    
    Q = torch.randn(batch_size, seq_len, dim, device=device)
    K = torch.randn(batch_size, seq_len, dim, device=device)
    V = torch.randn(batch_size, seq_len, dim, device=device)
    
    # Test 1: Dense attention
    print("\n" + "-" * 60)
    print("Test 1: Dense Attention")
    print("-" * 60)
    
    output_dense = dense_attention_pytorch(Q, K, V)
    output_naive = naive_attention_pytorch(Q, K, V)
    
    error = torch.abs(output_dense - output_naive).max().item()
    print(f"  Output shape: {output_dense.shape}")
    print(f"  Max difference (optimized vs naive): {error:.2e}")
    
    if error < 1e-4:
        print("Dense attention works correctly!")
    else:
        print("Error too large!")
    
    # Test 2: Sparse attention (sliding window)
    print("\n" + "-" * 60)
    print("Test 2: Sparse Attention (Sliding Window)")
    print("-" * 60)
    
    window_size = 16
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = True
    
    sparsity = 1.0 - (mask.sum() / mask.numel())
    print(f"  Window size: {window_size}")
    print(f"  Mask sparsity: {sparsity:.1%}")
    
    output_sparse = sparse_attention_pytorch(Q, K, V, mask)
    output_naive_sparse = naive_attention_pytorch(Q, K, V, mask)
    
    error = torch.abs(output_sparse - output_naive_sparse).max().item()
    print(f"  Output shape: {output_sparse.shape}")
    print(f"  Max difference (optimized vs naive): {error:.2e}")
    
    if error < 1e-4:
        print("Sparse attention works correctly!")
    else:
        print("Error too large!")
    
    # Test FlashAttention
    print("\n" + "-" * 60)
    print("Test 3: FlashAttention")
    print("-" * 60)

    Q = torch.randn(seq_len, dim, device=device)
    K = torch.randn(seq_len, dim, device=device)
    V = torch.randn(seq_len, dim, device=device)
    
    output_flash = flash_attention_pytorch(Q, K, V)
    output_naive_flash = naive_attention_pytorch(Q, K, V)
    
    error = torch.abs(output_flash - output_naive_flash).max().item()
    print(f"  Output shape: {output_flash.shape}")
    print(f"  Max difference (optimized vs naive): {error:.2e}")
    
    if error < 1e-4:
        print("FlashAttention works correctly!")
    else:
        print("Error too large!")

if __name__ == '__main__':
    test_correctness()
    

    


import torch

class AttentionTestData:

    """
    Generates test data for the attention module
    """

    def __init__(self, device = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"AttentionTestData initialized with device: {self.device}")
    
    def generate_qkv(self, batch_size, seq_len, dim, dtype = torch.float32):

        """
        Generates random Q, K, V tensors

        Args:
            batch_size (int): The batch size of the tensors
            seq_len (int): The sequence length of the tensors
            dim (int): The dimension of the tensors
            dtype (torch.dtype, optional): The data type of the tensors. Defaults to torch.float32.

        Returns:
            Q, K, V: [batch_size, seq_len, dim] tensors
        """

        Q = torch.randn(batch_size, seq_len, dim, dtype = dtype, device = self.device)
        K = torch.randn(batch_size, seq_len, dim, dtype = dtype, device = self.device)
        V = torch.randn(batch_size, seq_len, dim, dtype = dtype, device = self.device)

        return Q, K, V
    
    def generate_dense_mask(self, seq_len):

        """
        Dense attention attends to all positions
        """

        return torch.ones(seq_len, seq_len, dtype = torch.bool, device = self.device)
    
    def generate_sliding_window_mask(self, seq_len, window_size):

        """
        Generates a sliding window mask like Mistral format
        """

        mask = torch.zeros(seq_len, seq_len, dtype = torch.bool, device = self.device)

        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = 1 # start till end-1
        
        return mask

    def generate_block_sparse_mask(self, seq_len, block_size, sparse_ratio = 0.5):

        """
        Generates a block sparse mask
        """

        num_blocks = (seq_len + block_size - 1) // block_size

        # block-wise mask
        block_mask = torch.rand(num_blocks, num_blocks, device = self.device) > sparse_ratio

        # expand to full resolution
        mask = block_mask.repeat_interleave(block_size, dim = 0).repeat_interleave(block_size, dim = 1)

        # clip
        mask = mask[:seq_len, :seq_len]

        return mask

    def calculate_mask_sparsity(self, mask: torch.Tensor):
        total_elements = mask.numel()
        true_elements = mask.sum().item()

        sparsity = 1 - true_elements / total_elements
        return sparsity
    
    def get_standard_configs(self):

        return {
            'tiny': {
                'batch_size': 1,
                'seq_len': 128,
                'dim': 64,
                'description': 'Tiny config for quick testing'
            },
            'small': {
                'batch_size': 1,
                'seq_len': 512,
                'dim': 64,
                'description': 'Small config for development'
            },
            'medium': {
                'batch_size': 1,
                'seq_len': 1024,
                'dim': 128,
                'description': 'Medium config for benchmarking'
            },
            'large': {
                'batch_size': 1,
                'seq_len': 2048,
                'dim': 128,
                'description': 'Large config for stress testing'
            },
            'mistral_style': {
                'batch_size': 1,
                'seq_len': 1024,
                'dim': 128,
                'window_size': 256,
                'description': 'Mistral-style sliding window'
            }
        }

def demo():

    print("=" * 60)
    print("AttentionTestData Demo")
    print("=" * 60)
    
    gen = AttentionTestData()
    
    print("\n1. Generating Q, K, V...")
    batch, seq_len, dim = 2, 128, 64
    Q, K, V = gen.generate_qkv(batch, seq_len, dim)
    print(f"   Q shape: {Q.shape}")
    print(f"   K shape: {K.shape}")
    print(f"   V shape: {V.shape}")
    print(f"   Device: {Q.device}")
    
    print("\n2. Dense mask (all positions attend)...")
    mask_dense = gen.generate_dense_mask(seq_len)
    sparsity_dense = gen.calculate_mask_sparsity(mask_dense)
    print(f"   Mask shape: {mask_dense.shape}")
    print(f"   Sparsity: {sparsity_dense:.1%}")
    
    print("\n3. Sliding window mask...")
    window_sizes = [16, 32, 64]
    for ws in window_sizes:
        mask_sw = gen.generate_sliding_window_mask(seq_len, ws)
        sparsity_sw = gen.calculate_mask_sparsity(mask_sw)
        print(f"   Window size {ws}: sparsity = {sparsity_sw:.1%}")
    
    print("\n4. Block-sparse mask...")
    sparsity_ratios = [0.25, 0.5, 0.75]
    for sr in sparsity_ratios:
        mask_bs = gen.generate_block_sparse_mask(seq_len, block_size=16, sparse_ratio=sr)
        actual_sparsity = gen.calculate_mask_sparsity(mask_bs)
        print(f"   Target {sr:.0%}: actual = {actual_sparsity:.1%}")
    
    print("\n5. Standard configs...")
    configs = gen.get_standard_configs()
    for name, config in configs.items():
        print(f"   - {name}: {config['description']}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    demo()


        
"""Part 5 (stretch): Paged KV Cache

Fixed-size blocks instead of contiguous allocation.
Same idea as OS virtual memory pages.
"""

import torch


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int, num_layers: int,
                 num_heads: int, head_dim: int, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers

        self.k_pool = [
            torch.zeros(num_blocks, num_heads, block_size, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.v_pool = [
            torch.zeros(num_blocks, num_heads, block_size, head_dim,
                        device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        self.free_blocks: list[int] = list(range(num_blocks))
        self.seq_to_blocks: dict[int, list[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate blocks for a sequence. Returns list of block IDs."""
        num_required = (num_tokens + self.block_size - 1) // self.block_size
        current_blocks = self.seq_to_blocks.get(seq_id, [])

        if len(current_blocks) >= num_required:
            return current_blocks

        num_new_blocks = num_required - len(current_blocks)
        if num_new_blocks > len(self.free_blocks):
            raise RuntimeError("Not enough free blocks available.")

        # Extend the existing mapping so a sequence can grow over time.
        new_blocks = [self.free_blocks.pop(0) for _ in range(num_new_blocks)]
        updated_blocks = current_blocks + new_blocks
        self.seq_to_blocks[seq_id] = updated_blocks
        return updated_blocks

    def free(self, seq_id: int):
        """Free all blocks for a finished sequence."""
        block_ids = self.seq_to_blocks.pop(seq_id, [])
        if not block_ids:
            return

        for layer_idx in range(self.num_layers):
            # Clear released blocks so the next sequence starts from clean memory.
            self.k_pool[layer_idx][block_ids].zero_()
            self.v_pool[layer_idx][block_ids].zero_()

        self.free_blocks.extend(block_ids)
        self.free_blocks.sort()

    def get_block_ids(self, seq_id: int) -> list[int]:
        return self.seq_to_blocks.get(seq_id, [])

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

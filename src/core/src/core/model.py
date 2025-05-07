import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFrameAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiFrameAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        return self.linear(attn_output)


class ConformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.mfa = MultiFrameAttention(embed_dim, num_heads)
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv1d(
            embed_dim, embed_dim, kernel_size=31, padding=15, groups=embed_dim
        )  # Depthwise conv

    def forward(self, x):
        # Multi-Frame Attention
        residual = x
        x = self.mfa(x)
        x = self.norm1(residual + self.dropout(x))

        # Convolutional Module
        residual = x
        x = x.transpose(1, 2)  # (batch_size, embed_dim, sequence_length)
        x = F.gelu(self.conv(x))
        x = x.transpose(1, 2)  # Back to (batch_size, sequence_length, embed_dim)
        x = self.norm2(residual + self.dropout(x))

        # Feedforward Network
        residual = x
        x = F.gelu(self.ff1(x))
        x = self.ff2(x)
        return residual + self.dropout(x)


class ConformerMFA(nn.Module):
    def __init__(
        self, num_blocks, embed_dim, num_heads, ff_dim, num_classes, dropout=0.1
    ):
        super(ConformerMFA, self).__init__()
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(embed_dim, num_heads, ff_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, embed_dim)
        for block in self.blocks:
            x = block(x)
        # Average pooling for sequence aggregation
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        return self.fc_out(x)


# Example Usage
if __name__ == "__main__":
    # Define model
    model = ConformerMFA(
        num_blocks=4,
        embed_dim=256,
        num_heads=8,
        ff_dim=1024,
        num_classes=10,
        dropout=0.1,
    )

    # Dummy input (batch_size=16, sequence_length=100, embed_dim=256)
    dummy_input = torch.randn(16, 100, 256)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Should be (16, num_classes)

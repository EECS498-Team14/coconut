"""
Simple MLP predictor for forward distance prediction.

Predicts how many more latent reasoning steps are needed to reach a correct answer.
"""

import torch
from torch import nn


class ForwardDistancePredictor(nn.Module):
    """
    Simple MLP that predicts forward distance from hidden states.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list = [512, 512],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final regression layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, hidden_dim)

        Returns:
            predictions: (batch_size,) - predicted forward distance
        """
        output = self.network(hidden_states)
        return output.squeeze(-1)

    def predict(self, hidden_states):
        """
        Predict and round to nearest integer.

        Args:
            hidden_states: (batch_size, hidden_dim)

        Returns:
            predictions: (batch_size,) - integer predictions
        """
        with torch.no_grad():
            predictions = self.forward(hidden_states)
            return predictions.round().long()


def create_predictor(
    input_dim: int,
    hidden_layers: list = [512, 512],
    dropout: float = 0.1,
):
    """Factory function to create a predictor."""
    return ForwardDistancePredictor(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )


class MultiTargetPredictor(nn.Module):
    """
    MLP that predicts probability distribution over all possible token counts [0-6].

    Instead of predicting a single "optimal" token count, this outputs a distribution
    over all token counts, trained to match the set of ALL correct choices.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_layers: list = [1024, 512],
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final classification layer - outputs logits for each class
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, hidden_dim)

        Returns:
            logits: (batch_size, num_classes) - logits for each token count
        """
        return self.network(hidden_states)

    def predict_distribution(self, hidden_states):
        """
        Get probability distribution over token counts.

        Args:
            hidden_states: (batch_size, hidden_dim)

        Returns:
            probabilities: (batch_size, num_classes) - probability distribution
        """
        logits = self.forward(hidden_states)
        return torch.softmax(logits, dim=-1)

    def predict_argmax(self, hidden_states):
        """
        Predict most likely token count.

        Args:
            hidden_states: (batch_size, hidden_dim)

        Returns:
            predictions: (batch_size,) - predicted token count
        """
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)
            return dist.argmax(dim=-1)

    def predict_conservative(self, hidden_states, threshold=0.3):
        """
        Conservative prediction: pick highest token count with prob > threshold.

        Args:
            hidden_states: (batch_size, hidden_dim)
            threshold: minimum probability to accept

        Returns:
            predictions: (batch_size,) - predicted token count
        """
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)
            batch_size = dist.size(0)
            predictions = torch.zeros(batch_size, dtype=torch.long)

            for i in range(batch_size):
                # Search from highest to lowest token count
                for token_count in range(self.num_classes - 1, -1, -1):
                    if dist[i, token_count] > threshold:
                        predictions[i] = token_count
                        break
                else:
                    # If no token count meets threshold, default to max
                    predictions[i] = self.num_classes - 1

            return predictions

    def predict_sample(self, hidden_states, temperature=1.0):
        """
        Sample token count from distribution.

        Args:
            hidden_states: (batch_size, hidden_dim)
            temperature: temperature for sampling (higher = more random)

        Returns:
            predictions: (batch_size,) - sampled token count
        """
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)

            # Apply temperature scaling
            if temperature != 1.0:
                dist = dist ** (1.0 / temperature)
                dist = dist / dist.sum(dim=-1, keepdim=True)

            return torch.multinomial(dist, 1).squeeze(-1)

    def predict_ensemble(self, hidden_states, confidence_threshold=0.7, baseline_token=6):
        """
        Ensemble prediction: use predictor if confident, otherwise fallback to baseline.

        Strategy: Combine predictor with "always 6" baseline based on confidence.
        When predictor is confident (max prob > threshold), trust it.
        When uncertain (max prob <= threshold), fallback to safer baseline (token 6).

        Args:
            hidden_states: (batch_size, hidden_dim)
            confidence_threshold: minimum confidence to trust predictor
            baseline_token: token to use when confidence is low (default: 6)

        Returns:
            predictions: (batch_size,) - predicted token count
        """
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)
            batch_size = dist.size(0)
            predictions = torch.zeros(batch_size, dtype=torch.long)

            for i in range(batch_size):
                # Get most likely token and its probability
                max_prob, argmax_token = dist[i].max(dim=0)

                if max_prob.item() > confidence_threshold:
                    # High confidence: trust predictor
                    predictions[i] = argmax_token
                else:
                    # Low confidence: use baseline
                    predictions[i] = baseline_token

            return predictions


class ResidualBlock(nn.Module):
    """
    Residual block with LayerNorm and GELU activation.

    Enables training deeper networks by adding skip connections.
    """

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return x + self.net(x)  # Residual connection


class ResNetPredictor(nn.Module):
    """
    ResNet-style predictor with residual blocks.

    Uses residual connections to enable much deeper networks (6+ layers)
    compared to vanilla MLP (2-3 layers). LayerNorm stabilizes training.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Project input to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # Output layer
        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        x = self.input_proj(hidden_states)
        for block in self.blocks:
            x = block(x)
        return self.output(x)

    def predict_distribution(self, hidden_states):
        """Get probability distribution over token counts."""
        logits = self.forward(hidden_states)
        return torch.softmax(logits, dim=-1)

    def predict_argmax(self, hidden_states):
        """Predict most likely token count."""
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)
            return dist.argmax(dim=-1)

    def predict_conservative(self, hidden_states, threshold=0.3):
        """Conservative prediction: pick highest token count with prob > threshold."""
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)
            batch_size = dist.size(0)
            predictions = torch.zeros(batch_size, dtype=torch.long)

            for i in range(batch_size):
                # Search from highest to lowest token count
                for token_count in range(self.num_classes - 1, -1, -1):
                    if dist[i, token_count] > threshold:
                        predictions[i] = token_count
                        break
                else:
                    # If no token count meets threshold, default to max
                    predictions[i] = self.num_classes - 1

            return predictions

    def predict_sample(self, hidden_states, temperature=1.0):
        """Sample token count from distribution."""
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)

            # Apply temperature scaling
            if temperature != 1.0:
                dist = dist ** (1.0 / temperature)
                dist = dist / dist.sum(dim=-1, keepdim=True)

            return torch.multinomial(dist, 1).squeeze(-1)

    def predict_ensemble(self, hidden_states, confidence_threshold=0.7, baseline_token=6):
        """Ensemble prediction: use predictor if confident, otherwise fallback to baseline."""
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)
            batch_size = dist.size(0)
            predictions = torch.zeros(batch_size, dtype=torch.long)

            for i in range(batch_size):
                # Get most likely token and its probability
                max_prob, argmax_token = dist[i].max(dim=0)

                if max_prob.item() > confidence_threshold:
                    # High confidence: trust predictor
                    predictions[i] = argmax_token
                else:
                    # Low confidence: use baseline
                    predictions[i] = baseline_token

            return predictions


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor using self-attention to model feature interactions.

    Key innovation: Unlike MLP/ResNet which process features independently,
    Transformer uses self-attention to learn which combinations of h0 features
    are important for predicting optimal token count.

    Architecture:
    1. Reshape h0 (768-dim) into sequence of tokens (e.g., 12 tokens × 64 dim)
    2. Add learned positional embeddings
    3. Apply Transformer encoder layers (multi-head self-attention + FFN)
    4. Pool sequence into single vector (mean pooling)
    5. Classify into 7 classes
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 7,
        num_tokens: int = 12,
        d_model: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        assert input_dim % num_tokens == 0, f"input_dim ({input_dim}) must be divisible by num_tokens ({num_tokens})"
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.num_classes = num_classes
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.token_dim = input_dim // num_tokens

        # If token_dim != d_model, we need a projection
        if self.token_dim != d_model:
            self.input_proj = nn.Linear(self.token_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        # Learned positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # Use (batch, seq, feature) format
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, input_dim)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = hidden_states.size(0)

        # Reshape: (batch, input_dim) -> (batch, num_tokens, token_dim)
        x = hidden_states.view(batch_size, self.num_tokens, self.token_dim)

        # Project to d_model if needed
        x = self.input_proj(x)  # (batch, num_tokens, d_model)

        # Add positional embeddings
        x = x + self.pos_embedding  # (batch, num_tokens, d_model)

        # Apply transformer encoder
        x = self.transformer(x)  # (batch, num_tokens, d_model)

        # Mean pooling across sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Classify
        logits = self.classifier(x)  # (batch, num_classes)

        return logits

    def predict_distribution(self, hidden_states):
        """Get probability distribution over token counts."""
        logits = self.forward(hidden_states)
        return torch.softmax(logits, dim=-1)

    def predict_argmax(self, hidden_states):
        """Predict most likely token count."""
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)
            return dist.argmax(dim=-1)

    def predict_conservative(self, hidden_states, threshold=0.3):
        """Conservative prediction: pick highest token count with prob > threshold."""
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)
            batch_size = dist.size(0)
            predictions = torch.zeros(batch_size, dtype=torch.long)

            for i in range(batch_size):
                # Search from highest to lowest token count
                for token_count in range(self.num_classes - 1, -1, -1):
                    if dist[i, token_count] > threshold:
                        predictions[i] = token_count
                        break
                else:
                    # If no token count meets threshold, default to max
                    predictions[i] = self.num_classes - 1

            return predictions

    def predict_sample(self, hidden_states, temperature=1.0):
        """Sample token count from distribution."""
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)

            # Apply temperature scaling
            if temperature != 1.0:
                dist = dist ** (1.0 / temperature)
                dist = dist / dist.sum(dim=-1, keepdim=True)

            return torch.multinomial(dist, 1).squeeze(-1)

    def predict_ensemble(self, hidden_states, confidence_threshold=0.7, baseline_token=6):
        """Ensemble prediction: use predictor if confident, otherwise fallback to baseline."""
        with torch.no_grad():
            dist = self.predict_distribution(hidden_states)
            batch_size = dist.size(0)
            predictions = torch.zeros(batch_size, dtype=torch.long)

            for i in range(batch_size):
                # Get most likely token and its probability
                max_prob, argmax_token = dist[i].max(dim=0)

                if max_prob.item() > confidence_threshold:
                    # High confidence: trust predictor
                    predictions[i] = argmax_token
                else:
                    # Low confidence: use baseline
                    predictions[i] = baseline_token

            return predictions


def create_multi_target_predictor(
    input_dim: int,
    num_classes: int = 7,
    hidden_layers: list = [1024, 512],
    dropout: float = 0.3,
    architecture: str = "mlp",
    **kwargs,
):
    """
    Factory function to create a multi-target predictor.

    Args:
        input_dim: Dimension of input hidden states
        num_classes: Number of classes to predict (default: 7 for token counts 0-6)
        hidden_layers: List of hidden layer dimensions (for MLP architecture)
        dropout: Dropout probability
        architecture: Architecture type ('mlp', 'resnet', 'transformer', 'attention', 'moe')
        **kwargs: Additional architecture-specific arguments

    Returns:
        Predictor model
    """
    if architecture == "mlp":
        return MultiTargetPredictor(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
    elif architecture == "resnet":
        return ResNetPredictor(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=kwargs.get("hidden_dim", 512),
            num_blocks=kwargs.get("num_blocks", 6),
            dropout=dropout,
        )
    elif architecture == "transformer":
        return TransformerPredictor(
            input_dim=input_dim,
            num_classes=num_classes,
            num_tokens=kwargs.get("num_tokens", 12),
            d_model=kwargs.get("d_model", 64),
            num_heads=kwargs.get("num_heads", 8),
            num_layers=kwargs.get("num_layers", 2),
            dim_feedforward=kwargs.get("dim_feedforward", 256),
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


class MidpointRegressionPredictor(nn.Module):
    """
    Simple MLP for midpoint regression on consecutive samples.

    Predicts the midpoint of the consecutive interval of correct token counts.
    For example:
    - [2,3,4] → midpoint = 3.0
    - [3,4,5,6] → midpoint = 4.5
    - [5] → midpoint = 5.0
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_layers: list = [1024, 512],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Final regression layer (no activation - raw output)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, input_dim)

        Returns:
            predictions: (batch_size,) - predicted midpoint (continuous value)
        """
        output = self.network(hidden_states)
        return output.squeeze(-1)

    def predict(self, hidden_states):
        """
        Predict midpoint and round to nearest integer token count.

        Args:
            hidden_states: (batch_size, input_dim)

        Returns:
            token_counts: (batch_size,) - rounded and clamped to [0, 6]
        """
        with torch.no_grad():
            midpoints = self.forward(hidden_states)
            # Round to nearest integer and clamp to valid range
            token_counts = torch.clamp(torch.round(midpoints), min=0, max=6).long()
            return token_counts

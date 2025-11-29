"""
Train multi-target predictor: h0 → distribution over all correct token counts.
"""

import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from forward_predictor import create_multi_target_predictor


class MultiTargetKLLoss(nn.Module):
    """
    KL divergence loss for multi-target prediction.

    Trains model to match uniform distribution over all correct token counts.
    """

    def forward(self, logits, multi_hot_labels):
        """
        Args:
            logits: (batch_size, num_classes) - model outputs
            multi_hot_labels: (batch_size, num_classes) - binary labels for each class

        Returns:
            loss: scalar - KL divergence between predicted and target distributions
        """
        # Normalize multi-hot labels to get target distribution
        # Example: [0,0,1,1,1,0,0] → [0, 0, 0.33, 0.33, 0.33, 0, 0]
        target_dist = multi_hot_labels / multi_hot_labels.sum(dim=1, keepdim=True)

        # Log softmax of predictions
        log_pred = F.log_softmax(logits, dim=1)

        # KL divergence
        return F.kl_div(log_pred, target_dist, reduction='batchmean')


class MultiTargetBCELoss(nn.Module):
    """
    Binary cross-entropy loss for multi-target prediction.

    Treats each token count as independent binary classification.
    """

    def forward(self, logits, multi_hot_labels):
        """
        Args:
            logits: (batch_size, num_classes) - model outputs
            multi_hot_labels: (batch_size, num_classes) - binary labels

        Returns:
            loss: scalar - BCE loss
        """
        return F.binary_cross_entropy_with_logits(logits, multi_hot_labels)


class WeightedMultiTargetLoss(nn.Module):
    """
    Class-balanced weighted loss for multi-target prediction.

    Weights each token count inversely by frequency to address data imbalance.
    Token 6 appears in 88.5% of samples → gets low weight
    Token 0 appears in 11.5% of samples → gets high weight
    """

    def __init__(self, class_weights):
        """
        Args:
            class_weights: (num_classes,) tensor with weight for each token count
        """
        super().__init__()
        self.register_buffer('class_weights', class_weights)

    def forward(self, logits, multi_hot_labels):
        """
        Args:
            logits: (batch_size, num_classes) - model outputs
            multi_hot_labels: (batch_size, num_classes) - binary labels

        Returns:
            loss: per-sample loss - weighted BCE loss
        """
        # Use pos_weight parameter for class balancing
        # pos_weight[i] = weight for class i when label=1
        pos_weight = self.class_weights.unsqueeze(0).expand_as(multi_hot_labels)

        # Weighted binary cross-entropy with pos_weight
        # Use reduction='none' to get per-sample, per-class losses
        bce_per_class = F.binary_cross_entropy_with_logits(
            logits, multi_hot_labels, pos_weight=pos_weight, reduction='none'
        )

        # Sum across classes, average across batch
        # Return per-sample loss for compatibility with sample weighting
        return bce_per_class.sum(dim=1)


class WeightedFocalLoss(nn.Module):
    """
    Combined weighted + focal loss for multi-target prediction.

    Combines two strategies:
    1. Class-balanced weighting: Penalizes rare classes more
    2. Focal loss: Down-weights confident predictions to focus on hard examples

    This addresses both:
    - Data imbalance (88.5% samples have token 6 correct)
    - Easy vs hard examples (model confident on some, uncertain on others)
    """

    def __init__(self, class_weights, gamma=2.0):
        """
        Args:
            class_weights: (num_classes,) tensor with weight for each token count
            gamma: Focal loss parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.register_buffer('class_weights', class_weights)
        self.gamma = gamma

    def forward(self, logits, multi_hot_labels):
        """
        Args:
            logits: (batch_size, num_classes) - model outputs
            multi_hot_labels: (batch_size, num_classes) - binary labels

        Returns:
            loss: scalar - weighted focal loss
        """
        # Get probabilities
        probs = torch.sigmoid(logits)

        # Compute pt: probability of the true class
        # If label=1, pt = p; if label=0, pt = 1-p
        pt = probs * multi_hot_labels + (1 - probs) * (1 - multi_hot_labels)

        # Focal weight: (1-pt)^gamma
        # High when model is uncertain (pt close to 0.5)
        # Low when model is confident (pt close to 0 or 1)
        focal_weight = (1 - pt) ** self.gamma

        # Class-balanced BCE (using pos_weight)
        pos_weight = self.class_weights.unsqueeze(0).expand_as(multi_hot_labels)
        bce = F.binary_cross_entropy_with_logits(
            logits, multi_hot_labels, pos_weight=pos_weight, reduction='none'
        )

        # Combine focal weight with class-balanced BCE
        focal_loss = focal_weight * bce

        return focal_loss.mean()


def compute_class_weights(multi_hot_labels, power=1.0):
    """
    Compute inverse frequency weights for each token count with power scaling.

    Args:
        multi_hot_labels: (N, num_classes) tensor
        power: Power to raise weights to (>1.0 for stronger rare class emphasis)

    Returns:
        class_weights: (num_classes,) tensor
    """
    # Count frequency of each token count across all samples
    token_counts = multi_hot_labels.sum(dim=0)  # Shape: [7]
    total_positives = token_counts.sum()

    # Inverse frequency weighting
    class_weights = total_positives / (token_counts + 1e-6)

    # Apply power scaling to amplify rare class weights
    if power != 1.0:
        class_weights = class_weights ** power

    # Normalize so weights sum to num_classes
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    return class_weights


def train_multi_target_predictor(
    data_path: str,
    output_path: str,
    loss_type: str = "kl",
    architecture: str = "mlp",
    hidden_layers: list = [1024, 512],
    hidden_dim: int = 512,
    num_blocks: int = 6,
    num_tokens: int = 12,
    d_model: int = 64,
    num_heads: int = 8,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.3,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_epochs: int = 20,
    val_ratio: float = 0.15,
    weight_power: float = 1.0,
    focal_gamma: float = 2.0,
    token_penalty: float = 0.0,
    device: str = "cuda",
):
    """
    Train multi-target predictor.
    """
    print("=" * 80)
    print("TRAINING MULTI-TARGET PREDICTOR")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from {data_path}...")
    data = torch.load(data_path)

    hidden_states = data["hidden_states"]
    multi_hot_labels = data["multi_hot_labels"]
    num_steps = data["num_steps"]

    # Check for multistep dataset (has step_indices and sample_weights)
    is_multistep = "step_indices" in data and "sample_weights" in data
    if is_multistep:
        step_indices = data["step_indices"]
        sample_weights = data["sample_weights"]
        print(f"  Dataset type: MULTISTEP (includes h0, h1, h2, ...)")
    else:
        step_indices = None
        sample_weights = None
        print(f"  Dataset type: H0-ONLY")

    print(f"  Total samples: {len(hidden_states)}")
    print(f"  Hidden dim: {hidden_states.shape[1]}")
    print(f"  Num classes: {num_steps}")

    # Print label statistics
    num_correct_per_sample = multi_hot_labels.sum(dim=1)
    print(f"\nLabel statistics:")
    print(f"  Min correct per sample: {num_correct_per_sample.min().item():.0f}")
    print(f"  Max correct per sample: {num_correct_per_sample.max().item():.0f}")
    print(f"  Avg correct per sample: {num_correct_per_sample.mean().item():.2f}")

    # Distribution of number of correct token counts
    unique_counts, counts = torch.unique(num_correct_per_sample, return_counts=True)
    print(f"\nDistribution of # correct token counts:")
    for count_val, freq in zip(unique_counts.tolist(), counts.tolist()):
        pct = 100 * freq / len(hidden_states)
        print(f"  {int(count_val)} correct: {freq:5d} ({pct:5.1f}%)")

    # Multistep statistics
    if is_multistep:
        print(f"\nMultistep dataset statistics:")
        for step in range(num_steps):
            step_count = (step_indices == step).sum().item()
            step_pct = 100 * step_count / len(hidden_states)
            step_avg_weight = sample_weights[step_indices == step].mean().item() if step_count > 0 else 0
            print(f"  Step {step}: {step_count:5d} ({step_pct:5.1f}%) | avg_weight={step_avg_weight:.4f}")

    # Split train/val
    total_n = len(hidden_states)
    n_val = max(1, int(total_n * val_ratio))
    n_train = total_n - n_val

    perm = torch.randperm(total_n)
    train_hidden = hidden_states[perm[:n_train]]
    train_labels = multi_hot_labels[perm[:n_train]]
    val_hidden = hidden_states[perm[n_train:]]
    val_labels = multi_hot_labels[perm[n_train:]]

    # Split multistep-specific data if available
    if is_multistep:
        train_step_indices = step_indices[perm[:n_train]]
        train_sample_weights = sample_weights[perm[:n_train]]
        val_step_indices = step_indices[perm[n_train:]]
        val_sample_weights = sample_weights[perm[n_train:]]
    else:
        train_step_indices = None
        train_sample_weights = None
        val_step_indices = None
        val_sample_weights = None

    print(f"\nTrain samples: {n_train}")
    print(f"Val samples: {n_val}")

    # Create data loaders
    if is_multistep:
        train_loader = DataLoader(
            TensorDataset(train_hidden, train_labels, train_step_indices, train_sample_weights),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_hidden, val_labels, val_step_indices, val_sample_weights),
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        train_loader = DataLoader(
            TensorDataset(train_hidden, train_labels),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_hidden, val_labels),
            batch_size=batch_size,
            shuffle=False,
        )

    # Create model
    print(f"\nCreating multi-target predictor...")
    print(f"  Architecture: {architecture}")
    if architecture == "mlp":
        print(f"  Hidden layers: {hidden_layers}")
    elif architecture == "resnet":
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num blocks: {num_blocks}")
    elif architecture == "transformer":
        print(f"  Num tokens: {num_tokens}")
        print(f"  d_model: {d_model}")
        print(f"  Num heads: {num_heads}")
        print(f"  Num layers: {num_layers}")
        print(f"  Dim feedforward: {dim_feedforward}")
    print(f"  Dropout: {dropout}")
    print(f"  Loss type: {loss_type}")
    if token_penalty != 0.0:
        print(f"  Token penalty: {token_penalty} ({'penalize' if token_penalty > 0 else 'reward'} higher tokens)")

    model = create_multi_target_predictor(
        input_dim=hidden_states.shape[1],
        num_classes=num_steps,
        hidden_layers=hidden_layers,
        dropout=dropout,
        architecture=architecture,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_tokens=num_tokens,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Loss function
    if loss_type == "kl":
        criterion = MultiTargetKLLoss()
    elif loss_type == "bce":
        criterion = MultiTargetBCELoss()
    elif loss_type == "weighted":
        # Compute class weights for balanced training
        # IMPORTANT: For multistep datasets, use ONLY h0 samples to compute weights
        # This ensures we learn the h0 distribution (what matters at test time)
        if is_multistep:
            h0_mask = train_step_indices == 0
            h0_labels = train_labels[h0_mask]
            class_weights = compute_class_weights(h0_labels, power=weight_power)
            print(f"\nClass weights from h0 ONLY (inverse frequency with power={weight_power}):")
        else:
            class_weights = compute_class_weights(train_labels, power=weight_power)
            print(f"\nClass weights (inverse frequency with power={weight_power}):")

        for i, weight in enumerate(class_weights.tolist()):
            if is_multistep:
                token_count = h0_labels[:, i].sum().item()
                print(f"  Token {i}: weight={weight:.3f} (appears in {token_count:.0f} h0 samples)")
            else:
                token_count = train_labels[:, i].sum().item()
                print(f"  Token {i}: weight={weight:.3f} (appears in {token_count:.0f} samples)")
        criterion = WeightedMultiTargetLoss(class_weights.to(device))
    elif loss_type == "weighted_focal":
        # Compute class weights for balanced training
        # IMPORTANT: For multistep datasets, use ONLY h0 samples to compute weights
        if is_multistep:
            h0_mask = train_step_indices == 0
            h0_labels = train_labels[h0_mask]
            class_weights = compute_class_weights(h0_labels, power=weight_power)
            print(f"\nClass weights from h0 ONLY (inverse frequency with power={weight_power}):")
        else:
            class_weights = compute_class_weights(train_labels, power=weight_power)
            print(f"\nClass weights (inverse frequency with power={weight_power}):")

        for i, weight in enumerate(class_weights.tolist()):
            if is_multistep:
                token_count = h0_labels[:, i].sum().item()
                print(f"  Token {i}: weight={weight:.3f} (appears in {token_count:.0f} h0 samples)")
            else:
                token_count = train_labels[:, i].sum().item()
                print(f"  Token {i}: weight={weight:.3f} (appears in {token_count:.0f} samples)")
        print(f"Focal loss gamma: {focal_gamma}")
        criterion = WeightedFocalLoss(class_weights.to(device), gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    print(f"\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")
    best_val_any_correct = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            if is_multistep:
                batch_hidden, batch_labels, batch_step_indices, batch_sample_weights = batch
                batch_step_indices = batch_step_indices.to(device)
                batch_sample_weights = batch_sample_weights.to(device)
            else:
                batch_hidden, batch_labels = batch
                batch_step_indices = None
                batch_sample_weights = None

            batch_hidden = batch_hidden.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            logits = model(batch_hidden)

            # Mask invalid targets for multistep training (don't mask logits)
            if is_multistep:
                # Clone labels to avoid modifying original
                masked_labels = batch_labels.clone()
                for i, step in enumerate(batch_step_indices):
                    remaining = num_steps - 1 - step  # max possible remaining tokens
                    if remaining < num_steps - 1:
                        # Mask targets for invalid classes (loss will ignore these)
                        masked_labels[i, remaining+1:] = 0.0
            else:
                masked_labels = batch_labels

            # Compute loss (per-sample)
            loss_per_sample = criterion(logits, masked_labels)

            # Apply sample weights if available
            if is_multistep:
                loss = (loss_per_sample * batch_sample_weights).mean()
            else:
                loss = loss_per_sample.mean()

            # Add token count regularization
            if token_penalty != 0.0:
                probs = torch.sigmoid(logits)  # (batch_size, num_classes)
                token_indices = torch.arange(num_steps, dtype=torch.float32, device=logits.device)
                expected_token_count = (probs * token_indices).sum(dim=1).mean()
                loss = loss + token_penalty * expected_token_count

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_any_correct = 0  # Predicted token is ANY of the correct choices
        val_h0_any_correct = 0  # h0-only accuracy (for multistep datasets)
        val_total = 0
        val_h0_total = 0

        with torch.no_grad():
            for batch in val_loader:
                if is_multistep:
                    batch_hidden, batch_labels, batch_step_indices, batch_sample_weights = batch
                    batch_step_indices = batch_step_indices.to(device)
                    batch_sample_weights = batch_sample_weights.to(device)
                else:
                    batch_hidden, batch_labels = batch
                    batch_step_indices = None
                    batch_sample_weights = None

                batch_hidden = batch_hidden.to(device)
                batch_labels = batch_labels.to(device)

                # Forward pass
                logits = model(batch_hidden)

                # Mask invalid targets for multistep training (don't mask logits)
                if is_multistep:
                    # Clone labels to avoid modifying original
                    masked_labels = batch_labels.clone()
                    for i, step in enumerate(batch_step_indices):
                        remaining = num_steps - 1 - step
                        if remaining < num_steps - 1:
                            # Mask targets for invalid classes (loss will ignore these)
                            masked_labels[i, remaining+1:] = 0.0
                else:
                    masked_labels = batch_labels

                # Compute loss (per-sample)
                loss_per_sample = criterion(logits, masked_labels)
                if is_multistep:
                    loss = (loss_per_sample * batch_sample_weights).mean()
                else:
                    loss = loss_per_sample.mean()
                val_loss += loss.item()

                # Get predictions (argmax) - mask invalid logits before argmax
                if is_multistep:
                    # Clone logits for prediction
                    pred_logits = logits.clone()
                    for i, step in enumerate(batch_step_indices):
                        remaining = num_steps - 1 - step
                        if remaining < num_steps - 1:
                            # Mask invalid classes before argmax
                            pred_logits[i, remaining+1:] = float('-inf')
                    pred_tokens = pred_logits.argmax(dim=1)
                else:
                    pred_tokens = logits.argmax(dim=1)

                # Check if prediction is ANY correct token
                for i in range(len(pred_tokens)):
                    pred_token = pred_tokens[i].item()
                    correct_tokens = torch.where(batch_labels[i] > 0)[0].tolist()

                    if pred_token in correct_tokens:
                        val_any_correct += 1

                    # Track h0-only accuracy for multistep datasets
                    if is_multistep and batch_step_indices[i].item() == 0:
                        if pred_token in correct_tokens:
                            val_h0_any_correct += 1
                        val_h0_total += 1

                val_total += len(batch_labels)

        avg_val_loss = val_loss / len(val_loader)
        val_any_acc = val_any_correct / val_total
        val_h0_acc = val_h0_any_correct / val_h0_total if val_h0_total > 0 else 0.0

        # Update scheduler
        scheduler.step(avg_val_loss)

        # Print epoch results
        if is_multistep:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"train_loss: {avg_train_loss:.4f}, "
                f"val_loss: {avg_val_loss:.4f}, "
                f"val_any_correct: {val_any_acc:.4f}, "
                f"val_h0_correct: {val_h0_acc:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"train_loss: {avg_train_loss:.4f}, "
                f"val_loss: {avg_val_loss:.4f}, "
                f"val_any_correct: {val_any_acc:.4f}"
            )

        # Save best model (by validation accuracy)
        if val_any_acc > best_val_any_correct:
            best_val_any_correct = val_any_acc
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "input_dim": hidden_states.shape[1],
                        "num_classes": num_steps,
                        "architecture": architecture,
                        "hidden_layers": hidden_layers,
                        "hidden_dim": hidden_dim,
                        "num_blocks": num_blocks,
                        "num_tokens": num_tokens,
                        "d_model": d_model,
                        "num_heads": num_heads,
                        "num_layers": num_layers,
                        "dim_feedforward": dim_feedforward,
                        "dropout": dropout,
                        "loss_type": loss_type,
                        "token_penalty": token_penalty,
                    },
                    "metrics": {
                        "val_loss": avg_val_loss,
                        "val_any_correct": val_any_acc,
                        "val_h0_correct": val_h0_acc if is_multistep else val_any_acc,
                        "is_multistep": is_multistep,
                    },
                    "num_steps": num_steps,
                },
                output_path,
            )

    print(f"\nTraining complete!")
    print(f"Best epoch: {best_epoch} (selected by highest validation accuracy)")
    print(f"Best val 'any correct' accuracy: {best_val_any_correct:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train multi-target predictor"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="multi_target_dataset.pt",
        help="Path to dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="multi_target_predictor.pt",
        help="Path to save model",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="kl",
        choices=["kl", "bce", "weighted", "weighted_focal"],
        help="Loss function type (weighted = class-balanced BCE, weighted_focal = weighted + focal loss)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="mlp",
        choices=["mlp", "resnet", "transformer", "attention", "moe"],
        help="Model architecture (mlp=simple MLP, resnet=residual MLP, transformer=self-attention)",
    )
    parser.add_argument(
        "--hidden_layers",
        type=str,
        default="1024,512",
        help="Comma-separated hidden layer sizes (for MLP architecture)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension (for ResNet architecture)",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=6,
        help="Number of residual blocks (for ResNet architecture)",
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=12,
        help="Number of tokens to reshape input into (for Transformer architecture)",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Token dimension (for Transformer architecture)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads (for Transformer architecture)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of transformer encoder layers (for Transformer architecture)",
    )
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=256,
        help="Feedforward dimension (for Transformer architecture)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of epochs",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--weight_power",
        type=float,
        default=1.0,
        help="Power scaling for class weights (>1.0 for stronger rare class emphasis)",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (higher = more focus on hard examples)",
    )
    parser.add_argument(
        "--token_penalty",
        type=float,
        default=0.0,
        help="Regularization penalty for token count (>0 penalizes higher, <0 rewards higher, default: 0.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    train_multi_target_predictor(
        data_path=args.data_path,
        output_path=args.output_path,
        loss_type=args.loss_type,
        architecture=args.architecture,
        hidden_layers=hidden_layers,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_tokens=args.num_tokens,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        val_ratio=args.val_ratio,
        weight_power=args.weight_power,
        focal_gamma=args.focal_gamma,
        token_penalty=args.token_penalty,
        device=device,
    )


if __name__ == "__main__":
    main()

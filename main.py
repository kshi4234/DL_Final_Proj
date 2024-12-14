from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
#from models import MockModel
from models import JEPAModel
import glob
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import random

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds

def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    #model = MockModel()
    model = JEPAModel(device=device)
    model.to(device)
    try:
        model.load_state_dict(torch.load('jepa_model.pth'))
        print("Loaded saved JEPA model.")
    except FileNotFoundError:
        print("No saved model found, initializing a new model.")
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()

    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


def je_loss(predictions, targets):
    similarities = F.cosine_similarity(predictions, targets, dim=-1)  # [B, T]
    loss_per_sample = 1 - similarities  # [B, T]
    # Now sum over the trajectory dimension T
    loss_per_sample = loss_per_sample.sum(dim=1)  # [B]
    # Then, if desired, average over the batch
    loss = loss_per_sample.mean()  # scalar
    return loss

def barlow_twins_loss(z1, z2, lambda_param=0.0051):
    # Flatten time dimension into batch dimension
    B, T, D = z1.shape
    z1_flat = z1.view(-1, D)  # [B*T, D]
    z2_flat = z2.view(-1, D)  # [B*T, D]

    # Normalize representations along the batch dimension
    z1_norm = (z1_flat - z1_flat.mean(0)) / (z1_flat.std(0) + 1e-5)
    z2_norm = (z2_flat - z2_flat.mean(0)) / (z2_flat.std(0) + 1e-5)

    # Cross-correlation matrix
    batch_size = z1_norm.shape[0]
    c = torch.matmul(z1_norm.T, z2_norm) / batch_size

    # Loss terms
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = c.flatten()[:-1].view(c.size(0) - 1, c.size(1) + 1)[:, 1:].pow_(2).sum()

    loss = on_diag + lambda_param * off_diag
    return loss

def apply_augmentation(states):
    # states: [B, T, C, H, W]
    B, T, C, H, W = states.shape

    aug_states = []
    for b in range(B):
        sample = states[b]  # [T, C, H, W]

        # Decide on aug parameters (flip, rotate, etc.)
        do_hflip = random.random() < 0.5
        do_vflip = random.random() < 0.5
        angles = [0, 90, 180, 270]
        angle = random.choice(angles)

        # Apply same augmentation to each frame in the sequence
        aug_frames = []
        for t in range(T):
            frame = sample[t]  # [C, H, W]
            if do_hflip:
                frame = VF.hflip(frame)
            if do_vflip:
                frame = VF.vflip(frame)
            frame = VF.rotate(frame, angle)

            # Add noise
            noise = torch.randn_like(frame) * 0.01
            frame = frame + noise

            aug_frames.append(frame)
        aug_frames = torch.stack(aug_frames, dim=0)  # [T, C, H, W]
        aug_states.append(aug_frames)

    aug_states = torch.stack(aug_states, dim=0)  # [B, T, C, H, W]
    return aug_states

def train_model(device):
    # Load training data
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        train=True,
    )

    model = JEPAModel(device=device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 1
    max_iterations = 500
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            if batch_idx >= max_iterations:
                break
            states = batch.states  # [B, T, C, H, W]
            actions = batch.actions  # [B, T-1, action_dim]

            predictions = model(states, actions)  # [B, T, D]
            # Compute target representations using the target encoder
            with torch.no_grad():
                targets = model.encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)  # [B, T, D]

            # Compute loss between predictions[:, 1:] and targets[:, 1:]
            jepa_loss = je_loss(predictions[:, 1:], targets[:, 1:])

            aug_states = apply_augmentation(states)  # [B, T, C, H, W]
            with torch.no_grad():
                z1 = model.encoder(
                    aug_states.view(-1, *aug_states.shape[2:])
                ).view(aug_states.size(0), aug_states.size(1), -1)  # [B, T, D]

            z2 = targets
            bt_loss = barlow_twins_loss(z1, z2)

            total_loss_batch = 0.6 * jepa_loss + 0.4 * bt_loss
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            total_loss += total_loss_batch.item()
            if batch_idx % 100 == 0:
                print(f"Batch [{batch_idx}], Total Loss: {total_loss_batch:.4f}, "
                      f"JEPA Loss: {jepa_loss:.4f}, BT Loss: {bt_loss:.4f}")
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8e}")

        # Optionally evaluate the model
        #if (epoch + 1) % 2 == 0:
            #evaluate_current_model(model, device)

    # Save the trained model
    torch.save(model.state_dict(), 'jepa_model.pth')


def evaluate_current_model(model, device):
    probe_train_ds, probe_val_ds = load_data(device)
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)
    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()
    train_model(device)

    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)

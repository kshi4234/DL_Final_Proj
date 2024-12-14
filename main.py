from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
#from models import MockModel
from models import JEPAModel
import glob
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

data_augmentation = T.Compose([
    T.RandomCrop(65, padding=4),               # Spatial cropping
    T.RandomHorizontalFlip(p=0.5),            # Random horizontal flip
    T.RandomVerticalFlip(p=0.5),              # Random vertical flip
    T.RandomRotation(degrees=15),             # Small rotations
    T.ColorJitter(brightness=0.2, contrast=0.2), # Adjust brightness and contrast
    T.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),  # Gaussian noise
])

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
    loss = 1 - F.cosine_similarity(predictions, targets, dim=-1).mean()
    return loss

def barlow_twins_loss(z1, z2, lambda_coeff=5e-3):
    """
    Compute Barlow Twins loss.
    Args:
        z1: First set of embeddings (view 1), shape [B, D].
        z2: Second set of embeddings (view 2), shape [B, D].
        lambda_coeff: Weighting coefficient for off-diagonal elements.
    """
    # Normalize representations
    z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
    z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)

    # Compute cross-correlation matrix
    batch_size = z1.size(0)
    c = torch.mm(z1.T, z2) / batch_size

    # Invariance term: Penalize deviation from diagonal=1
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()

    # Redundancy reduction: Penalize off-diagonal elements
    off_diag = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())

    return on_diag + lambda_coeff * off_diag

def train_model(device):
    # Load training data
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        train=True,
    )

    model = JEPAModel(device=device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            states = batch.states  # [B, T, C, H, W]
            actions = batch.actions  # [B, T-1, action_dim]

            view1 = torch.stack([data_augmentation(state) for state in states.view(-1, *states.shape[2:])])
            view2 = torch.stack([data_augmentation(state) for state in states.view(-1, *states.shape[2:])])

            view1 = view1.view(states.size())  # Reshape to original dimensions
            view2 = view2.view(states.size())

            z1 = model.encoder(view1)
            z2 = model.encoder(view2)
            bt_loss = barlow_twins_loss(z1, z2)

            predictions = model(states, actions)  # [B, T, D]

            # Compute target representations using the target encoder
            with torch.no_grad():
                targets = model.encoder(
                    states.view(-1, *states.shape[2:])
                ).view(states.size(0), states.size(1), -1)  # [B, T, D]

            # Compute loss between predictions[:, 1:] and targets[:, 1:]
            jepa_loss = je_loss(predictions[:, 1:], targets[:, 1:])

            total_batch_loss = 0.6 * jepa_loss + 0.4 * bt_loss

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, JEPA Loss: {jepa_loss.item():.8e}, BT Loss: {bt_loss.item():.8e}, Total Loss: {total_batch_loss.item():.8e}")
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

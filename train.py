import torch
from dataset import create_wall_dataloader
from models import SimpleJEPA
import torch.optim as optim
from tqdm import tqdm


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        batch_size=32,
        train=True
    )
    model = SimpleJEPA().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in pbar:
            optimizer.zero_grad()
            states = batch.states
            actions = batch.actions
            loss = model.compute_loss(states, actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'jepa_checkpoint_epoch_{epoch + 1}.pt')


if __name__ == "__main__":
    train()

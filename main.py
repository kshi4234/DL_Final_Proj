from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
#from models import MockModel
from models import JEPAModel
import glob
from tqdm import tqdm
import torch.nn.functional as F
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

def train_model(device):
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        batch_size=64,
        train=True,
    )

    model = JEPAModel(device=device)
    model.to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.online_encoder.parameters(), "lr": 1e-4},
        {"params": model.predictor.parameters(), "lr": 1e-4}
    ], weight_decay=0.05)

    # 使用OneCycleLR学习率调度
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=30,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    num_epochs = 1
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            states = batch.states
            actions = batch.actions

            optimizer.zero_grad()
            predictions, online_states, target_states = model(states, actions)
            
            # 计算损失
            loss = compute_loss(predictions[:, 1:], target_states[:, 1:], online_states)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 更新目标编码器
            model.update_target_encoder()

            total_loss += loss.item()

            # 每100个batch打印一次loss
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {avg_loss:.6f}")

        # 打印每个epoch的平均loss
        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.6f}")

    # 只保存最终模型
    torch.save(model.state_dict(), 'jepa_model.pth')
    return model

def compute_loss(pred_states, target_states, online_states=None):
    """计算损失"""
    # L2距离在归一化空间
    pred_norm = F.normalize(pred_states, dim=-1)
    target_norm = F.normalize(target_states, dim=-1)
    prediction_loss = F.mse_loss(pred_norm, target_norm)
    
    # 余弦相似度损失
    cos_sim = F.cosine_similarity(pred_states, target_states, dim=-1).mean()
    cos_loss = 1 - cos_sim
    
    # 组合损失
    total_loss = prediction_loss + cos_loss
    
    if online_states is not None:
        # 添加状态连续性约束
        state_diff = online_states[:, 1:] - online_states[:, :-1]
        consistency_loss = torch.mean(torch.abs(state_diff))
        total_loss = total_loss + 0.1 * consistency_loss
        
    return total_loss


if __name__ == "__main__":
    device = get_device()
    train_model(device)

    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
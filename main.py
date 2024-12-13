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

def je_loss(predictions, targets):
    #predictions = F.normalize(predictions, dim=-1)
    #targets = F.normalize(targets, dim=-1)
    
    #loss = F.mse_loss(predictions, targets)
    loss = 1 - F.cosine_similarity(predictions, targets, dim=-1).mean()
    return loss

def train_model(device):
    # 加载训练数据
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        train=True,
    )

    model = JEPAModel(device=device)
    model.to(device)

    # 使用AdamW优化器并添加权重衰减
    optimizer = torch.optim.AdamW([
        {"params": model.online_encoder.parameters(), "lr": 1e-4, "weight_decay": 0.05},
        {"params": model.online_projector.parameters(), "lr": 1e-4, "weight_decay": 0.05},
        {"params": model.online_predictor.parameters(), "lr": 1e-4, "weight_decay": 0.05},
        {"params": model.predictor.parameters(), "lr": 1e-4, "weight_decay": 0.05},
    ])

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    num_epochs = 1
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Training Epoch {epoch+1}") as bar:
            for batch_idx, batch in enumerate(bar):
                states = batch.states
                actions = batch.actions

                optimizer.zero_grad()
                predictions, online_preds, targets = model(states, actions)

                # 计算BYOL损失
                byol_loss = 1 - F.cosine_similarity(online_preds, targets.detach(), dim=-1).mean()
                
                # 计算预测损失
                with torch.no_grad():
                    target_reps = model.target_encoder(
                        states.view(-1, *states.shape[2:])
                    ).view(states.size(0), states.size(1), -1)
                
                pred_loss = F.smooth_l1_loss(predictions[:, 1:], target_reps[:, 1:])
                
                # 总损失
                loss = byol_loss + pred_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 更新目标编码器
                model.update_target_encoder()

                total_loss += loss.item()
                bar.set_postfix(batch_num=batch_idx, loss=loss.item(), 
                              byol_loss=byol_loss.item(), pred_loss=pred_loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.8e}")
        
        # 更新学习率
        scheduler.step()
        

    # 保存最终模型
    torch.save(model.state_dict(), 'final_jepa_model.pth')




if __name__ == "__main__":
    device = get_device()
    train_model(device)

    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)

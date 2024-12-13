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
    # 数据加载器
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        batch_size=32,  # 减小batch size以增加更新频率
        train=True,
    )

    model = JEPAModel(device=device)
    model.to(device)

    # 使用AdamW优化器
    optimizer = torch.optim.AdamW([
        {"params": model.online_encoder.parameters(), "lr": 2e-4},
        {"params": model.predictor.parameters(), "lr": 2e-4},
    ], weight_decay=0.01)

    # 余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-6
    )
    
    num_epochs = 20
    best_energy = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_energy = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                states = batch.states
                actions = batch.actions

                optimizer.zero_grad()
                
                # 计算系统能量
                energy, _ = model(states, actions)
                
                # 反向传播
                energy.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                
                optimizer.step()
                
                # 更新目标编码器
                model.update_target_encoder()

                total_energy += energy.item()
                pbar.set_postfix(
                    energy=energy.item(),
                    avg_energy=total_energy/(batch_idx+1)
                )

        # 更新学习率
        scheduler.step()
        
        # 每轮结束后评估
        #val_energy = evaluate_model(model, device)
        #print(f"Validation Energy: {val_energy:.4f}")
        
        # 保存最佳模型
        # if val_energy < best_energy:
        #     best_energy = val_energy
        #     torch.save(model.state_dict(), 'best_jepa_model.pth')
        #     print(f"Saved best model with energy: {best_energy:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), 'final_jepa_model.pth')
    return model

# def evaluate_model(model, device):
#     """评估模型性能"""
#     model.eval()
#     probe_train_ds, probe_val_ds = load_data(device)
    
#     total_energy = 0
#     num_batches = 0
    
#     with torch.no_grad():
#         for batch in probe_val_ds['normal']:
#             energy, _ = model(batch.states, batch.actions)
#             total_energy += energy.item()
#             num_batches += 1
            
#     return total_energy / num_batches


if __name__ == "__main__":
    device = get_device()
    train_model(device)

    probe_train_ds, probe_val_ds = load_data(device)
    model = load_model()
    evaluate_model(device, model, probe_train_ds, probe_val_ds)
# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

BOS_IDX = 2   # 0, 1: 실제 비트 / 2: BOS 토큰


# ---------------------------
# Seed 고정
# ---------------------------
def set_seed(seed: int):
    print(f"🔒 Setting global seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------
# Canonicalization for y-axis symmetry (좌우 대칭)
# ---------------------------
def horizontal_flip_y_axis(X_flat, height, width):
    N, num_bits = X_flat.shape
    assert num_bits == height * width
    X_reshaped = X_flat.reshape(N, height, width)
    X_flipped = X_reshaped[:, :, ::-1]
    return X_flipped.reshape(N, num_bits)


def canonicalize_under_yflip(X_flat, height, width):
    X_flat = X_flat.copy()
    X_flip = horizontal_flip_y_axis(X_flat, height, width)
    N = X_flat.shape[0]
    X_can = np.empty_like(X_flat)

    for i in range(N):
        a = X_flat[i]
        b = X_flip[i]
        X_can[i] = a if list(a) <= list(b) else b
    return X_can


# ---------------------------
# Hilbert curve utilities
# ---------------------------
def _hilbert_rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def _hilbert_d2xy(n, d):
    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def get_order_indices(ordering, num_bits, height, width):
    assert num_bits == height * width, "num_bits must be H*W"

    if ordering == "raster":
        order = np.arange(num_bits, dtype=np.int64)

    elif ordering == "snake":
        order = []
        for r in range(height):
            cols = range(width) if r % 2 == 0 else reversed(range(width))
            for c in cols:
                order.append(r * width + c)
        order = np.array(order, dtype=np.int64)

    elif ordering == "hilbert":
        max_side = max(height, width)
        n_side = 1
        while n_side < max_side:
            n_side *= 2

        coords = []
        for d in range(n_side * n_side):
            x, y = _hilbert_d2xy(n_side, d)
            if x < width and y < height:
                coords.append((x, y))
            if len(coords) == num_bits:
                break

        assert len(coords) == num_bits, f"Hilbert coords length {len(coords)} != num_bits {num_bits}"
        order = np.array([y * width + x for (x, y) in coords], dtype=np.int64)

    else:
        raise ValueError(f"Unknown ordering: {ordering}")

    assert len(order) == num_bits
    return order


# ---------------------------
# 1D ResNet encoder for spectral condition
# ---------------------------
class ResNet1DEncoder(nn.Module):
    def __init__(self, d_model, num_blocks=3, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.input_proj = nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=padding)
        self.input_bn = nn.BatchNorm1d(d_model)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(d_model),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(d_model),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.act = nn.ReLU(inplace=True)

    def forward(self, y):
        x = self.input_proj(y)
        x = self.input_bn(x)
        x = self.act(x)

        for block in self.blocks:
            residual = x
            out = block(x)
            x = self.act(out + residual)

        x = x.mean(dim=-1)
        return x


# ---------------------------
# Autoregressive Transformer
# ---------------------------
class SmallTransformerAR(nn.Module):
    def __init__(
        self,
        num_points,
        d_model=512,
        nhead=4,
        num_layers=15,
        dim_feedforward=768,
        max_len=100,
        vocab_size=3,
        dropout=0.1,
        spectral_cond="resnet1d",
        use_2d_pos=False,
        chain2spatial=None,
        height=10,
        width=10,
    ):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model
        self.max_len = max_len
        self.spectral_cond_type = spectral_cond
        self.use_2d_pos = use_2d_pos
        self.height = height
        self.width = width

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        if self.use_2d_pos:
            assert chain2spatial is not None, "use_2d_pos=True이면 chain2spatial이 필요합니다."
            assert chain2spatial.numel() >= max_len
            self.pos2d_embed = nn.Embedding(height * width, d_model)
            self.register_buffer("chain2spatial", chain2spatial.clone())

        if spectral_cond == "linear":
            self.cond_encoder = nn.Linear(num_points, d_model)
        elif spectral_cond == "mlp":
            self.cond_encoder = nn.Sequential(
                nn.Linear(num_points, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        elif spectral_cond == "transformer":
            self.freq_in_proj = nn.Linear(1, d_model)
            self.freq_pos_embed = nn.Embedding(num_points, d_model)
            cond_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.cond_transformer = nn.TransformerEncoder(cond_encoder_layer, num_layers=2)
        elif spectral_cond == "resnet1d":
            self.resnet1d = ResNet1DEncoder(d_model=d_model, num_blocks=3, kernel_size=3)
        else:
            raise ValueError(f"Unknown spectral_cond: {spectral_cond}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def _generate_causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def encode_condition(self, y):
        if self.spectral_cond_type in ["linear", "mlp"]:
            return self.cond_encoder(y)

        if self.spectral_cond_type == "transformer":
            B, P = y.shape
            device = y.device
            y_seq = y.unsqueeze(-1)
            feat = self.freq_in_proj(y_seq)

            pos_idx = torch.arange(P, device=device).unsqueeze(0).expand(B, P)
            pos_emb = self.freq_pos_embed(pos_idx)
            feat = feat + pos_emb
            h = self.cond_transformer(feat)
            return h.mean(dim=1)

        if self.spectral_cond_type == "resnet1d":
            y_seq = y.unsqueeze(1)
            return self.resnet1d(y_seq)

        raise RuntimeError("Invalid spectral_cond_type")

    def forward(self, y, tokens):
        B, L = tokens.shape
        device = tokens.device

        tok_emb = self.token_embed(tokens)

        pos_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_embed(pos_idx)

        if self.use_2d_pos:
            spatial_idx = self.chain2spatial[:L].to(device)
            spatial_idx = spatial_idx.unsqueeze(0).expand(B, L)
            pos2d_emb = self.pos2d_embed(spatial_idx)
            pos_emb = pos_emb + pos2d_emb

        cond_vec = self.encode_condition(y)
        cond = cond_vec.unsqueeze(1).expand(B, L, self.d_model)

        x = tok_emb + pos_emb + cond
        src_mask = self._generate_causal_mask(L, device=device)
        h = self.transformer(x, mask=src_mask)
        logits = self.fc_out(h).squeeze(-1)
        return logits


# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch_transformer(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_bit_acc = 0.0
    total_pattern_acc = 0.0

    for yb, xb in loader:
        yb = yb.to(device)
        xb = xb.to(device)
        B, _ = xb.shape

        bos = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
        x_prev = xb[:, :-1].long()
        tokens_in = torch.cat([bos, x_prev], dim=1)
        targets = xb

        optimizer.zero_grad()
        logits = model(yb, tokens_in)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            bit_correct = (preds == targets).float().mean().item()
            pattern_correct = (preds == targets).all(dim=1).float().mean().item()

            total_bit_acc += bit_correct * B
            total_pattern_acc += pattern_correct * B

    avg_loss = total_loss / total_samples
    avg_bit_acc = total_bit_acc / total_samples
    avg_pattern_acc = total_pattern_acc / total_samples
    return avg_loss, avg_bit_acc, avg_pattern_acc


def eval_transformer(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_bit_acc = 0.0
    total_pattern_acc = 0.0

    with torch.no_grad():
        for yb, xb in loader:
            yb = yb.to(device)
            xb = xb.to(device)
            B, _ = xb.shape

            bos = torch.full((B, 1), BOS_IDX, dtype=torch.long, device=device)
            x_prev = xb[:, :-1].long()
            tokens_in = torch.cat([bos, x_prev], dim=1)
            targets = xb

            logits = model(yb, tokens_in)
            loss = criterion(logits, targets)

            total_loss += loss.item() * B
            total_samples += B

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            bit_correct = (preds == targets).float().mean().item()
            pattern_correct = (preds == targets).all(dim=1).float().mean().item()

            total_bit_acc += bit_correct * B
            total_pattern_acc += pattern_correct * B

    avg_loss = total_loss / total_samples
    avg_bit_acc = total_bit_acc / total_samples
    avg_pattern_acc = total_pattern_acc / total_samples
    return avg_loss, avg_bit_acc, avg_pattern_acc


# ---------------------------
# Utils
# ---------------------------
def build_run_name(args):
    return (
        f"{args.run_prefix}-"
        f"L{args.num_layers}-d{args.d_model}-h{args.nhead}-ff{args.dim_ff}-"
        f"dr{args.dropout}-ord{args.ordering}-spec{args.spectral_cond}-"
        f"2dpos{int(args.use_2d_pos)}-canon{int(args.canonical_target)}-"
        f"lr{args.lr}-bs{args.batch_size}-seed{args.seed}"
    )


def maybe_init_wandb(args, run_name):
    if not args.use_wandb:
        print("ℹ️ W&B logging is disabled by argument.")
        return False

    if not WANDB_AVAILABLE:
        print("⚠️ wandb is not installed. Training will continue without W&B logging.")
        return False

    try:
        wandb.init(
            project=args.project,
            name=run_name,
            settings=wandb.Settings(init_timeout=180)
        )
        wandb.config.update(vars(args))
        return True

    except Exception as e:
        print(f"⚠️ W&B init failed: {e}")
        print("⚠️ Training will continue without W&B logging.")
        return False


# ---------------------------
# Main
# ---------------------------
def main(args):
    # 각 ordering run마다 동일 seed에서 새로 시작
    set_seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.abspath(os.path.join(script_dir, args.data_root))
    if not os.path.isabs(args.save_dir):
        args.save_dir = os.path.abspath(os.path.join(script_dir, args.save_dir))

    print(f"📁 Script directory : {script_dir}")
    print(f"📁 Data root        : {args.data_root}")
    print(f"📁 Save dir         : {args.save_dir}")

    expected_files = [
        os.path.join(args.data_root, "dataset_train.npz"),
        os.path.join(args.data_root, "dataset_valid.npz"),
        os.path.join(args.data_root, "dataset_test.npz"),
    ]
    for fp in expected_files:
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"Required dataset file not found:\n{fp}\n\n"
                f"Please create a folder named 'dataset' in the same directory as this script "
                f"and place dataset_train.npz / dataset_valid.npz / dataset_test.npz inside it."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")

    train_npz = np.load(expected_files[0])
    valid_npz = np.load(expected_files[1])
    test_npz = np.load(expected_files[2])

    X_train_2d = train_npz["X"].astype(np.float32)
    Y_train = train_npz["Y"].astype(np.float32)
    X_val_2d = valid_npz["X"].astype(np.float32)
    Y_val = valid_npz["Y"].astype(np.float32)
    X_test_2d = test_npz["X"].astype(np.float32)
    Y_test = test_npz["Y"].astype(np.float32)

    freq = train_npz["freq"].astype(np.float32)
    print("Using original resolution:", len(freq), "points")

    N_train, H, W = X_train_2d.shape
    num_points = Y_train.shape[1]
    num_bits = H * W
    print(
        f"✅ Dataset loaded: train={len(X_train_2d)}, val={len(X_val_2d)}, test={len(X_test_2d)}, "
        f"H={H}, W={W}, num_bits={num_bits}, num_points={num_points}"
    )

    X_train_flat = X_train_2d.reshape(N_train, num_bits)
    X_val_flat = X_val_2d.reshape(X_val_2d.shape[0], num_bits)
    X_test_flat = X_test_2d.reshape(X_test_2d.shape[0], num_bits)

    if args.canonical_target:
        print("🔁 Applying canonicalization under y-axis flip to targets.")
        X_train_flat = canonicalize_under_yflip(X_train_flat, height=H, width=W)
        X_val_flat = canonicalize_under_yflip(X_val_flat, height=H, width=W)
        X_test_flat = canonicalize_under_yflip(X_test_flat, height=H, width=W)

    order_idx = get_order_indices(args.ordering, num_bits, H, W)
    print(f"🔁 Using ordering = {args.ordering}, order_idx[:20] = {order_idx[:20].tolist()}")

    X_train_ord = X_train_flat[:, order_idx]
    X_val_ord = X_val_flat[:, order_idx]
    X_test_ord = X_test_flat[:, order_idx]

    if args.normalize_input:
        def norm_per_sample(Y):
            mean = Y.mean(axis=1, keepdims=True)
            return Y - mean

        Y_train = norm_per_sample(Y_train)
        Y_val = norm_per_sample(Y_val)
        Y_test = norm_per_sample(Y_test)
        print("🔧 Per-sample normalization applied to Y.")

    Y_train_t = torch.tensor(Y_train)
    X_train_t = torch.tensor(X_train_ord)
    Y_val_t = torch.tensor(Y_val)
    X_val_t = torch.tensor(X_val_ord)
    Y_test_t = torch.tensor(Y_test)
    X_test_t = torch.tensor(X_test_ord)

    train_loader = DataLoader(TensorDataset(Y_train_t, X_train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Y_val_t, X_val_t), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(Y_test_t, X_test_t), batch_size=args.batch_size, shuffle=False)

    run_name = build_run_name(args)
    use_wandb = maybe_init_wandb(args, run_name)

    chain2spatial = torch.from_numpy(order_idx).long()

    model = SmallTransformerAR(
        num_points=num_points,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        max_len=num_bits,
        vocab_size=3,
        dropout=args.dropout,
        spectral_cond=args.spectral_cond,
        use_2d_pos=args.use_2d_pos,
        chain2spatial=chain2spatial,
        height=H,
        width=W,
    ).to(device)

    print(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "cosine":
        warmup_epochs = 5
        lr_warmup = 1e-4
        lr_max = args.lr

        def lr_lambda(epoch):
            ep = epoch + 1
            if ep <= warmup_epochs:
                return lr_warmup / lr_max

            T = max(args.epochs - warmup_epochs, 1)
            t = float(ep - warmup_epochs - 1) / float(T - 1) if T > 1 else 0.0
            cos_factor = 0.5 * (1.0 + np.cos(np.pi * t))
            return cos_factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        print(
            f"🔄 Using warm-up + cosine LR schedule: warmup_epochs={warmup_epochs}, "
            f"lr_warmup={lr_warmup}, lr_max={lr_max}"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    run_save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_save_dir, exist_ok=True)

    best_model_path = os.path.join(run_save_dir, "best_model.pth")
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_bit_acc, train_pat_acc = train_one_epoch_transformer(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_bit_acc, val_pat_acc = eval_transformer(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"Train Loss: {train_loss:.6f}, BitAcc: {train_bit_acc:.4f}, PatAcc: {train_pat_acc:.4f} | "
            f"Val Loss: {val_loss:.6f}, BitAcc: {val_bit_acc:.4f}, PatAcc: {val_pat_acc:.4f} | "
            f"lr={current_lr:.3e}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_bit_acc": train_bit_acc,
                "train_pattern_acc": train_pat_acc,
                "val_loss": val_loss,
                "val_bit_acc": val_bit_acc,
                "val_pattern_acc": val_pat_acc,
            })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Best model updated & saved to {best_model_path}")

            if use_wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss

        if scheduler is not None:
            scheduler.step()

    print("🔍 Evaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_loss, test_bit_acc, test_pat_acc = eval_transformer(model, test_loader, criterion, device)
    print(f"📊 Test Loss: {test_loss:.6f}, Test BitAcc: {test_bit_acc:.4f}, Test PatAcc: {test_pat_acc:.4f}")

    if use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_bit_acc": test_bit_acc,
            "test_pattern_acc": test_pat_acc,
        })
        wandb.run.summary["test_loss"] = test_loss
        wandb.run.summary["test_bit_acc"] = test_bit_acc
        wandb.run.summary["test_pattern_acc"] = test_pat_acc

    print("\n✅ Training finished.")
    print(f"Best model path: {best_model_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 데이터/저장 경로: 현재 .py 파일과 같은 디렉토리 기준
    parser.add_argument("--data_root", type=str, default="/hai/home/lsh/antenna/year_hai/data/datasets")
    parser.add_argument("--save_dir", type=str, default="./inverse_models_changing_ordering")

    # 교수님 최종 설정값
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=15)
    parser.add_argument("--dim_ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)

    # ordering은 snake로만 고정
    parser.add_argument("--ordering", type=str, default="snake", choices=["snake"])
    parser.add_argument("--spectral_cond", type=str, default="resnet1d",
                        choices=["linear", "mlp", "transformer", "resnet1d"])
    parser.add_argument("--use_2d_pos", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--canonical_target", type=lambda x: x.lower() == "true", default=True)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    parser.add_argument("--normalize_input", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--project", type=str, default="260422")
    parser.add_argument("--run_prefix", type=str, default="ar10x10")

    args = parser.parse_args()

    # snake ordering 고정 + canonical 유/무만 실행
    run_settings = [
        #("snake", False),
        ("snake", True),
    ]

    for ordering, canonical_target in run_settings:
        print("\n" + "=" * 80)
        print(f"🚀 Starting training with ordering = {ordering}, canonical_target = {canonical_target}, seed = {args.seed}")
        print("=" * 80)
        args.ordering = ordering
        args.canonical_target = canonical_target
        main(args)
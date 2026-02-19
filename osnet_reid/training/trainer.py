"""
Training pipeline for OSNet ReID model.

Single-stage training with combined CrossEntropy + metric loss (Triplet or Circle).
Uses RandomIdentitySampler for batch construction (P identities x K images).
"""
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from ..models import ReIDModel, CrossEntropyLabelSmooth, TripletLoss, CircleLoss
from ..utils import (
    ReIDDataset, RandomIdentitySampler,
    get_train_transforms, get_val_transforms,
    ReduceLROnPlateau, ModelEMA,
    colorstr, increment_path,
)
from .evaluator import validate


class Trainer:
    """
    ReID training pipeline with combined CE + metric loss (Triplet or Circle).

    Uses RandomIdentitySampler to construct batches with P identities
    and K images per identity, enabling batch hard mining for metric loss.
    """

    def __init__(self, config: Dict[str, Any], args, save_dir: Path, device: torch.device):
        self.config = config
        self.args = args
        self.save_dir = save_dir
        self.device = device
        self.weights_dir = save_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)

        self.model = None
        self.ema = None
        self.train_loader = None
        self.val_loader = None

    def setup_data(self) -> None:
        """Setup datasets and dataloaders with train/val split."""
        config = self.config
        train_cfg = config['train']
        data_cfg = config['data']
        val_split = config['val'].get('val_split', 0.2)

        img_h, img_w = config['model']['input_size']
        aug_cfg = config['data'].get('augmentation', {})
        train_transform = get_train_transforms(img_h, img_w, aug_cfg)
        val_transform = get_val_transforms(img_h, img_w)

        # Full dataset
        full_dataset = ReIDDataset(
            csv_file=data_cfg['csv'],
            dataset_root=data_cfg['root'],
            transform=train_transform,
        )

        # Update num_classes from dataset
        config['model']['num_classes'] = full_dataset.num_pids

        # Split into train/val
        generator = torch.Generator().manual_seed(42)
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

        # For the sampler, we need to create a wrapper that exposes pid_index
        # Map original dataset indices to Subset positional indices
        train_pid_index = {}
        for pos, orig_idx in enumerate(train_ds.indices):
            row = full_dataset.data.iloc[orig_idx]
            pid = full_dataset.pid_to_label[row['person_id']]
            if pid not in train_pid_index:
                train_pid_index[pid] = []
            train_pid_index[pid].append(pos)

        # Create a simple namespace for sampler
        class _SamplerDataset:
            pass
        sampler_ds = _SamplerDataset()
        sampler_ds.pid_index = train_pid_index

        num_instances = data_cfg.get('num_instances', 4)
        sampler = RandomIdentitySampler(sampler_ds, num_instances=num_instances)

        num_workers = train_cfg['num_workers']

        self.train_loader = DataLoader(
            train_ds, batch_size=train_cfg['batch_size'],
            sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=train_cfg['batch_size'] * 2, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        print(f"\nData split: train {train_size} / val {val_size}")
        print(f"Identities: {full_dataset.num_pids}")
        print(f"Batch: P x K = {train_cfg['batch_size'] // num_instances} x {num_instances}")

    def setup_model(self) -> None:
        """Create model."""
        self.model = ReIDModel(self.config).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")

        if getattr(self.args, 'ema', False):
            self.ema = ModelEMA(self.model)

    def _save_checkpoint(self, state: Dict, filename: str) -> None:
        """Save checkpoint to weights directory."""
        torch.save(state, self.weights_dir / filename)

    def _load_checkpoint(self, model, optimizer, scheduler, filename: str):
        """Load checkpoint if it exists. Returns (start_epoch, best_metric)."""
        ckpt_path = self.weights_dir / filename
        if ckpt_path.exists():
            print(colorstr('bright_cyan', f'Resuming from {ckpt_path}'))
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scheduler and ckpt.get('scheduler_state_dict'):
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_metric = ckpt['best_metric']
            print(colorstr('bright_green', f'Resumed: epoch {start_epoch}, best_metric {best_metric:.4f}'))
            return start_epoch, best_metric
        return 0, float('inf')

    def _get_warmup_lr(self, epoch, base_lr, warmup_epochs):
        """Linear warmup learning rate."""
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        return base_lr

    def train(self) -> None:
        """Run the training pipeline."""
        config = self.config
        train_cfg = config['train']
        num_epochs = train_cfg['epochs']
        warmup_epochs = train_cfg.get('warmup_epochs', 5)

        self.model.to(self.device)

        # Layered learning rates
        params_backbone = []
        params_other = []
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                params_backbone.append(param)
            else:
                params_other.append(param)

        optimizer = optim.Adam([
            {'params': params_backbone, 'lr': train_cfg['backbone_lr']},
            {'params': params_other, 'lr': train_cfg['lr']},
        ], weight_decay=train_cfg['weight_decay'])

        criterion_ce = CrossEntropyLabelSmooth(
            config['model']['num_classes'],
            epsilon=train_cfg['label_smooth'],
        )

        loss_type = train_cfg.get('loss_type', 'triplet')
        if loss_type == 'circle':
            criterion_metric = CircleLoss(
                margin=train_cfg.get('circle_margin', 0.25),
                scale=train_cfg.get('circle_scale', 64),
            )
        else:
            criterion_metric = TripletLoss(margin=train_cfg['triplet_margin'])
        metric_weight = train_cfg.get('metric_weight', 1.0)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

        # Try to resume
        start_epoch, best_loss = self._load_checkpoint(
            self.model, optimizer, scheduler, 'checkpoint.pt'
        )
        best_state = copy.deepcopy(self.model.state_dict())

        print(colorstr('bright_green', 'bold', f'\n{"="*60}'))
        print(colorstr('bright_green', 'bold', 'OSNet ReID Training'))
        print(colorstr('bright_green', 'bold', f'{"="*60}'))
        print(f"Epochs: {num_epochs}, Starting from: {start_epoch + 1}")
        if loss_type == 'circle':
            metric_desc = f"Circle (margin={train_cfg.get('circle_margin', 0.25)}, scale={train_cfg.get('circle_scale', 64)})"
        else:
            metric_desc = f"Triplet (margin={train_cfg['triplet_margin']})"
        print(f"Loss: CE (smooth={train_cfg['label_smooth']}) + {metric_weight}x {metric_desc}")
        print(f"LR: backbone={train_cfg['backbone_lr']}, head={train_cfg['lr']}")
        print(f"Warmup: {warmup_epochs} epochs")
        print(colorstr('bright_green', f'Results saved to {self.save_dir}\n'))

        for epoch in range(start_epoch, num_epochs):
            self.model.train()

            # Warmup LR
            if epoch < warmup_epochs:
                warmup_lr_backbone = self._get_warmup_lr(epoch, train_cfg['backbone_lr'], warmup_epochs)
                warmup_lr_head = self._get_warmup_lr(epoch, train_cfg['lr'], warmup_epochs)
                optimizer.param_groups[0]['lr'] = warmup_lr_backbone
                optimizer.param_groups[1]['lr'] = warmup_lr_head

            total_ce = 0
            total_metric = 0
            total_loss = 0
            batch_count = 0

            pbar = tqdm(self.train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]')
            for images, pids, _ in pbar:
                images = images.to(self.device)
                pids = pids.to(self.device)

                optimizer.zero_grad()

                features, logits = self.model(images, task='both')
                loss_ce = criterion_ce(logits, pids)
                loss_metric = criterion_metric(features, pids)
                loss = loss_ce + metric_weight * loss_metric

                loss.backward()
                optimizer.step()

                if self.ema:
                    self.ema.update(self.model)

                total_ce += loss_ce.item()
                total_metric += loss_metric.item()
                total_loss += loss.item()
                batch_count += 1

                pbar.set_postfix({
                    'ce': f'{loss_ce.item():.3f}',
                    loss_type: f'{loss_metric.item():.3f}',
                    'lr': f'{optimizer.param_groups[1]["lr"]:.1e}',
                })

            avg_ce = total_ce / batch_count
            avg_metric = total_metric / batch_count
            avg_loss = total_loss / batch_count

            # Validation
            val_total, val_ce, val_metric = validate(
                self.model, self.val_loader,
                criterion_ce, criterion_metric, metric_weight, self.device,
            )

            # Update scheduler (after warmup)
            if epoch >= warmup_epochs:
                scheduler.step(val_total)

            metric_label = loss_type.capitalize()
            print(f"  Train - CE: {avg_ce:.4f} | {metric_label}: {avg_metric:.4f} | Total: {avg_loss:.4f}")
            print(f"  Val   - CE: {val_ce:.4f} | {metric_label}: {val_metric:.4f} | Total: {val_total:.4f} | "
                  f"LR: {optimizer.param_groups[1]['lr']:.2e}")

            if val_total < best_loss:
                best_loss = val_total
                best_state = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint({
                    'model_state_dict': best_state,
                    'epoch': epoch,
                    'best_metric': best_loss,
                }, 'best.pt')
                print(colorstr('bright_green', f'  Best model saved! (loss: {best_loss:.4f})'))

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_metric': best_loss,
                }, 'checkpoint.pt')

        # Load best model
        self.model.load_state_dict(best_state)

        # Save final model with config
        final_path = self.weights_dir / 'final.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, final_path)

        print(colorstr('bright_green', 'bold', f'\n{"="*60}'))
        print(colorstr('bright_green', 'bold', 'Training Complete!'))
        print(colorstr('bright_green', 'bold', f'{"="*60}'))
        print(f"Best model: {self.weights_dir / 'best.pt'} (loss: {best_loss:.4f})")
        print(f"Final model: {final_path}")

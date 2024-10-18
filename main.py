import os
import random
import logging
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
import torch.autograd as autograd

from utils.datasets import load_dataloader

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random.seed(config['RANDOM_SEED'])
np.random.seed(config['RANDOM_SEED'])
torch.manual_seed(config['RANDOM_SEED'])
torch.cuda.manual_seed(config['RANDOM_SEED'])
torch.cuda.manual_seed_all(config['RANDOM_SEED'])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

logger.info(f"Random seeds set to {config['RANDOM_SEED']} for reproducibility")

def show_images(images, label_lists=None, is_batch=False, exp_label=None):
    if is_batch:
        images = images.cpu().unbind(0)
        if label_lists is not None:
            if not isinstance(label_lists, dict):
                label_lists = {"label": label_lists}
            label_lists = {k: v.cpu().tolist() for k, v in label_lists.items()}

    n = len(images)
    n_rows = int(np.sqrt(n))
    n_cols = int(np.ceil(n / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * config['PLT_COL_SIZE'], n_rows * config['PLT_ROW_SIZE']))
    axes = axes.flatten()

    for i, (img, ax) in enumerate(zip(images, axes)):
        title = f'n{i}'
        if label_lists:
            title += '\n' + '\n'.join(f"{k}=\"{v[i]}\"" for k, v in label_lists.items())
        ax.set_title(title)
        ax.axis('off')
        
        cmap = "gray" if img.squeeze().ndim == 2 else None
        ax.imshow(img.permute(1, 2, 0), cmap=cmap)

    for ax in axes[len(images):]:
        ax.remove()

    plt.tight_layout()
    
    if exp_label:
        fig.suptitle(f"Experiment: {exp_label}", fontsize=16)
        plt.subplots_adjust(top=0.9)  # Adjust the top margin to accommodate the suptitle
    
    plt.show()
    logger.info(f"Displayed {'batch of ' if is_batch else ''}{n} images with dimensions {images[0].shape}")

def show_dataloader_first_batch(dataloader, exp_label=None):
    images_batch, labels_batch = next(iter(dataloader))
    if isinstance(labels_batch, list):
        labels_batch = {name: lb.cpu() for name, lb in zip(config["LATENT_NAMES"], labels_batch)}
    show_images(images_batch, labels_batch, is_batch=True, exp_label=exp_label)

class MultiheadResNet18(nn.Module):
    def __init__(self, n_classes, n_heads=1):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_ftrs, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n_classes)
            ) for _ in range(n_heads)
        ])
        self.penultimate_features = {}
        logger.info(f"Built MultiheadResNet18 with {n_classes} classes and {n_heads} heads")

    def _get_penultimate_features(self, head_idx):
        def hook(_, input, __):
            self.penultimate_features[head_idx] = input[0].detach()
        return hook

    def forward(self, x):
        features = self.backbone(x)
        for idx, head in enumerate(self.heads):
            head[-1].register_forward_hook(self._get_penultimate_features(idx))
        return [head(features) for head in self.heads]

    def get_features(self):
        return self.penultimate_features

class ModelTrainer:
    def __init__(self, n_classes, task_cue, train_dataloader=None, val_dataloaders=None):
        self.latents_names = config['LATENT_NAMES']
        self.n_classes = n_classes
        self.task_cue = task_cue
        self.task_label_index = self.latents_names.index(self.task_cue)
        
        self.model = MultiheadResNet18(n_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_dataloader = train_dataloader
        self.val_dataloaders = val_dataloaders
        logger.info(f"Using device: {self.device}")

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config['DEFAULT_START_LR'],
            momentum=0.9
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9)
        logger.info(f"ModelTrainer initialized with {n_classes} classes, training on '{task_cue}' cue")
        
    def train_loop(self, images_batch, labels_batch, epoch=None):
        images_batch = images_batch.to(self.device)
        labels_batch = labels_batch[self.task_label_index].to(self.device)
        predictions = self.model(images_batch)
        losses = [self.criterion(pred, labels_batch) for pred in predictions]
        avg_loss = torch.mean(torch.stack(losses))
        
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()
        return avg_loss.item()
    
    def train(self, epoch):        
        self.model.train()
        dataloader = self.train_dataloader

        total_loss = 0
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
        for images_batch, labels_batch in progress_bar:
            
            loss = self.train_loop(images_batch=images_batch, labels_batch=labels_batch, epoch=epoch)
            
            total_loss += loss
            total_samples += labels_batch[0].size(0)
            progress_bar.set_postfix({"Loss": f"{loss:.4f}"})
        self.scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        
        log_message = f"Epoch {epoch} training completed. Average Loss: {avg_loss:.4f}"
        logger.info(log_message)
        
    def eval(self, eval_key, epoch, head_idx=0):        
        dataloader = self.val_dataloaders[eval_key]
        self.model.eval()

        total_losses = {cue: 0 for cue in self.latents_names}
        correct_predictions = {cue: 0 for cue in self.latents_names}
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} {eval_key.capitalize()} Evaluation", leave=False)
        for images_batch, labels_batch in progress_bar:
            images_batch = images_batch.to(self.device)
            labels_batch = [label.to(self.device) for label in labels_batch]
            
            with torch.no_grad():
                pred_batches = self.model(images_batch)
                pred_batch = pred_batches[head_idx]
            
            for idx, cue in enumerate(self.latents_names):
                loss = self.criterion(pred_batch, labels_batch[idx])
                total_losses[cue] += loss.item()
                
                _, predicted = torch.max(pred_batch, 1)
                correct_predictions[cue] += (predicted == labels_batch[idx]).sum().item()
            
            total_samples += labels_batch[0].size(0)
            
            progress_bar.set_postfix({"Batch": f"{progress_bar.n}/{len(dataloader)}"})
        
        avg_losses = {cue: total_loss / len(dataloader) for cue, total_loss in total_losses.items()}
        accuracies = {cue: correct / total_samples * 100 for cue, correct in correct_predictions.items()}
        
        log_message = f"Epoch {epoch}, {eval_key} evaluation completed for {'all heads' if head_idx is None else f'head {head_idx}'}."
        for cue in self.latents_names:
            log_message += f"\n{cue.capitalize()} - Loss: {avg_losses[cue]:.4f}, Accuracy: {accuracies[cue]:.2f}%"
        
        logger.info(log_message)
class ModelTrainerIRM(ModelTrainer):
    def __init__(self, n_classes, task_cue, train_dataloader=None, val_dataloaders=None,
                 l2_regularizer_weight=1e-5, penalty_weight=10000.0, penalty_anneal_epochs=2):
        super().__init__(n_classes, task_cue, train_dataloader, val_dataloaders)
        self._l2_regularizer_weight = l2_regularizer_weight
        self._penalty_weight = penalty_weight
        self._penalty_anneal_epochs = penalty_anneal_epochs

    def _mean_nll(self, logits, y):
        return nn.functional.cross_entropy(logits, y)

    def _penalty(self, logits, y):
        scale = torch.tensor(1.).to(self.device).requires_grad_()
        loss = self._mean_nll(logits * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)
    
    def train(self, epoch):        
        self.model.train()

        total_loss = 0
        correct_predictions = {cue: 0 for cue in self.latents_names}
        total_samples = 0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch} Training", leave=False)
        for images_batch, labels_batch in progress_bar:
            images_batch = images_batch.to(self.device)
            labels_batch = labels_batch[self.task_label_index].to(self.device)
            
            logits = self.model(images_batch)[0]  # Use only the first head
            _, predicted = torch.max(logits, 1)

            env_nll = self._mean_nll(logits, labels_batch)
            env_penalty = self._penalty(logits, labels_batch)
            
            train_nll = env_nll
            train_penalty = env_penalty

            correct_predictions[self.latents_names[self.task_label_index]] += (predicted == labels_batch).sum().item()
            total_samples += labels_batch.size(0)

            weight_norm = torch.tensor(0.).to(self.device)
            for w in self.model.parameters():
                weight_norm += w.norm().pow(2)

            loss = train_nll.clone()
            loss += self._l2_regularizer_weight * weight_norm
            penalty_weight = (self._penalty_weight if epoch >= self._penalty_anneal_epochs else 1.0)
            loss += penalty_weight * train_penalty
            if penalty_weight > 1.0:
                loss /= penalty_weight
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        self.scheduler.step()
        
        avg_loss = total_loss / len(self.train_dataloader)
        accuracies = {cue: correct / total_samples * 100 for cue, correct in correct_predictions.items()}
        
        log_message = f"Epoch {epoch} training completed. Average Loss: {avg_loss:.4f}"
        for cue in self.latents_names:
            log_message += f", {cue.capitalize()} Accuracy: {accuracies[cue]:.2f}%"
        
        logger.info(log_message)
        
class ModelTrainerPoE(ModelTrainer):
    def __init__(self, n_classes, n_heads, task_cue, train_dataloader=None, val_dataloaders=None):
        super().__init__(n_classes, task_cue, train_dataloader, val_dataloaders)
        self.n_heads = n_heads
        self.model = MultiheadResNet18(n_classes, n_heads=n_heads)

    def _poe_loss(self, logits, labels):
        summed_logits = sum(logits)
        loss = F.cross_entropy(summed_logits, labels)
        return loss

    def train_loop(self, images_batch, labels_batch, epoch=None):
        images_batch = images_batch.to(self.device)
        labels_batch = labels_batch[self.task_label_index].to(self.device)
        
        predictions = self.model(images_batch)
        
        loss = self._poe_loss(predictions, labels_batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class ModelTrainerHSIC(ModelTrainer):
    def __init__(self, n_classes, n_heads, alpha, task_cue, train_dataloader=None, val_dataloaders=None):
        super().__init__(n_classes, task_cue, train_dataloader, val_dataloaders)
        self.n_heads = n_heads
        self.model = MultiheadResNet18(n_classes, n_heads=n_heads)
        self._alpha = alpha

    def _hsic_loss(self, embeddings):
        if len(embeddings) != 2:
            raise ValueError("Expected 2 sets of embeddings")
        
        def centering(K):
            n = K.shape[0]
            unit = torch.ones([n, n], device=K.device)
            I = torch.eye(n, device=K.device)
            H = I - unit / n
            return torch.matmul(torch.matmul(H, K), H)

        def rbf(X, sigma=None):
            GX = torch.matmul(X, X.T)
            KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
            if sigma is None:
                mdist = torch.median(KX[KX != 0])
                sigma = math.sqrt(mdist.item())
            KX *= -0.5 / (sigma * sigma)
            KX = torch.exp(KX)
            return KX

        X = embeddings[0]
        Y = embeddings[1]
        
        K = centering(rbf(X))
        L = centering(rbf(Y))
        
        n = K.shape[0]
        hsic = torch.trace(torch.matmul(K, L)) / (n * n)
        
        return hsic

    def train_loop(self, images_batch, labels_batch, epoch=None):
        images_batch = images_batch.to(self.device)
        labels_batch = labels_batch[self.task_label_index].to(self.device)
        
        predictions = self.model(images_batch)
        embeddings = self.model.get_features()
        ce_losses = [self.criterion(pred, labels_batch) for pred in predictions]
        
        loss = torch.mean(torch.stack(ce_losses))+ self._alpha * self._hsic_loss(embeddings)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

        
def download_datasets():
    logger.info("Starting main function")
    if not os.path.exists(config['DSPRITES_LOCAL_PATH']):
        logger.info(f"Downloading dSprites dataset to {config['DSPRITES_LOCAL_PATH']}")
        os.system(f"wget {config['DSPRITES_URL']} -O {config['DSPRITES_LOCAL_PATH']}")

def visualize_unbiased_data():
    unbiased_test_dataloader = load_dataloader(
        data_setting="unbiased",
        split="test", 
        dataset_size=config['TEST_DATASET_SIZE'],
    )
    logger.info("Showing first batch of test multilabel dataloader")
    show_dataloader_first_batch(dataloader=unbiased_test_dataloader, exp_label="Unbiased Data")
    
def visualize_diagonal_data():
    biased_test_dataloader = load_dataloader(
        data_setting="diagonal",
        split="test", 
        dataset_size=config['TEST_DATASET_SIZE'],
        bias_cue=config["BIAS_CUE"],
        task_cue=config["TASK_CUE"],
        off_diag_proportion=0,
    )
    logger.info("Showing first batch of biased test multilabel dataloader")
    show_dataloader_first_batch(dataloader=biased_test_dataloader, exp_label="Diagonal Data")

def visualize_domain_generalization_data():
    dg_test_dataloader = load_dataloader(
        data_setting="domain_generalization",
        split="test", 
        dataset_size=config['TEST_DATASET_SIZE'],
        bias_cue=config["BIAS_CUE"],
        bias_cue_classes=config["BIAS_CUE_CLASSES_DG_TEST"],
    )
    logger.info("Showing first batch of domain generalization test dataloader")
    dg_train_dataloader = load_dataloader(
        data_setting="domain_generalization",
        split="train", 
        dataset_size=config['TRAIN_DATASET_SIZE'],
        bias_cue=config["BIAS_CUE"],
        bias_cue_classes=config["BIAS_CUE_CLASSES_DG_TRAIN"],
    )
    logger.info("Showing first batch of domain generalization train dataloader")
    show_dataloader_first_batch(dataloader=dg_train_dataloader, exp_label="Domain Generalization Train Data")
    show_dataloader_first_batch(dataloader=dg_test_dataloader, exp_label="Domain Generalization Test Data")

def train_erm():
    logger.info("Initializing ground truth trainer")
    trainer = ModelTrainer(
        task_cue=config["TASK_CUE"],
        n_classes=config['NUM_CLASSES'],
        train_dataloader=load_dataloader(
            data_setting="unbiased",
            split="train", 
            dataset_size=config['TRAIN_DATASET_SIZE']),
        val_dataloaders={
            "unbiased": load_dataloader(
                data_setting="unbiased",
                split="test",
                dataset_size=config['TEST_DATASET_SIZE'])},
    )
    logger.info("Starting ground truth model training and evaluation")
    trainer.eval(eval_key="unbiased", epoch=0)
    for epoch in range(1, config['NUM_EPOCHS'] + 1):
        logger.info(f"Starting epoch {epoch}/{config['NUM_EPOCHS']}")
        trainer.train(epoch=epoch)
        trainer.eval(eval_key="unbiased", epoch=epoch)

def train_poe_diverse_ensemble():
    logger.info("Initializing Diversify trainer")
    trainer = ModelTrainerPoE(
        task_cue=config["TASK_CUE"],
        n_classes=config['NUM_CLASSES'],
        n_heads=config['NUM_HEADS'],
        train_dataloader=load_dataloader(
            data_setting="diagonal",
            split="train", 
            dataset_size=config['TRAIN_DATASET_SIZE'], 
            bias_cue=config["BIAS_CUE"], 
            task_cue=config["TASK_CUE"],
            off_diag_proportion=0),
        val_dataloaders={
            "unbiased": load_dataloader(
                data_setting="unbiased",
                split="test", 
                dataset_size=config['TEST_DATASET_SIZE'])
            },
        )
    logger.info("Starting Diversify model training and evaluation")
    for head_idx in range(config['NUM_HEADS']):
        trainer.eval(eval_key="unbiased", epoch=0, head_idx=head_idx)
    for epoch in range(1, config['NUM_EPOCHS'] + 1):
        logger.info(f"Starting epoch {epoch}/{config['NUM_EPOCHS']}")
        trainer.train(epoch=epoch)
        for head_idx in range(config['NUM_HEADS']):
            trainer.eval(eval_key="unbiased", epoch=epoch, head_idx=head_idx)

def train_hsic_diverse_ensemble():
    logger.info("Initializing Diversify trainer")
    trainer = ModelTrainerHSIC(
        task_cue=config["TASK_CUE"],
        n_classes=config['NUM_CLASSES'],
        n_heads=config['NUM_HEADS'],
        alpha=0.1,
        train_dataloader=load_dataloader(
            data_setting="diagonal",
            split="train", 
            dataset_size=config['TRAIN_DATASET_SIZE'], 
            bias_cue=config["BIAS_CUE"], 
            task_cue=config["TASK_CUE"],
            off_diag_proportion=0),
        val_dataloaders={
            "unbiased": load_dataloader(
                data_setting="unbiased",
                split="test", 
                dataset_size=config['TEST_DATASET_SIZE'])
            },
    )
    logger.info("Starting Diversify model training and evaluation")
    for head_idx in range(config['NUM_HEADS']):
        trainer.eval(eval_key="unbiased", epoch=0, head_idx=head_idx)
    for epoch in range(1, config['NUM_EPOCHS'] + 1):
        logger.info(f"Starting epoch {epoch}/{config['NUM_EPOCHS']}")
        trainer.train(epoch=epoch)
        for head_idx in range(config['NUM_HEADS']):
            trainer.eval(eval_key="unbiased", epoch=epoch, head_idx=head_idx)


def train_irm():
    logger.info("Initializing IRM trainer")
    trainer = ModelTrainerIRM(
        task_cue=config["TASK_CUE"],
        n_classes=config['NUM_CLASSES'],
        train_dataloader=load_dataloader(
            data_setting="domain_generalization",
            split="train",
            dataset_size=config['TRAIN_DATASET_SIZE'],
            bias_cue=config["BIAS_CUE"],
            bias_cue_classes=config["BIAS_CUE_CLASSES_DG_TRAIN"]),
        val_dataloaders={
            "dg_test": load_dataloader(
                data_setting="domain_generalization",
                split="test",
                dataset_size=config['TEST_DATASET_SIZE'],
                bias_cue=config["BIAS_CUE"],
                bias_cue_classes=config["BIAS_CUE_CLASSES_DG_TEST"]),
            },
    )
    logger.info("Starting IRM model training and evaluation")
    trainer.eval(eval_key="dg_test", epoch=0)
    for epoch in range(1, config['NUM_EPOCHS'] + 1):
        logger.info(f"Starting epoch {epoch}/{config['NUM_EPOCHS']}")
        trainer.train(epoch=epoch)
        trainer.eval(eval_key="dg_test", epoch=epoch)


if __name__ == "__main__":
    download_datasets()
    visualize_unbiased_data()
    visualize_diagonal_data()
    visualize_domain_generalization_data()
    train_erm()
    train_poe_diverse_ensemble()
    train_hsic_diverse_ensemble()
    train_irm()

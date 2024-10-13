import os
import random
import logging
import json

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

def show_images(images, label_lists=None, is_batch=False):
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
    plt.show()
    logger.info(f"Displayed {'batch of ' if is_batch else ''}{n} images with dimensions {images[0].shape}")

def show_dataloader_first_batch(dataloader):
    images_batch, labels_batch = next(iter(dataloader))
    if isinstance(labels_batch, list):
        labels_batch = {name: lb.cpu() for name, lb in zip(config["LATENT_NAMES"], labels_batch)}
    show_images(images_batch, labels_batch, is_batch=True)

def build_resnet18(n_classes):
    model = models.resnet18(pretrained=False)
    default_model = models.resnet18(pretrained=False)
    if n_classes != default_model.fc.out_features:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_features=num_ftrs, out_features=n_classes)
    logger.info(f"Built ResNet18 model with {n_classes} output classes")
    return model

class ModelTrainer:
    def __init__(self, n_classes, task_cue, train_dataloaders=None, val_dataloaders=None):
        self.latents_names = config['LATENT_NAMES']
        self.n_classes = n_classes
        self.task_cue = task_cue
        self.task_label_index = self.latents_names.index(self.task_cue)
        
        self.model = build_resnet18(n_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_dataloaders = train_dataloaders
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

        pred_batch = self.model(images_batch)
        loss = self.criterion(pred_batch, labels_batch)
        
        _, predicted = torch.max(pred_batch, 1)
                
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, predicted
        
    def train(self, epoch):        
        self.model.train()
        dataloader = next(iter(self.train_dataloaders.values()))  # By default, use only the first dataloader

        total_loss = 0
        correct_predictions = {cue: 0 for cue in self.latents_names}
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
        for images_batch, labels_batch in progress_bar:
            
            loss, predicted = self.train_loop(images_batch=images_batch, labels_batch=labels_batch, epoch=epoch)
            
            total_loss += loss.item()
            total_samples += labels_batch.size(0)
            for cue, label in zip(self.latents_names, labels_batch):
                correct_predictions[cue] += (predicted == label).sum().item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        self.scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        accuracies = {cue: correct / total_samples * 100 for cue, correct in correct_predictions.items()}
        
        log_message = f"Epoch {epoch} training completed. Average Loss: {avg_loss:.4f}"
        for cue in self.latents_names:
            log_message += f", {cue.capitalize()} Accuracy: {accuracies[cue]:.2f}%"
        
        logger.info(log_message)
        
    def eval(self, eval_key, epoch):        
        dataloader = self.val_dataloaders[eval_key]
        self.model.eval()

        total_losses = {cue: 0 for cue in self.latents_names}
        correct_predictions = {cue: 0 for cue in self.latents_names}
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} {eval_key.capitalize()} Evaluation", leave=False)
        for images_batch, labels_batch in progress_bar:
            images_batch = images_batch.to(self.device)
            labels_batch = [label.to(self.device) for label in labels_batch]
            
            pred_batch = self.model(images_batch)
            
            for idx, cue in enumerate(self.latents_names):
                loss = self.criterion(pred_batch, labels_batch[idx])
                total_losses[cue] += loss.item()
                
                _, predicted = torch.max(pred_batch, 1)
                correct_predictions[cue] += (predicted == labels_batch[idx]).sum().item()
            
            total_samples += labels_batch[0].size(0)
            
            progress_bar.set_postfix({"Batch": f"{progress_bar.n}/{len(dataloader)}"})
        
        avg_losses = {cue: total_loss / len(dataloader) for cue, total_loss in total_losses.items()}
        accuracies = {cue: correct / total_samples * 100 for cue, correct in correct_predictions.items()}
        
        log_message = f"Epoch {epoch}, {eval_key} evaluation completed."
        for cue in self.latents_names:
            log_message += f"\n{cue.capitalize()} - Loss: {avg_losses[cue]:.4f}, Accuracy: {accuracies[cue]:.2f}%"
        
        logger.info(log_message)

class ModelTrainerIRM(ModelTrainer):
    def __init__(self, n_classes, train_cue, train_dataloaders=None, val_dataloaders=None,
                 l2_regularizer_weight=1e-5, penalty_weight=10000.0, penalty_anneal_epochs=2):
        super().__init__(n_classes, train_cue, train_dataloaders, val_dataloaders)
        self._l2_regularizer_weight = l2_regularizer_weight
        self._penalty_weight = penalty_weight
        self._penalty_anneal_epochs = penalty_anneal_epochs

    def _mean_nll(self, logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

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

        zipped_loaders = zip(*self.train_dataloaders.values())
        min_len = min(len(loader) for loader in self.train_dataloaders.values())

        progress_bar = tqdm(range(min_len), desc=f"Epoch {epoch} Training", leave=False)
        for _ in progress_bar:
            batches = next(zipped_loaders)
            
            nlls = []
            penalties = []
            
            for batch in batches:
                images_batch = batch[0].to(self.device)
                labels_batch = batch[1][self.task_label_index].to(self.device)
                
                logits = self.model(images_batch)
                _, predicted = torch.max(logits, 1)

                env_nll = self._mean_nll(logits, labels_batch)
                env_penalty = self._penalty(logits, labels_batch)
                
                nlls.append(env_nll)
                penalties.append(env_penalty)

                correct_predictions[self.latents_names[self.task_label_index]] += (predicted == labels_batch).sum().item()
                total_samples += labels_batch.size(0)

            train_nll = torch.stack(nlls).mean()
            train_penalty = torch.stack(penalties).mean()

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
        
        avg_loss = total_loss / min_len
        accuracies = {cue: correct / total_samples * 100 for cue, correct in correct_predictions.items()}
        
        log_message = f"Epoch {epoch} training completed. Average Loss: {avg_loss:.4f}"
        for cue in self.latents_names:
            log_message += f", {cue.capitalize()} Accuracy: {accuracies[cue]:.2f}%"
        
        logger.info(log_message)
        
class ModelTrainerDiversify(ModelTrainer):
    def __init__(self, n_classes, n_models, alpha, task_cue, train_dataloaders=None, val_dataloaders=None):
        super().__init__(n_classes, task_cue, train_dataloaders, val_dataloaders)
        self.n_models = n_models
        original_model = self.model
        self.model = nn.ModuleList([original_model] + [build_resnet18(n_classes) for _ in range(max(0, n_models - 1))])
        self._alpha = alpha

    def _a2d_loss(self, logits, batch_size):
        """logits: num_models x batch_size x num_classes"""
        div_loss = 0.0
        num_models = len(logits)
        
        for m in range(num_models):
            for l in range(m):
                pm = F.softmax(logits[m], dim=1)
                pl = F.softmax(logits[l], dim=1)
                y_hat_m = pm.argmax(dim=1)
                a2d_term = -torch.log(pm[range(batch_size), y_hat_m] * (1 - pl[range(batch_size), y_hat_m]) +
                                    pl[range(batch_size), y_hat_m] * (1 - pm[range(batch_size), y_hat_m]))
                div_loss += a2d_term.mean()
        
        return div_loss / (num_models * (num_models - 1) / 2)
            
    def train_loop(self, images_batch, labels_batch, epoch=None):
        images_batch = images_batch.to(self.device)
        labels_batch = labels_batch[self.task_label_index].to(self.device)
        
        losses = []
        predictions = []

        for model in self.model:
            model.train()

            pred_batch = model(images_batch)
            loss = self.criterion(pred_batch, labels_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss)
            predictions.append(F.softmax(pred_batch, dim=1))
        
        loss = torch.stack(losses).mean() + self._alpha * self._a2d_loss(torch.stack(losses))
        mean_prediction = torch.stack(predictions).mean(dim=0)
        
        return loss, mean_prediction
        
    def eval(self, eval_key, epoch):        
        dataloader = self.val_dataloaders[eval_key]
        for model in self.model:
            model.eval()

        total_losses = {cue: 0 for cue in self.latents_names}
        correct_predictions = {cue: 0 for cue in self.latents_names}
        total_samples = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} {eval_key.capitalize()} Evaluation", leave=False)
        for images_batch, labels_batch in progress_bar:
            images_batch = images_batch.to(self.device)
            labels_batch = [label.to(self.device) for label in labels_batch]
            
            pred_batches = [model(images_batch) for model in self.model]
            mean_pred_batch = torch.stack(pred_batches).mean(dim=0)
            
            for idx, cue in enumerate(self.latents_names):
                loss = self.criterion(mean_pred_batch, labels_batch[idx])
                total_losses[cue] += loss.item()
                
                _, predicted = torch.max(mean_pred_batch, 1)
                correct_predictions[cue] += (predicted == labels_batch[idx]).sum().item()
            
            total_samples += labels_batch[0].size(0)
            
            progress_bar.set_postfix({"Batch": f"{progress_bar.n}/{len(dataloader)}"})
        
        avg_losses = {cue: total_loss / len(dataloader) for cue, total_loss in total_losses.items()}
        accuracies = {cue: correct / total_samples * 100 for cue, correct in correct_predictions.items()}
        
        log_message = f"Epoch {epoch}, {eval_key} evaluation completed."
        for cue in self.latents_names:
            log_message += f"\n{cue.capitalize()} - Loss: {avg_losses[cue]:.4f}, Accuracy: {accuracies[cue]:.2f}%"
        
        logger.info(log_message)


def download_datasets():
    logger.info("Starting main function")
    if not os.path.exists(config['DSPRITES_LOCAL_PATH']):
        logger.info(f"Downloading dSprites dataset to {config['DSPRITES_LOCAL_PATH']}")
        os.system(f"wget {config['DSPRITES_URL']} -O {config['DSPRITES_LOCAL_PATH']}")

def visualize_unbiased_data():
    unbiased_test_dataloader = load_dataloader(split="test", dataset_size=config['TEST_DATASET_SIZE'])
    logger.info("Showing first batch of test multilabel dataloader")
    show_dataloader_first_batch(dataloader=unbiased_test_dataloader)
    
def visualize_biased_data():
    biased_test_dataloader = load_dataloader(split="test", dataset_size=config['TEST_DATASET_SIZE'], bias_cue=config["BIAS_CUE"], task_cue=config["TASK_CUE"])
    logger.info("Showing first batch of biased test multilabel dataloader")
    show_dataloader_first_batch(dataloader=biased_test_dataloader)

def train_erm():
    n_epochs = config['NUM_EPOCHS']
    logger.info("Initializing ground truth trainer")
    trainer = ModelTrainer(
        task_cue=config["TASK_CUE"],
        n_classes=config['NUM_CLASSES'],
        train_dataloaders={"unbiased": load_dataloader(split="train", dataset_size=config['TRAIN_DATASET_SIZE'])},
        val_dataloaders={"unbiased": load_dataloader(split="test", dataset_size=config['TEST_DATASET_SIZE'])},
    )
    logger.info("Starting ground truth model training and evaluation")
    trainer.eval(eval_key="unbiased", epoch=0)
    for epoch in range(1, n_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{n_epochs}")
        trainer.train(epoch=epoch)
        trainer.eval(eval_key="unbiased", epoch=epoch)
        

def train_cbm():
    n_epochs = config['NUM_EPOCHS']
    logger.info("Initializing CBM trainers for each cue")
    trainers = {}
    for cue in config['LATENT_NAMES']:
        trainers[cue] = ModelTrainer(
            task_cue=cue,
            n_classes=config['NUM_CLASSES'],
            train_dataloaders={"unbiased": load_dataloader(split="train", dataset_size=config['TRAIN_DATASET_SIZE'])},
            val_dataloaders={"unbiased": load_dataloader(split="test", dataset_size=config['TEST_DATASET_SIZE'])},
        )
    
    logger.info("Starting CBM model training and evaluation for each cue")
    for cue, trainer in trainers.items():
        logger.info(f"Evaluating initial model for cue: {cue}")
        trainer.eval(eval_key="unbiased", epoch=0)
    
    for epoch in range(1, n_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{n_epochs}")
        for cue, trainer in trainers.items():
            logger.info(f"Training and evaluating model for cue: {cue}")
            trainer.train(epoch=epoch)
            trainer.eval(eval_key="unbiased", epoch=epoch)
        
        
def train_irm():
    n_epochs = config['NUM_EPOCHS']
    logger.info("Initializing IRM trainer")
    trainer = ModelTrainerIRM(
        task_cue=config["TASK_CUE"],
        n_classes=config['NUM_CLASSES'],
        train_dataloaders={
            "biased": load_dataloader(split="train", dataset_size=config['TRAIN_DATASET_SIZE'], bias_cue=config["BIAS_CUE"], task_cue=config["TASK_CUE"]),
            "less_biased": load_dataloader(split="train", dataset_size=config['TRAIN_DATASET_SIZE'], bias_cue=config["BIAS_CUE"], task_cue=config["TASK_CUE"], off_diag_proportion=.01),
        },
        val_dataloaders={"unbiased": load_dataloader(split="test", dataset_size=config['TEST_DATASET_SIZE'])},
    )
    logger.info("Starting IRM model training and evaluation")
    trainer.eval(eval_key="unbiased", epoch=0)
    for epoch in range(1, n_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{n_epochs}")
        trainer.train(epoch=epoch)
        trainer.eval(eval_key="unbiased", epoch=epoch)
    # TODO use biased dataset loaders.


def train_diversify():
    n_epochs = config['NUM_EPOCHS']
    logger.info("Initializing Diversify trainer")
    trainer = ModelTrainerDiversify(
        task_cue=config["TASK_CUE"],
        n_classes=config['NUM_CLASSES'],
        n_models=2,
        alpha=0.1,
        train_dataloaders={"biased": load_dataloader(split="train", dataset_size=config['TRAIN_DATASET_SIZE'], bias_cue=config["BIAS_CUE"], task_cue=config["TASK_CUE"])},
        val_dataloaders={"unbiased": load_dataloader(split="test", dataset_size=config['TEST_DATASET_SIZE'])},
    )
    logger.info("Starting Diversify model training and evaluation")
    trainer.eval(eval_key="unbiased", epoch=0)
    for epoch in range(1, n_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{n_epochs}")
        trainer.train(epoch=epoch)
        trainer.eval(eval_key="unbiased", epoch=epoch)

def train_debiased_model():
    n_epochs = config['NUM_EPOCHS']
    logger.info("Initializing de-biasing trainer")
    trainer = ModelTrainer(
        task_cue=config["TASK_CUE"],
        n_classes=config['NUM_CLASSES'],
        train_dataloader={
            "biased": load_dataloader(split="train", dataset_size=config['TRAIN_DATASET_SIZE'], bias_cue=config["BIAS_CUE"], task_cue=config["TASK_CUE"])
        },
        val_dataloaders={
            "unbiased": load_dataloader(split="test", dataset_size=config['TEST_DATASET_SIZE']),
            "biased": load_dataloader(split="test", dataset_size=config['TEST_DATASET_SIZE'], bias_cue=config["BIAS_CUE"], task_cue=config["TASK_CUE"])
        },
    )
    logger.info("Starting de-biasing model training and evaluation")
    trainer.eval(eval_key="unbiased", epoch=0)
    trainer.eval(eval_key="biased", epoch=0)
    for epoch in range(1, n_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{n_epochs}")
        trainer.train(epoch=epoch)
        trainer.eval(eval_key="unbiased", epoch=epoch)
        trainer.eval(eval_key="biased", epoch=epoch)

if __name__ == "__main__":
    download_datasets()
    visualize_unbiased_data()
    visualize_biased_data()
    train_erm()
    train_cbm()
    train_irm()
    train_diversify()

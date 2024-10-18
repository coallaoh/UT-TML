import json
import logging
import random
from collections import defaultdict

from picklecachefunc import check_cache
import numpy as np
import torch
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)
    logging.info("Configuration loaded successfully")

def read_dsprites_npz(filename):
    logging.info(f"Reading dSprites data from {filename}")
    data = np.load(filename, allow_pickle=True, encoding='latin1')
    logging.info("dSprites data loaded successfully")
    return data


class DSpritesDataset:
    def __init__(self, split, train_val_split_ratio: float = config["TRAIN_VAL_SPLIT_RATIO"]):
        logging.info("Initializing DSpritesDataset")
        self.split = split
        
        dsprites_zip = read_dsprites_npz(config["DSPRITES_LOCAL_PATH"])
        self.imgs = dsprites_zip['imgs']
        
        metadata = dsprites_zip['metadata'][()]
        self.latent = {
            'names': metadata['latents_names'],
            'names_to_indices': {},
            'sizes': metadata['latents_sizes'],
            'bases': None
        }
        
        if tuple(self.latent['names']) != tuple(config["LATENT_NAMES"]):
            logging.warning("Mismatch between latent names in data and config:")
            logging.warning(f"Data latent names: {self.latent['names']}")
            logging.warning(f"Config latent names: {config['LATENT_NAMES']}")
            logging.warning("Using latent names from the data.")
        else:
            logging.info("Latent names in data match the config.")
            
        for i, name in enumerate(self.latent['names']):
            self.latent['names_to_indices'][name] = i
        self.latent['sizes'][0] = 3
        self.latent['bases'] = np.concatenate((self.latent['sizes'][::-1].cumprod()[::-1][1:], np.array([1,])))

        self._num_images_no_color = self.imgs.shape[0]
        self._total_num_images = self._num_images_no_color * self.latent['sizes'][0]

        self.colored_dsprites_indices = self._split_into_train_and_test(train_val_split_ratio=train_val_split_ratio)
        logging.info(f"DSpritesDataset initialized for {split} split with {len(self.colored_dsprites_indices)} samples")

    def _split_into_train_and_test(self, train_val_split_ratio):
        logging.info("Splitting data into train and test sets")
        all_indices = np.arange(self._total_num_images, dtype=np.int64)
        train_indices_positions = np.linspace(
            0,
            self._total_num_images - 1,
            num=max(1, int(train_val_split_ratio * self._total_num_images)),
            dtype=np.int64
        )
        train_indices = all_indices[train_indices_positions].tolist()
        test_indices = list(set(all_indices) - set(train_indices))
        logging.info(f"Split complete. Train set size: {len(train_indices)}, Test set size: {len(test_indices)}")
        if self.split == "train":
            return train_indices
        elif self.split == "test":
            return test_indices
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __getitem__(self, idx_colored_dsprites_indices):
        colored_dsprites_idx = self.colored_dsprites_indices[idx_colored_dsprites_indices]
        color = int(colored_dsprites_idx / self._num_images_no_color)
        original_idx = colored_dsprites_idx % self._num_images_no_color
        colored_image = np.zeros((3, *self.imgs[0].shape))
        colored_image[color, ...] = self.imgs[original_idx]
        return torch.Tensor(colored_image)

    def __len__(self):
        return len(self.colored_dsprites_indices)

class DSpritesDatasetMultiLabel(DSpritesDataset):
    def __init__(self, num_classes, split, dataset_size=None):
        super().__init__(split)
        self.num_classes = num_classes
        self.idx_colored_dsprites_indices_and_labels = self._setup_labels(
            file_name=f"{config['CACHE_DIR']}/dsprites_{split}.pkl")
        random.shuffle(self.idx_colored_dsprites_indices_and_labels)
        if dataset_size is not None and dataset_size < len(self.idx_colored_dsprites_indices_and_labels):
            self.idx_colored_dsprites_indices_and_labels = random.sample(self.idx_colored_dsprites_indices_and_labels, dataset_size)
            logging.info(f"Subsampled dataset to {len(self.idx_colored_dsprites_indices_and_labels)} samples")

    @check_cache(arg_name="file_name", override=config["OVERRIDE_CACHE"])
    def _setup_labels(self, file_name):
        logging.info("Setting up labels for all cues")
        values2labels = {}
        for cue_name in self.latent['names']:
            cue_id = self.latent['names_to_indices'][cue_name]
            values2labels[cue_name] = self._get_values2labels(
                self.num_classes,
                self.latent['sizes'][cue_id]
            )
        idx_colored_dsprites_indices_and_labels = []
        total_samples = len(self.colored_dsprites_indices)
        for idx_colored_dsprites_indices, colored_dsprites_idx in tqdm(enumerate(self.colored_dsprites_indices), total=total_samples, desc="Setting up labels", unit="sample", ncols=100):
            latent_values = self._index_to_latent_values(colored_dsprites_idx)
            labels = tuple(values2labels[cue_name][value] for cue_name, value in zip(self.latent['names'], latent_values))
            idx_colored_dsprites_indices_and_labels.append((idx_colored_dsprites_indices, labels))
        logging.info(f"Labels set up for {total_samples} samples")
        logging.info("Labels set up for all cues")
        return idx_colored_dsprites_indices_and_labels

    def _index_to_latent_values(self, colored_dsprites_idx):
        remainder = colored_dsprites_idx
        values = []
        for base in self.latent['bases']:
            values.append(int(remainder / base))
            remainder = remainder % base
        return values

    def _get_values2labels(self, num_classes, latent_size):
        if num_classes > latent_size:
            raise ValueError("Number of classes cannot exceed latent size")
        values_per_class, extra_values = divmod(latent_size, num_classes)
        class_boundaries = [
            (class_idx + 1) * values_per_class + min(class_idx + 1, extra_values)
            for class_idx in range(num_classes)
        ]
        value_to_class_mapping = {}
        for latent_value in range(latent_size):
            current_class = next(i for i, boundary in enumerate(class_boundaries) if latent_value < boundary)
            value_to_class_mapping[latent_value] = current_class
        return value_to_class_mapping
    
    def __getitem__(self, idx):
        idx_colored_dsprites_indices, labels_list = self.idx_colored_dsprites_indices_and_labels[idx]
        image = super().__getitem__(idx_colored_dsprites_indices)
        return image, [torch.tensor(label, dtype=torch.int64) for label in labels_list]

    def __len__(self):
        return len(self.idx_colored_dsprites_indices_and_labels)


class DiagonalOffDiagonalDataset(DSpritesDatasetMultiLabel):
    def __init__(
        self,
        dataset_size,
        num_classes,
        split,
        bias_cue,
        task_cue,
        off_diag_proportion=0,
    ):
        super().__init__(num_classes=num_classes, split=split)
        self.bias_cue = bias_cue
        self.task_cue = task_cue
        self.biased_idx_colored_dsprites_indices = self._setup_indices(dataset_size, off_diag_proportion)
        random.shuffle(self.biased_idx_colored_dsprites_indices)

    def _setup_indices(self, dataset_size, off_diag_proportion):
        logging.info("Setting up indices for DiagonalOffDiagonalDataset")
        if not 0 <= off_diag_proportion <= 1:
            raise ValueError("off_diag_proportion must be between 0 and 1")
        
        bias_indices = defaultdict(set)
        task_indices = defaultdict(set)
        bias_idx = self.latent['names'].index(self.bias_cue)
        task_idx = self.latent['names'].index(self.task_cue)

        for idx_idx_colored_dsprites_indices_and_labels, (_, labels) in tqdm(enumerate(self.idx_colored_dsprites_indices_and_labels), desc="Creating labels to indices sets"):
            bias_label = labels[bias_idx]
            task_label = labels[task_idx]
            bias_indices[bias_label].add(idx_idx_colored_dsprites_indices_and_labels)
            task_indices[task_label].add(idx_idx_colored_dsprites_indices_and_labels)

        logging.debug(f"Created labels to indices set for bias cue with {len(bias_indices)} labels")
        logging.debug(f"Created labels to indices set for task cue with {len(task_indices)} labels")

        diag_indices = []
        off_diag_indices = []

        for label in tqdm(task_indices.keys(), desc="Processing samples"):
            diag_samples = set.intersection(bias_indices[label], task_indices[label])
            diag_indices.extend(sorted(diag_samples))
            
            if off_diag_proportion > 0:
                off_diag_samples = task_indices[label] - diag_samples
                off_diag_indices.extend(sorted(off_diag_samples))

        num_off_diag = int(dataset_size * off_diag_proportion)
        num_diag = dataset_size - num_off_diag

        if num_diag <= 0 or (off_diag_proportion > 0 and num_off_diag <= 0):
            raise ValueError("Not enough samples for diagonal or off-diagonal cells")

        if num_diag > len(diag_indices):
            raise ValueError("Not enough diagonal samples available")

        indices = random.sample(diag_indices, num_diag)
        if off_diag_proportion > 0:
            indices += random.sample(off_diag_indices, num_off_diag)

        logging.info(f"Created {len(indices)} indices")
        return indices
    
    def __getitem__(self, idx):
        idx_idx_colored_dsprites_indices_and_labels = self.biased_idx_colored_dsprites_indices[idx]
        return super().__getitem__(idx_idx_colored_dsprites_indices_and_labels)

    def __len__(self):
        return len(self.biased_idx_colored_dsprites_indices)
    
class DomainGeneralizationDataset(DSpritesDatasetMultiLabel):
    def __init__(
        self,
        dataset_size,
        num_classes,
        split,
        bias_cue,
        bias_cue_classes,
    ):
        super().__init__(num_classes=num_classes, split=split)
        self.bias_cue = bias_cue
        self.biased_idx_colored_dsprites_indices = self._setup_indices(dataset_size, bias_cue_classes)
        random.shuffle(self.biased_idx_colored_dsprites_indices)

    def _setup_indices(self, dataset_size, bias_cue_classes):
        logging.info("Setting up indices for DomainGeneralizationDataset")
        if any(class_value >= self.num_classes for class_value in bias_cue_classes):
            raise ValueError("All bias_cue_classes values must be less than num_classes")
        
        bias_idx = self.latent['names'].index(self.bias_cue)

        relevant_indices = [
            idx for idx, (_, labels) in enumerate(self.idx_colored_dsprites_indices_and_labels)
            if labels[bias_idx] in bias_cue_classes
        ]

        indices = random.sample(relevant_indices, dataset_size)

        logging.info(f"Created {len(indices)} indices")
        return indices
    
    def __getitem__(self, idx):
        idx_idx_colored_dsprites_indices_and_labels = self.biased_idx_colored_dsprites_indices[idx]
        return super().__getitem__(idx_idx_colored_dsprites_indices_and_labels)

    def __len__(self):
        return len(self.biased_idx_colored_dsprites_indices)


def load_dataloader(
    data_setting: str = "diagonal",
    split: str = "train",
    dataset_size: int = config["TRAIN_DATASET_SIZE"],
    bias_cue: str = None,
    task_cue: str = None,
    off_diag_proportion: float = 0,
    bias_cue_classes: list = None
) -> torch.utils.data.DataLoader:
    
    if data_setting == "unbiased":
        logging.info(f"Creating unbiased dsprites dataloader for split: {split}")
        dataset = DSpritesDatasetMultiLabel(
            num_classes=config["NUM_CLASSES"],
            split=split,
            dataset_size=dataset_size
        )
    elif data_setting == "diagonal":
        logging.info(f"Creating biased dsprites dataloader for split: {split}")
        dataset = DiagonalOffDiagonalDataset(
            dataset_size=dataset_size,
            num_classes=config["NUM_CLASSES"],
            split=split,
            bias_cue=bias_cue,
            task_cue=task_cue,
            off_diag_proportion=off_diag_proportion,
        )
    elif data_setting == "domain_generalization":
        dataset = DomainGeneralizationDataset(
            dataset_size=dataset_size,
            num_classes=config["NUM_CLASSES"],
            split=split,
            bias_cue=bias_cue,
            bias_cue_classes=bias_cue_classes,
        )
    else:
        raise ValueError(f"Unknown data setting: {data_setting}")

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True
    )
    logging.info(f"Created dataloader with {len(dataset)} samples")
    return dataloader

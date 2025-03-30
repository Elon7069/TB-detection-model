import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split, StratifiedKFold
import cv2
from glob import glob
from scipy.ndimage import gaussian_filter
import albumentations as A
import random
from collections import defaultdict

class TBDataLoader:
    """
    Data loader for TB detection from X-ray images.
    Handles loading, preprocessing, and splitting of data for federated learning.
    Implements robust augmentation and domain adaptation techniques for improved generalization.
    """
    
    def __init__(self, data_dir, img_size=(224, 224), split_ratio=0.2, seed=42, 
                 augmentation_strength='medium', simulate_domain_shift=True):
        """
        Initialize the TB data loader with enhanced generalization capabilities.
        
        Args:
            data_dir: Root directory containing 'train/normal', 'train/tb', and 'test' folders
            img_size: Input image size for the model (height, width)
            split_ratio: Train/validation split ratio
            seed: Random seed for reproducibility
            augmentation_strength: Level of augmentation ('light', 'medium', 'strong')
            simulate_domain_shift: Whether to simulate domain shift (different hospitals/machines)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.split_ratio = split_ratio
        self.seed = seed
        self.augmentation_strength = augmentation_strength
        self.simulate_domain_shift = simulate_domain_shift
        
        # Paths for training and test data
        self.train_normal_dir = os.path.join(data_dir, 'train', 'normal')
        self.train_tb_dir = os.path.join(data_dir, 'train', 'tb')
        self.test_dir = os.path.join(data_dir, 'test')
        
        # Validate the dataset structure exists
        self._validate_dataset_structure()
        
        # Setup augmentation pipelines based on strength
        self._setup_augmentation_pipelines()
        
        # Initialize domain simulation parameters if enabled
        if self.simulate_domain_shift:
            self._setup_domain_simulation()
            
    def _setup_augmentation_pipelines(self):
        """Set up data augmentation pipelines with varying strengths."""
        # Light augmentation (basic operations)
        self.light_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3, brightness_limit=0.1, contrast_limit=0.1),
            A.Rotate(limit=10, p=0.5),
        ])
        
        # Medium augmentation (standard medical imaging augmentation)
        self.medium_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            A.Rotate(limit=15, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.Blur(blur_limit=3, p=0.3),  # Simulate image quality variations
            A.GaussNoise(var_limit=(5, 20), p=0.3),  # Simulate sensor noise
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),  # Simulate anatomical variations
        ])
        
        # Strong augmentation (for maximum generalization)
        self.strong_aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.7, brightness_limit=0.3, contrast_limit=0.3),
            A.Rotate(limit=20, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.6),
            A.Blur(blur_limit=5, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.GridDistortion(num_steps=7, distort_limit=0.3, p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.5),  # Improve contrast in different exposure settings
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),  # Simulate different X-ray energies
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),  # Simulate patient positioning
            A.Cutout(num_holes=8, max_h_size=16, max_w_size=16, p=0.3),  # Simulate occlusions or artifacts
        ])
        
    def _setup_domain_simulation(self):
        """Set up parameters for simulating domain shift (different hospitals/machines)."""
        # Create synthetic "hospital domains" with different imaging characteristics
        self.domains = [
            {  # Hospital 1: Higher contrast, sharper images
                "contrast_factor": 1.2,
                "brightness_offset": 0,
                "noise_level": 5,
                "blur_sigma": 0.5,
                "resolution_factor": 1.0
            },
            {  # Hospital 2: Lower contrast, brighter images
                "contrast_factor": 0.9,
                "brightness_offset": 15,
                "noise_level": 10,
                "blur_sigma": 1.0,
                "resolution_factor": 0.9
            },
            {  # Hospital 3: Standard images with more noise
                "contrast_factor": 1.0,
                "brightness_offset": -5,
                "noise_level": 20,
                "blur_sigma": 0.7,
                "resolution_factor": 1.0
            },
            {  # Hospital 4: Older equipment simulation (more blur, less contrast)
                "contrast_factor": 0.8,
                "brightness_offset": 10,
                "noise_level": 15,
                "blur_sigma": 1.5,
                "resolution_factor": 0.8
            }
        ]
    
    def _apply_domain_simulation(self, img, domain_idx):
        """
        Apply domain-specific transformations to simulate different hospital equipment.
        
        Args:
            img: Input image
            domain_idx: Index of the domain to simulate
            
        Returns:
            Transformed image
        """
        domain = self.domains[domain_idx]
        
        # Apply domain-specific transformations
        # 1. Contrast adjustment
        img = img.astype(np.float32)
        mean = np.mean(img)
        img = (img - mean) * domain["contrast_factor"] + mean
        
        # 2. Brightness adjustment
        img = img + domain["brightness_offset"]
        
        # 3. Add noise
        if domain["noise_level"] > 0:
            noise = np.random.normal(0, domain["noise_level"], img.shape)
            img = img + noise
        
        # 4. Apply blur
        if domain["blur_sigma"] > 0:
            img = gaussian_filter(img, sigma=domain["blur_sigma"])
        
        # 5. Simulate resolution differences
        if domain["resolution_factor"] < 1.0:
            h, w = img.shape[:2]
            new_h, new_w = int(h * domain["resolution_factor"]), int(w * domain["resolution_factor"])
            img = cv2.resize(img, (new_w, new_h))
            img = cv2.resize(img, (w, h))
        
        # Clip values to valid range
        img = np.clip(img, 0, 255)
        
        return img
        
    def _validate_dataset_structure(self):
        """Validate that the dataset directories exist."""
        dirs_to_check = [self.train_normal_dir, self.train_tb_dir, self.test_dir]
        for dir_path in dirs_to_check:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
    
    def _extract_image_characteristics(self, img):
        """
        Extract key characteristics from an image to enable stratified sampling.
        
        Args:
            img: Input image
            
        Returns:
            Dictionary of image characteristics
        """
        # Calculate image statistics
        if img is None:
            return None
            
        try:
            # Convert to grayscale if it's RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = img.astype(np.uint8)
            
            # Calculate basic statistics
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist.flatten() / np.sum(hist)  # Normalize
            
            # Calculate texture features (GLCM)
            # For simplicity, we'll use standard deviation as a proxy for texture
            
            # Brightness category (low, medium, high)
            if mean_val < 80:
                brightness = "low"
            elif mean_val < 160:
                brightness = "medium"
            else:
                brightness = "high"
                
            # Contrast category (low, medium, high)
            if std_val < 30:
                contrast = "low"
            elif std_val < 60:
                contrast = "medium"
            else:
                contrast = "high"
                
            # Calculate edges as a measure of detail
            edges = cv2.Canny(gray, 100, 200)
            edge_percentage = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Detail level category
            if edge_percentage < 0.05:
                detail = "low"
            elif edge_percentage < 0.15:
                detail = "medium"
            else:
                detail = "high"
            
            # Return characteristics
            return {
                "brightness": brightness,
                "contrast": contrast,
                "detail": detail,
                "mean": mean_val,
                "std": std_val,
                "edge_percentage": edge_percentage
            }
            
        except Exception as e:
            print(f"Error extracting image characteristics: {e}")
            return None
                
    def _load_and_preprocess_image(self, image_path, augment=False, domain_idx=None):
        """
        Load and preprocess a single image with enhanced augmentation options.
        
        Args:
            image_path: Path to the image file
            augment: Whether to apply data augmentation
            domain_idx: Domain index for simulation (if None, no simulation is applied)
            
        Returns:
            Preprocessed image as a numpy array
        """
        # Load image and resize
        img = cv2.imread(image_path)
        if img is None:
            # Handle corrupted images
            print(f"Warning: Could not load image {image_path}")
            return None
            
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Extract characteristics before augmentation (for analysis)
        characteristics = self._extract_image_characteristics(img)
        
        # Apply domain simulation if specified
        if self.simulate_domain_shift and domain_idx is not None:
            img = self._apply_domain_simulation(img, domain_idx)
        
        # Apply data augmentation if specified
        if augment:
            if self.augmentation_strength == 'light':
                img = self.light_aug(image=img)['image']
            elif self.augmentation_strength == 'medium':
                img = self.medium_aug(image=img)['image']
            elif self.augmentation_strength == 'strong':
                img = self.strong_aug(image=img)['image']
        
        # Apply preprocessing for ResNet50
        img = img.astype(np.float32)
        img = preprocess_input(img)
        
        return img, characteristics
    
    def _create_dataframe(self):
        """
        Create a DataFrame containing image paths and labels.
        
        Returns:
            DataFrame with 'path', 'label', and image characteristics columns
        """
        # Get image paths
        normal_paths = glob(os.path.join(self.train_normal_dir, '*.*'))
        tb_paths = glob(os.path.join(self.train_tb_dir, '*.*'))
        
        # Create data entries
        data = []
        
        # Process all images and extract their characteristics
        for path in normal_paths:
            img, characteristics = self._load_and_preprocess_image(path)
            if img is not None and characteristics is not None:
                entry = {
                    'path': path,
                    'label': 0,  # Normal
                    'brightness': characteristics['brightness'],
                    'contrast': characteristics['contrast'],
                    'detail': characteristics['detail'],
                    'mean': characteristics['mean'],
                    'std': characteristics['std'],
                    'edge_percentage': characteristics['edge_percentage']
                }
                data.append(entry)
        
        for path in tb_paths:
            img, characteristics = self._load_and_preprocess_image(path)
            if img is not None and characteristics is not None:
                entry = {
                    'path': path,
                    'label': 1,  # TB
                    'brightness': characteristics['brightness'],
                    'contrast': characteristics['contrast'],
                    'detail': characteristics['detail'],
                    'mean': characteristics['mean'],
                    'std': characteristics['std'],
                    'edge_percentage': characteristics['edge_percentage']
                }
                data.append(entry)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def _create_test_dataframe(self):
        """
        Create a DataFrame containing test image paths.
        
        Returns:
            DataFrame with 'path', 'id', and image characteristics columns
        """
        # Get test image paths
        test_paths = glob(os.path.join(self.test_dir, '*.*'))
        
        # Create data entries
        data = []
        
        for path in test_paths:
            # Extract ID from filename (e.g., TB_test_0001.png -> TB_test_0001)
            file_id = os.path.splitext(os.path.basename(path))[0]
            img, characteristics = self._load_and_preprocess_image(path)
            
            if img is not None and characteristics is not None:
                entry = {
                    'path': path,
                    'id': file_id,
                    'brightness': characteristics['brightness'],
                    'contrast': characteristics['contrast'],
                    'detail': characteristics['detail'],
                    'mean': characteristics['mean'],
                    'std': characteristics['std'],
                    'edge_percentage': characteristics['edge_percentage']
                }
                data.append(entry)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def get_train_val_data(self, stratify_by_characteristics=True):
        """
        Get training and validation data with stratified sampling.
        
        Args:
            stratify_by_characteristics: Whether to stratify by image characteristics
            
        Returns:
            X_train, y_train, X_val, y_val
        """
        # Create DataFrame with image characteristics
        df = self._create_dataframe()
        
        # Split into train and validation sets
        if stratify_by_characteristics and 'brightness' in df.columns and 'contrast' in df.columns:
            # Create a combined stratification column
            df['strat_group'] = df['label'].astype(str) + '_' + df['brightness'] + '_' + df['contrast']
            train_df, val_df = train_test_split(
                df, 
                test_size=self.split_ratio, 
                stratify=df['strat_group'],
                random_state=self.seed
            )
        else:
            # Fallback to basic label stratification
            train_df, val_df = train_test_split(
                df, 
                test_size=self.split_ratio, 
                stratify=df['label'],
                random_state=self.seed
            )
        
        # Load and preprocess images for training set with augmentation
        X_train = []
        y_train = []
        
        for _, row in train_df.iterrows():
            # Apply augmentation and domain simulation for training
            domain_idx = np.random.randint(0, len(self.domains)) if self.simulate_domain_shift else None
            img, _ = self._load_and_preprocess_image(row['path'], augment=True, domain_idx=domain_idx)
            if img is not None:
                X_train.append(img)
                y_train.append(row['label'])
        
        # Load and preprocess images for validation set (no augmentation)
        X_val = []
        y_val = []
        
        for _, row in val_df.iterrows():
            img, _ = self._load_and_preprocess_image(row['path'], augment=False)
            if img is not None:
                X_val.append(img)
                y_val.append(row['label'])
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        
        return X_train, y_train, X_val, y_val
    
    def get_federated_data(self, num_clients=3, stratify_by_characteristics=True):
        """
        Split the training data into multiple client datasets for federated learning,
        with domain simulation to mimic different hospitals.
        
        Args:
            num_clients: Number of federated learning clients
            stratify_by_characteristics: Whether to stratify by image characteristics
            
        Returns:
            List of (X_train, y_train) tuples, one for each client
        """
        # Create DataFrame
        df = self._create_dataframe()
        
        # Split into train and validation sets first
        if stratify_by_characteristics and 'brightness' in df.columns and 'contrast' in df.columns:
            # Create a combined stratification column
            df['strat_group'] = df['label'].astype(str) + '_' + df['brightness'] + '_' + df['contrast']
            train_df, _ = train_test_split(
                df, 
                test_size=self.split_ratio, 
                stratify=df['strat_group'],
                random_state=self.seed
            )
        else:
            train_df, _ = train_test_split(
                df, 
                test_size=self.split_ratio, 
                stratify=df['label'],
                random_state=self.seed
            )
        
        # For more realistic federated learning, assign different image characteristics
        # to different clients to simulate hospital equipment variations
        client_data = []
        
        if stratify_by_characteristics and 'brightness' in df.columns and 'contrast' in df.columns:
            # Group by characteristics
            grouped = train_df.groupby(['brightness', 'contrast', 'detail', 'label'])
            
            # Distribute groups across clients to ensure diversity
            client_dfs = [[] for _ in range(num_clients)]
            
            for name, group in grouped:
                # Assign this characteristic group primarily to one client
                primary_client = random.randint(0, num_clients-1)
                
                # Give 70% to primary client, distribute rest among others
                group = group.sample(frac=1, random_state=self.seed).reset_index(drop=True)
                split_idx = int(len(group) * 0.7)
                
                client_dfs[primary_client].append(group.iloc[:split_idx])
                
                # Distribute remaining 30% among other clients
                remaining = group.iloc[split_idx:]
                if len(remaining) > 0:
                    splits = np.array_split(remaining, num_clients-1)
                    j = 0
                    for i in range(num_clients):
                        if i != primary_client and j < len(splits):
                            client_dfs[i].append(splits[j])
                            j += 1
            
            # Combine grouped dataframes for each client
            for i in range(num_clients):
                if client_dfs[i]:
                    client_df = pd.concat(client_dfs[i])
                    
                    # Load and preprocess images with client-specific domain simulation
                    X_client = []
                    y_client = []
                    
                    for _, row in client_df.iterrows():
                        # Use client index to simulate consistent hospital domain
                        img, _ = self._load_and_preprocess_image(
                            row['path'], 
                            augment=True, 
                            domain_idx=i % len(self.domains)
                        )
                        if img is not None:
                            X_client.append(img)
                            y_client.append(row['label'])
                    
                    if X_client:
                        X_client = np.array(X_client)
                        y_client = np.array(y_client)
                        client_data.append((X_client, y_client))
                    
        else:
            # Fallback to basic split by normal and TB
            normal_df = train_df[train_df['label'] == 0]
            tb_df = train_df[train_df['label'] == 1]
            
            # Create splits
            normal_splits = np.array_split(normal_df.sample(frac=1, random_state=self.seed), num_clients)
            tb_splits = np.array_split(tb_df.sample(frac=1, random_state=self.seed), num_clients)
            
            # Create a dataset for each client
            for i in range(num_clients):
                client_df = pd.concat([normal_splits[i], tb_splits[i]])
                client_df = client_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
                
                # Load and preprocess images with client-specific domain
                X_client = []
                y_client = []
                
                for _, row in client_df.iterrows():
                    img, _ = self._load_and_preprocess_image(
                        row['path'], 
                        augment=True, 
                        domain_idx=i % len(self.domains)
                    )
                    if img is not None:
                        X_client.append(img)
                        y_client.append(row['label'])
                
                X_client = np.array(X_client)
                y_client = np.array(y_client)
                
                client_data.append((X_client, y_client))
        
        # Ensure we have the requested number of clients
        while len(client_data) < num_clients:
            # If we have fewer clients than requested, duplicate and modify some data
            source_idx = random.randint(0, len(client_data)-1)
            X_source, y_source = client_data[source_idx]
            
            # Apply strong augmentation to create a new synthetic client
            X_new = []
            for img in X_source:
                # Apply random transformations
                transformed = self.strong_aug(image=img.astype(np.uint8))['image']
                X_new.append(preprocess_input(transformed.astype(np.float32)))
            
            X_new = np.array(X_new)
            client_data.append((X_new, y_source))
        
        # Generate a report on data distribution across clients
        self._generate_client_distribution_report(client_data)
        
        return client_data
    
    def _generate_client_distribution_report(self, client_data):
        """Generate a report on data distribution across federated clients."""
        print("\n=== Federated Client Data Distribution ===")
        
        for i, (X, y) in enumerate(client_data):
            n_normal = np.sum(y == 0)
            n_tb = np.sum(y == 1)
            total = len(y)
            
            print(f"Client {i+1}: {total} samples ({n_normal} normal, {n_tb} TB), TB rate: {n_tb/total:.2%}")
        
        print("This distribution simulates realistic hospital data variations.")
    
    def get_test_data(self, apply_multiple_domains=True):
        """
        Get test data with option to test across multiple simulated domains.
        
        Args:
            apply_multiple_domains: Whether to create multiple versions of the test set
                                   with different domain transformations
        
        Returns:
            If apply_multiple_domains=False: (X_test, test_ids)
            If apply_multiple_domains=True: (X_test_dict, test_ids)
                where X_test_dict is a dictionary with keys 'original', 'domain1', 'domain2', etc.
        """
        # Create test DataFrame
        test_df = self._create_test_dataframe()
        
        # Load and preprocess images for standard test set
        X_test = []
        test_ids = []
        
        for _, row in test_df.iterrows():
            img, _ = self._load_and_preprocess_image(row['path'])
            if img is not None:
                X_test.append(img)
                test_ids.append(row['id'])
        
        X_test = np.array(X_test)
        
        # If testing across domains is not required, return the standard test set
        if not apply_multiple_domains or not self.simulate_domain_shift:
            return X_test, test_ids
        
        # Otherwise, create a version of the test set for each domain
        X_test_dict = {'original': X_test}
        
        for domain_idx in range(len(self.domains)):
            X_domain = []
            
            for _, row in test_df.iterrows():
                img, _ = self._load_and_preprocess_image(row['path'], domain_idx=domain_idx)
                if img is not None:
                    X_domain.append(img)
            
            if X_domain:
                X_test_dict[f'domain{domain_idx+1}'] = np.array(X_domain)
        
        return X_test_dict, test_ids
        
    def analyze_data_characteristics(self):
        """
        Analyze the dataset characteristics to understand the distribution.
        
        Returns:
            DataFrame with analysis results
        """
        # Create DataFrames
        train_df = self._create_dataframe()
        test_df = self._create_test_dataframe()
        
        # Add dataset column
        train_df['dataset'] = 'train'
        test_df['dataset'] = 'test'
        
        # Merge for unified analysis
        combined_df = pd.concat([
            train_df[['brightness', 'contrast', 'detail', 'label', 'dataset', 'mean', 'std', 'edge_percentage']],
            test_df[['brightness', 'contrast', 'detail', 'dataset', 'mean', 'std', 'edge_percentage']]
        ]).reset_index(drop=True)
        
        # Generate analysis
        analysis = {}
        
        # Count by dataset and label
        counts = combined_df.groupby(['dataset', 'label']).size().unstack(fill_value=0)
        analysis['counts'] = counts
        
        # Distribution of characteristics
        for characteristic in ['brightness', 'contrast', 'detail']:
            dist = combined_df.groupby(['dataset', characteristic]).size().unstack(fill_value=0)
            analysis[f'{characteristic}_distribution'] = dist
        
        # Statistical analysis
        for feature in ['mean', 'std', 'edge_percentage']:
            stats = combined_df.groupby(['dataset'])[feature].agg(['mean', 'std', 'min', 'max', 'median'])
            analysis[f'{feature}_stats'] = stats
        
        print("\n=== Dataset Characteristics Analysis ===")
        for key, value in analysis.items():
            print(f"\n{key.replace('_', ' ').title()}:")
            print(value)
        
        return analysis 
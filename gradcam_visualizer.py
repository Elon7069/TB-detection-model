import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

class GradCAMVisualizer:
    """
    Enhanced GradCAM visualization for TB detection model with clinical interpretation features.
    
    This class implements Gradient-weighted Class Activation Mapping (Grad-CAM)
    for visualizing the regions of the X-ray that influence the model's decision,
    with anatomical context and clinical interpretability for healthcare professionals.
    """
    
    def __init__(self, model, last_conv_layer_name=None):
        """
        Initialize GradCAM visualizer.
        
        Args:
            model: The TB detection model
            last_conv_layer_name: Name of the last convolutional layer (optional)
        """
        self.model = model
        
        # Find the last convolutional layer if not specified
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break
        
        if last_conv_layer_name is None:
            raise ValueError("Could not find a convolutional layer in the model")
        
        self.last_conv_layer_name = last_conv_layer_name
        
        # Create Grad model that outputs both the predictions and the last conv layer activations
        grad_model = Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )
        
        self.grad_model = grad_model
        
        # Define anatomical regions of interest for chest X-rays
        self.anatomical_regions = {
            'right_upper_lung': {'x1': 0.15, 'y1': 0.15, 'x2': 0.45, 'y2': 0.35, 'color': 'blue'},
            'left_upper_lung': {'x1': 0.55, 'y1': 0.15, 'x2': 0.85, 'y2': 0.35, 'color': 'green'},
            'right_mid_lung': {'x1': 0.15, 'y1': 0.35, 'x2': 0.45, 'y2': 0.55, 'color': 'purple'},
            'left_mid_lung': {'x1': 0.55, 'y1': 0.35, 'x2': 0.85, 'y2': 0.55, 'color': 'yellow'},
            'right_lower_lung': {'x1': 0.15, 'y1': 0.55, 'x2': 0.45, 'y2': 0.75, 'color': 'cyan'},
            'left_lower_lung': {'x1': 0.55, 'y1': 0.55, 'x2': 0.85, 'y2': 0.75, 'color': 'magenta'},
            'heart': {'x1': 0.40, 'y1': 0.30, 'x2': 0.60, 'y2': 0.60, 'color': 'red'}
        }
    
    def generate_gradcam(self, img_array, class_idx=0, eps=1e-8):
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            img_array: Input image as a numpy array
            class_idx: Class index (0 for binary classification)
            eps: Small constant to avoid division by zero
            
        Returns:
            Heatmap and superimposed image
        """
        # Ensure image has batch dimension
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        
        # Calculate gradients and feature map
        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(img_array)
            loss = predictions[:, class_idx]
        
        # Extract gradients
        grads = tape.gradient(loss, conv_output)
        
        # Average gradients spatially
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps with gradients
        conv_output = conv_output[0]
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[-1]):
            conv_output[:, :, i] *= pooled_grads[i]
        
        # Average all feature maps to get heatmap
        heatmap = np.mean(conv_output, axis=-1)
        
        # Apply ReLU to heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        heatmap = heatmap / (np.max(heatmap) + eps)
        
        # Resize heatmap to match input image size
        img_height, img_width = img_array.shape[1:3]
        heatmap = cv2.resize(heatmap, (img_width, img_height))
        
        # Convert heatmap to RGB using a color map
        heatmap_rgb = np.uint8(255 * heatmap)
        heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
        
        # Prepare original image for overlay
        orig_img = img_array[0].copy()
        
        # Denormalize if necessary
        if orig_img.max() <= 1.0:
            orig_img = orig_img * 255
        
        # Ensure image is RGB
        if len(orig_img.shape) == 2:
            orig_img = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif orig_img.shape[2] == 1:
            orig_img = cv2.cvtColor(orig_img.astype(np.uint8).squeeze(axis=2), cv2.COLOR_GRAY2RGB)
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(
            orig_img.astype(np.uint8),
            0.6,
            heatmap_rgb,
            0.4,
            0
        )
        
        return heatmap, superimposed_img, heatmap_rgb
    
    def analyze_activation_by_region(self, heatmap):
        """
        Analyze activation levels within predefined anatomical regions.
        
        Args:
            heatmap: Generated Grad-CAM heatmap
            
        Returns:
            Dictionary of regions with their activation statistics
        """
        height, width = heatmap.shape
        region_statistics = {}
        
        for region_name, coords in self.anatomical_regions.items():
            # Convert relative coordinates to absolute pixel values
            x1 = int(coords['x1'] * width)
            y1 = int(coords['y1'] * height)
            x2 = int(coords['x2'] * width)
            y2 = int(coords['y2'] * height)
            
            # Extract region from heatmap
            region_heatmap = heatmap[y1:y2, x1:x2]
            
            if region_heatmap.size > 0:
                # Compute statistics for region
                region_statistics[region_name] = {
                    'mean_activation': float(np.mean(region_heatmap)),
                    'max_activation': float(np.max(region_heatmap)),
                    'activation_area': float(np.sum(region_heatmap > 0.5) / region_heatmap.size),
                    'coords': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                }
        
        return region_statistics
    
    def get_clinical_findings(self, region_statistics, prediction_score):
        """
        Generate clinical findings based on activation patterns and prediction score.
        
        Args:
            region_statistics: Dictionary of region statistics 
            prediction_score: Model's prediction score
            
        Returns:
            Dictionary of clinical findings
        """
        findings = {
            'main_findings': [],
            'affected_regions': [],
            'pattern_type': None,
            'confidence_level': None,
            'notes': []
        }
        
        # Determine affected regions (regions with high activation)
        for region, stats in region_statistics.items():
            if stats['mean_activation'] > 0.4:
                findings['affected_regions'].append({
                    'name': region,
                    'activation': stats['mean_activation'],
                    'extent': stats['activation_area']
                })
        
        # Sort affected regions by activation level
        findings['affected_regions'] = sorted(
            findings['affected_regions'], 
            key=lambda x: x['activation'], 
            reverse=True
        )
        
        # Determine main findings based on activation patterns
        if len(findings['affected_regions']) > 0:
            # Check pattern type
            if any('upper_lung' in region['name'] for region in findings['affected_regions']):
                findings['pattern_type'] = 'Apical/Upper lobe predominant'
                findings['main_findings'].append('Possible upper lobe infiltrates')
            
            if any('lower_lung' in region['name'] for region in findings['affected_regions']):
                findings['pattern_type'] = 'Lower lobe predominant' if findings['pattern_type'] is None else 'Diffuse'
                findings['main_findings'].append('Possible lower lobe infiltrates')
            
            if all(region['activation'] > 0.6 for region in findings['affected_regions'][:2]):
                findings['main_findings'].append('Prominent opacities in highly activated regions')
            
            if any(region['activation'] > 0.7 for region in findings['affected_regions']):
                findings['main_findings'].append('Focal high-intensity finding')
        else:
            findings['main_findings'].append('No significant focal findings')
            findings['pattern_type'] = 'Non-specific'
        
        # Determine confidence level based on prediction score
        if prediction_score > 0.9:
            findings['confidence_level'] = 'High'
        elif prediction_score > 0.7:
            findings['confidence_level'] = 'Moderate'
        else:
            findings['confidence_level'] = 'Low'
            findings['notes'].append('Consider clinical correlation')
        
        # Add TB-specific notes based on pattern
        if prediction_score > 0.7:
            if findings['pattern_type'] == 'Apical/Upper lobe predominant':
                findings['notes'].append('Pattern consistent with post-primary TB')
            elif findings['pattern_type'] == 'Diffuse':
                findings['notes'].append('Pattern may suggest miliary TB')
            elif findings['pattern_type'] == 'Lower lobe predominant':
                findings['notes'].append('Atypical distribution for TB, consider other diagnoses')
        
        return findings
    
    def visualize(self, img_array, class_idx=0, title=None, save_path=None, with_clinical_interpretation=True):
        """
        Visualize Grad-CAM results with medical interpretation.
        
        Args:
            img_array: Input image as a numpy array
            class_idx: Class index (0 for binary classification)
            title: Title for the plot
            save_path: Path to save the visualization (optional)
            with_clinical_interpretation: Whether to include clinical interpretation
            
        Returns:
            Matplotlib figure
        """
        # Generate heatmap and superimposed image
        heatmap, superimposed_img, _ = self.generate_gradcam(img_array, class_idx)
        
        # Get prediction score from the title or calculate if not available
        prediction_score = 0.0
        if title and "Prediction:" in title:
            try:
                prediction_score = float(title.split("Prediction:")[1].split('-')[0].strip())
            except:
                # If we can't extract from title, make a new prediction
                if len(img_array.shape) == 3:
                    img_array = np.expand_dims(img_array, axis=0)
                prediction_score = self.model.predict(img_array)[0][0]
        
        # Analyze activation by anatomical region
        region_statistics = self.analyze_activation_by_region(heatmap)
        
        # Get clinical findings
        if with_clinical_interpretation:
            clinical_findings = self.get_clinical_findings(region_statistics, prediction_score)
        
        # Determine layout based on interpretation needs
        if with_clinical_interpretation:
            # Create figure with subplots for visualization and interpretation
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(2, 6)
            
            # Main visualization row
            ax_orig = fig.add_subplot(gs[0, 0:2])
            ax_heatmap = fig.add_subplot(gs[0, 2:4])
            ax_superimposed = fig.add_subplot(gs[0, 4:6])
            
            # Clinical interpretation row
            ax_region = fig.add_subplot(gs[1, 0:3])
            ax_findings = fig.add_subplot(gs[1, 3:6])
        else:
            # Create figure with three subplots for basic visualization
            fig, (ax_orig, ax_heatmap, ax_superimposed) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Display original image
        orig_img = img_array[0].copy() if len(img_array.shape) == 4 else img_array.copy()
        
        # Denormalize if necessary
        if orig_img.max() <= 1.0:
            orig_img = orig_img * 255
        
        # Ensure image is RGB
        if len(orig_img.shape) == 2:
            orig_img = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif orig_img.shape[2] == 1:
            orig_img = cv2.cvtColor(orig_img.astype(np.uint8).squeeze(axis=2), cv2.COLOR_GRAY2RGB)
        
        ax_orig.imshow(orig_img.astype(np.uint8))
        ax_orig.set_title('Original Chest X-ray')
        ax_orig.axis('off')
        
        # Display heatmap
        ax_heatmap.imshow(heatmap, cmap='jet')
        ax_heatmap.set_title('Activation Heatmap')
        ax_heatmap.axis('off')
        
        # Display superimposed image
        ax_superimposed.imshow(superimposed_img)
        ax_superimposed.set_title('X-ray with Activation Overlay')
        ax_superimposed.axis('off')
        
        # If including clinical interpretation, add anatomical overlay and findings
        if with_clinical_interpretation:
            # Set main title if provided
            if title:
                plt.suptitle(title, fontsize=16, y=0.98)
            
            # Create anatomical region visualization
            ax_region.imshow(orig_img.astype(np.uint8))
            ax_region.set_title('Anatomical Regions of Interest')
            
            # Add region overlays with activation stats
            for region_name, region_stats in region_statistics.items():
                coords = region_stats['coords']
                activation = region_stats['mean_activation']
                color = self.anatomical_regions[region_name]['color']
                
                # Style based on activation level
                alpha = min(0.3 + activation * 0.7, 0.8)  # More visible for higher activation
                linewidth = 1 + 2 * int(activation > 0.5)  # Thicker for higher activation
                
                # Add rectangle
                rect = Rectangle(
                    (coords['x1'], coords['y1']),
                    coords['x2'] - coords['x1'],
                    coords['y2'] - coords['y1'],
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor=color,
                    alpha=alpha,
                    label=f"{region_name}: {activation:.2f}"
                )
                ax_region.add_patch(rect)
            
            # Add legend outside the plot
            ax_region.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            ax_region.axis('off')
            
            # Display clinical findings
            ax_findings.axis('off')
            ax_findings.set_title('Clinical Interpretation')
            
            # Create text for findings
            findings_text = [
                f"Prediction: {'TB' if prediction_score > 0.5 else 'Normal'} (Score: {prediction_score:.2f})",
                f"Confidence: {clinical_findings['confidence_level']}",
                f"Pattern: {clinical_findings['pattern_type'] or 'Non-specific'}\n"
            ]
            
            if clinical_findings['main_findings']:
                findings_text.append("Main Findings:")
                for idx, finding in enumerate(clinical_findings['main_findings'], 1):
                    findings_text.append(f"  {idx}. {finding}")
                findings_text.append("")
            
            if clinical_findings['affected_regions']:
                findings_text.append("Affected Regions (by activation):")
                for idx, region in enumerate(clinical_findings['affected_regions'][:3], 1):
                    findings_text.append(f"  {idx}. {region['name']} - Activation: {region['activation']:.2f}, Extent: {region['extent']:.2f}")
                findings_text.append("")
            
            if clinical_findings['notes']:
                findings_text.append("Notes:")
                for note in clinical_findings['notes']:
                    findings_text.append(f"  • {note}")
            
            # Add disclaimer
            findings_text.append("\nDisclaimer: This is an AI-generated interpretation to")
            findings_text.append("assist healthcare professionals. It does not replace")
            findings_text.append("clinical judgment or radiological expertise.")
            
            ax_findings.text(0.05, 0.95, "\n".join(findings_text), 
                             transform=ax_findings.transAxes,
                             fontsize=10, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
        else:
            # Set main title if provided (for basic visualization)
            if title:
                plt.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig, clinical_findings if with_clinical_interpretation else (fig, None)

def load_lung_segmentation_model():
    """
    Load a model for lung segmentation to improve anatomical region detection.
    This is a placeholder - in a real implementation, you would load an actual 
    segmentation model or include the code for lung boundary detection.
    
    Returns:
        A pretend lung segmentation model (None for now)
    """
    # In a real implementation, this would load a model like U-Net
    # trained for lung segmentation
    return None 
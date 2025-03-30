import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import json
import random
from matplotlib.patches import Rectangle

# Set page config
st.set_page_config(
    page_title="TB Detection with Federated Learning",
    page_icon="🫁",
    layout="wide"
)

# Define constants
IMG_SIZE = (224, 224)

def simulate_prediction():
    """Simulate a model prediction (random value for demo)"""
    return random.uniform(0.2, 0.9)

def simulate_gradcam(img, pred):
    """Simulate GradCAM visualization for demo purposes"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Simulate heatmap with Matplotlib
    x = np.linspace(0, 1, img.size[0])
    y = np.linspace(0, 1, img.size[1])
    X, Y = np.meshgrid(x, y)
    
    # Create a simulated heatmap focused on upper parts of lungs
    Z = np.exp(-((X-0.3)**2 + (Y-0.3)**2)/0.05) + 0.8*np.exp(-((X-0.7)**2 + (Y-0.3)**2)/0.05)
    
    # If prediction is higher, make the heatmap more intense
    if pred > 0.5:
        Z = Z * 1.5
    
    # Display the heatmap overlay
    ax[1].imshow(img)
    heatmap = ax[1].imshow(Z, cmap='jet', alpha=0.5)
    ax[1].set_title(f"Grad-CAM Visualization\nPrediction: {pred:.2f}")
    ax[1].axis('off')
    
    # Add anatomical region markers
    if pred > 0.5:
        # Highlight upper lungs more if TB predicted
        rect1 = Rectangle((img.size[0]*0.2, img.size[1]*0.2), img.size[0]*0.2, img.size[1]*0.2, 
                         linewidth=2, edgecolor='lime', facecolor='none')
        rect2 = Rectangle((img.size[0]*0.6, img.size[1]*0.2), img.size[0]*0.2, img.size[1]*0.2,
                         linewidth=2, edgecolor='lime', facecolor='none')
        ax[1].add_patch(rect1)
        ax[1].add_patch(rect2)
        
        # Add labels
        ax[1].text(img.size[0]*0.3, img.size[1]*0.18, "Right Upper Lung", color='white', fontsize=9)
        ax[1].text(img.size[0]*0.7, img.size[1]*0.18, "Left Upper Lung", color='white', fontsize=9)
    
    plt.tight_layout()
    return fig

def get_clinical_findings(pred):
    """Generate simulated clinical findings based on prediction"""
    confidence_level = "High" if pred > 0.8 else "Medium" if pred > 0.6 else "Low"
    
    if pred > 0.7:
        pattern_type = "Apical/Upper lobe predominant"
        findings = [
            "Upper lobe infiltrates",
            "Possible cavitation",
            "Fibrotic changes"
        ]
        affected_regions = [
            {'name': 'right_upper_lung', 'activation': round(random.uniform(0.7, 0.9), 2), 'extent': round(random.uniform(0.4, 0.6), 2)},
            {'name': 'left_upper_lung', 'activation': round(random.uniform(0.6, 0.8), 2), 'extent': round(random.uniform(0.3, 0.5), 2)}
        ]
        notes = ["Pattern consistent with post-primary TB", "Consider sputum test for confirmation"]
    
    elif pred > 0.5:
        pattern_type = "Diffuse involvement"
        findings = [
            "Bilateral infiltrates",
            "Small nodular opacities",
            "Hilar prominence"
        ]
        affected_regions = [
            {'name': 'right_upper_lung', 'activation': round(random.uniform(0.5, 0.7), 2), 'extent': round(random.uniform(0.3, 0.5), 2)},
            {'name': 'left_upper_lung', 'activation': round(random.uniform(0.5, 0.7), 2), 'extent': round(random.uniform(0.3, 0.5), 2)},
            {'name': 'right_middle_lung', 'activation': round(random.uniform(0.5, 0.7), 2), 'extent': round(random.uniform(0.2, 0.4), 2)}
        ]
        notes = ["Consider miliary TB in differential", "Recommend follow-up CT scan"]
    
    else:
        pattern_type = "Non-specific"
        findings = [
            "No significant abnormalities",
            "Normal lung parenchyma",
            "Clear costophrenic angles"
        ]
        affected_regions = [
            {'name': 'right_lung', 'activation': round(random.uniform(0.1, 0.3), 2), 'extent': round(random.uniform(0.1, 0.2), 2)},
        ]
        notes = ["Normal chest radiograph", "Consider clinical correlation"]
    
    return {
        'main_findings': findings,
        'affected_regions': affected_regions,
        'pattern_type': pattern_type,
        'confidence_level': confidence_level,
        'notes': notes
    }

def display_clinical_findings(clinical_findings, pred_score):
    """Display clinical findings in a structured format for healthcare professionals."""
    # Display prediction score
    if pred_score > 0.5:
        st.error(f"⚠️ TB Detected with {pred_score:.2%} confidence")
    else:
        st.success(f"✓ Normal X-ray with {1-pred_score:.2%} confidence")
    
    # Create columns for different aspects of the findings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confidence & Pattern")
        st.info(f"**Confidence Level:** {clinical_findings['confidence_level']}")
        st.info(f"**Pattern Type:** {clinical_findings['pattern_type'] or 'Non-specific'}")
        
        if clinical_findings['notes']:
            st.subheader("Clinical Notes")
            for note in clinical_findings['notes']:
                st.write(f"• {note}")
    
    with col2:
        if clinical_findings['main_findings']:
            st.subheader("Main Radiological Findings")
            for idx, finding in enumerate(clinical_findings['main_findings'], 1):
                st.write(f"{idx}. {finding}")
        
        if clinical_findings['affected_regions']:
            st.subheader("Affected Regions")
            # Create a dataframe for better display
            regions_df = pd.DataFrame([
                {
                    "Region": r['name'].replace('_', ' ').title(),
                    "Activation": f"{r['activation']:.2f}",
                    "Extent (%)": f"{r['extent']*100:.1f}%"
                }
                for r in clinical_findings['affected_regions']
            ])
            st.dataframe(regions_df, hide_index=True)
    
    # Add disclaimer
    st.warning("""
    **Medical Disclaimer:** This AI analysis is provided as a decision support tool only. The findings should be correlated with clinical symptoms, examination, and other laboratory tests. This does not replace the judgment of a qualified healthcare professional.
    """)

def main():
    """Main function for the Streamlit app."""
    # Title and description
    st.title("Tuberculosis (TB) Detection from Chest X-rays")
    st.markdown("""
    This application demonstrates TB detection from chest X-rays using deep learning with ResNet50.
    The model was trained using Federated Learning to maintain privacy of medical data.
    
    Upload a chest X-ray image to get a prediction and visualization of what the model focuses on.
    
    **DEMO MODE:** This is a demonstration without the actual model. Predictions are simulated.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    ### Techkriti 2025 ML Hackathon
    **IIT Kanpur**
    
    This system uses Federated Learning to detect Tuberculosis from chest X-rays
    while preserving privacy of medical data.
    
    Features:
    - TB detection with ResNet50
    - Federated Learning across multiple hospitals
    - GradCAM visualization for explainability
    - Clinical interpretation for healthcare professionals
    - Evaluation with F1-score and recall
    """)
    
    # Settings in the sidebar
    st.sidebar.title("Settings")
    clinical_mode = st.sidebar.checkbox("Enable Clinical Mode", value=True, 
                                       help="Provides detailed clinical interpretation for healthcare professionals")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["TB Detection", "Clinical Interpretation", "Model Information", "About Federated Learning"])
    
    # Upload section (outside tabs to keep it visible)
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
    
    # Variables to store results
    image = None
    pred = None
    gradcam_fig = None
    clinical_findings = None
    
    # Process uploaded image or use default
    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        
        # Simulate prediction and gradcam
        pred = simulate_prediction()
        gradcam_fig = simulate_gradcam(image, pred)
        clinical_findings = get_clinical_findings(pred)
    else:
        # Use a default example if no file is uploaded
        st.info("Please upload a chest X-ray image or click below to use an example image.")
        
        if st.button("Use Example X-ray Image"):
            # Try to load a sample image from test directory first
            test_dir = os.path.join("data", "test")
            if os.path.exists(test_dir) and os.listdir(test_dir):
                sample_path = os.path.join(test_dir, os.listdir(test_dir)[0])
                image = Image.open(sample_path)
            else:
                # Create a simple sample image if none exists
                image = Image.new('RGB', (256, 256), color='black')
                
            # Simulate prediction and gradcam
            pred = simulate_prediction()
            gradcam_fig = simulate_gradcam(image, pred)
            clinical_findings = get_clinical_findings(pred)
    
    # Tab 1: TB Detection
    with tab1:
        st.header("TB Detection from X-ray Images")
        
        if image is not None:
            # Display the image
            st.image(image, caption="X-ray Image", width=400)
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            # Display prediction in col1
            with col1:
                st.subheader("Prediction")
                
                # Create a gauge-like visualization
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.barh([0], [pred], color='red', height=0.4)
                ax.barh([0], [1-pred], left=[pred], color='green', height=0.4)
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_yticks([])
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xlabel("Probability")
                
                # Add text
                if pred > 0.5:
                    ax.text(pred/2, 0, f"TB: {pred:.2f}", ha='center', va='center', color='white', fontweight='bold')
                    ax.text((1+pred)/2, 0, f"Normal: {1-pred:.2f}", ha='center', va='center', color='white')
                    st.pyplot(fig)
                    
                    st.error(f"TB Detected with {pred:.2%} confidence")
                    st.markdown("⚠️ **Disclaimer:** This is a demonstration only. Please consult a healthcare professional for an accurate diagnosis.")
                else:
                    ax.text(pred/2, 0, f"TB: {pred:.2f}", ha='center', va='center', color='white')
                    ax.text((1+pred)/2, 0, f"Normal: {1-pred:.2f}", ha='center', va='center', color='white', fontweight='bold')
                    st.pyplot(fig)
                    
                    st.success(f"Normal X-ray with {1-pred:.2%} confidence")
                    st.markdown("⚠️ **Disclaimer:** This is a demonstration only. Please consult a healthcare professional for an accurate diagnosis.")
            
            # Display GradCAM visualization in col2
            with col2:
                st.subheader("Model Attention Visualization")
                
                if gradcam_fig is not None:
                    st.pyplot(gradcam_fig)
                
                st.markdown("""
                **Explanation:** The heatmap shows regions that the model focuses on to make its prediction. 
                In TB detection, the model typically looks for patterns in the upper lobes of the lungs,
                which are common sites for TB infection.
                
                For detailed clinical interpretation, see the "Clinical Interpretation" tab.
                """)
        else:
            st.info("Please upload a chest X-ray image or use the example image to get started.")
    
    # Tab 2: Clinical Interpretation
    with tab2:
        st.header("Clinical Interpretation for Healthcare Professionals")
        
        if image is not None and clinical_findings is not None:
            # Display original image
            st.image(image, caption="X-ray Image", width=400)
            
            # Display clinical findings
            st.header("Clinical Analysis")
            display_clinical_findings(clinical_findings, pred)
            
            # Add reference images for comparison
            with st.expander("📸 Reference TB X-ray Images"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image("https://radiopaedia.org/uploads/radio/2019/3/1/e3ab2e5e67c4744cb28b08eb3ed9a52a_jumbo.jpeg", 
                             caption="Post-primary TB (upper lobe cavitary)", width=250)
                with col2:
                    st.image("https://radiopaedia.org/uploads/radio/2013/1/21/a3f96ecd9ec3dc60acf8d21ae6b93a_jumbo.jpg", 
                             caption="Miliary TB (diffuse pattern)", width=250)
                with col3:
                    st.image("https://radiopaedia.org/uploads/radio/2017/12/28/db8e7bfad25a60a5e2b9ed4be7f27d_jumbo.jpeg", 
                             caption="Primary TB (lower lobe focus)", width=250)
        else:
            st.info("Please upload a chest X-ray image or use the example image to view clinical interpretation.")
    
    # Tab 3: Model Information
    with tab3:
        st.header("Model Information")
        
        st.markdown("""
        ### Model Architecture
        
        This application uses a ResNet50 model pre-trained on ImageNet and fine-tuned for TB detection.
        
        **Key details:**
        - Base model: ResNet50
        - Input size: 224 × 224 pixels
        - Training approach: Transfer learning with fine-tuning
        - Federated Learning: Model trained across 3 simulated hospitals
        
        ### Performance Metrics
        
        The model is evaluated using the following metrics:
        - F1-score (primary): Harmonic mean of precision and recall
        - Recall/Sensitivity: Ability to detect actual positive cases (TB)
        - Precision: Accuracy of positive predictions
        - Accuracy: Overall correctness of predictions
        """)
        
        # Display some sample metrics (would be actual in a real app)
        metrics = {
            'f1': 0.92,
            'recall': 0.93,
            'precision': 0.91,
            'accuracy': 0.90
        }
        
        # Create a bar chart of metrics
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(metrics.keys(), metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics')
        
        for i, v in enumerate(metrics.values()):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        st.pyplot(fig)
    
    # Tab 4: About Federated Learning
    with tab4:
        st.header("About Federated Learning")
        
        st.markdown("""
        ### What is Federated Learning?
        
        Federated Learning (FL) is a machine learning approach that enables training models across multiple 
        decentralized devices or servers holding local data samples, without exchanging the data itself.
        This is particularly valuable for healthcare data, which is sensitive and private.
        
        ### How it works in this TB Detection system:
        
        1. **Local Training**: Each hospital trains the model on its local chest X-ray data
        2. **Model Aggregation**: Only model weights (not patient data) are sent to a central server
        3. **Weight Averaging**: The central server aggregates the models (FedAvg algorithm)
        4. **Model Distribution**: The improved model is sent back to all hospitals
        5. **Iteration**: This process repeats for multiple rounds until the model converges
        
        ### Benefits for TB Detection:
        
        - **Privacy Preservation**: Patient X-rays never leave their respective hospitals
        - **Regulatory Compliance**: Helps meet HIPAA and other healthcare data regulations
        - **Wider Data Access**: Model learns from diverse populations without data sharing
        - **Better Performance**: More training data leads to more robust models
        """)
        
        # Add a simple diagram of FL
        st.image("https://blog.openmined.org/content/images/2019/10/federated_learning_diagram-1.png", 
                 caption="Federated Learning Process", width=700)

if __name__ == "__main__":
    main() 
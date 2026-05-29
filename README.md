\# Brain Tumor MRI Classification with Explainable AI



This project explores the use of deep learning models for brain tumor classification using MRI scans. A custom Convolutional Neural Network (CNN) was developed from scratch and compared against a DenseNet121 transfer learning approach. Explainable AI techniques were implemented using Grad-CAM to visualize the image regions influencing model predictions.



The goal of this project is not only to achieve accurate classification performance, but also to improve model interpretability by understanding how deep learning models analyze medical images.



\---



\## Project Overview



Brain tumor detection from MRI scans is a challenging medical imaging task where interpretability is important. This project implements and compares:



\- Custom CNN architecture built from scratch

\- DenseNet121 transfer learning model

\- Image preprocessing and augmentation pipeline

\- Model evaluation using classification metrics

\- Grad-CAM explainability for visual interpretation



\---



\## Models



\### Custom CNN



A custom Convolutional Neural Network was developed from scratch using PyTorch to classify MRI scans without relying on pretrained weights. The architecture was designed to progressively extract spatial features from MRI images using convolution, activation, and pooling operations before classification.



\#### Architecture Summary



\- \*\*Input:\*\* RGB MRI image with 3 channels (224×224 resolution)

\- \*\*Convolutional Layer 1:\*\* 3 input channels → 8 output channels, 3×3 kernel

\- \*\*Convolutional Layer 2:\*\* 8 input channels → 16 output channels, 3×3 kernel

\- \*\*Convolutional Layer 3:\*\* 16 input channels → 32 output channels, 3×3 kernel

\- \*\*Activation Function:\*\* ReLU activation after each convolutional layer

\- \*\*Pooling:\*\* 2×2 max pooling after each convolutional layer

\- \*\*Flattening:\*\* Final feature maps flattened into a one-dimensional feature vector

\- \*\*Fully Connected Layer 1:\*\* 32 × 27 × 27 input features → 500 neurons

\- \*\*Fully Connected Layer 2:\*\* 500 neurons → 4 output classes



\*\*Output Classes:\*\*

\- Glioma

\- Meningioma

\- Pituitary

\- Healthy



\#### Training Configuration



\- \*\*Model:\*\* Custom CNN trained from scratch

\- \*\*Epochs:\*\* 10

\- \*\*Learning Rate:\*\* 1 × 10⁻⁴

\- \*\*Weight Decay:\*\* 1 × 10⁻⁴

\- \*\*Optimizer:\*\* Adam

\- \*\*Input Image Size:\*\* 224×224

\- \*\*Batch Size:\*\* 32

\- \*\*Loss Function:\*\* Cross Entropy Loss

\- \*\*Number of Classes:\*\* 4

\- \*\*Checkpointing:\*\* Saved model weights whenever validation loss improved

\- \*\*Hardware:\*\* NVIDIA T4 GPU using Google Colab with CUDA acceleration



\### DenseNet121 Transfer Learning



A pretrained DenseNet121 convolutional neural network was fine-tuned using transfer learning to classify MRI scans. The pretrained feature extraction layers were frozen, allowing the model to leverage learned image representations while training a custom classifier for brain tumor classification.



\#### Architecture Summary



\- \*\*Base Model:\*\* Pretrained DenseNet121

\- \*\*Transfer Learning Approach:\*\* Frozen feature extraction layers

\- \*\*Classifier Layer 1:\*\* DenseNet output features → 256 neurons

\- \*\*Activation Function:\*\* ReLU activation

\- \*\*Regularization:\*\* Dropout with probability of 0.25

\- \*\*Classifier Layer 2:\*\* 256 neurons → 4 output classes

\- \*\*Output Activation:\*\* LogSoftmax



\*\*Output Classes:\*\*

\- Glioma

\- Meningioma

\- Pituitary

\- Healthy



\#### Training Configuration



\- \*\*Model:\*\* DenseNet121 Transfer Learning

\- \*\*Epochs:\*\* 10

\- \*\*Learning Rate:\*\* 1 × 10⁻³

\- \*\*Weight Decay:\*\* 1 × 10⁻⁴

\- \*\*Optimizer:\*\* Adam

\- \*\*Input Image Size:\*\* 224×224

\- \*\*Batch Size:\*\* 32

\- \*\*Loss Function:\*\* Negative Log Likelihood Loss (NLLLoss)

\- \*\*Number of Classes:\*\* 4

\- \*\*Checkpointing:\*\* Saved model weights whenever validation loss improved

\- \*\*Hardware:\*\* NVIDIA T4 GPU using Google Colab with CUDA acceleration



\---

\## Evaluation Metrics



Model performance was evaluated using accuracy, precision, recall, and F1-score for each brain tumor class.



\- \*\*Accuracy:\*\* Measures the overall percentage of correctly classified MRI scans.

\- \*\*Precision:\*\* Measures how many predictions for a specific class were actually correct.

\- \*\*Recall:\*\* Measures how many actual samples of a specific class were correctly identified by the model.

\- \*\*F1-Score:\*\* Represents the balance between precision and recall.



These metrics were used to compare both models across four classes: Glioma, Meningioma, Pituitary, and Healthy.



\---



\## Explainable AI (Grad-CAM)



Grad-CAM was implemented to improve model transparency by visualizing the MRI regions that contributed most to each prediction.



The process involves:



1\. Computing gradients of the predicted class with respect to convolutional feature maps

2\. Applying global average pooling to determine feature map importance

3\. Creating a weighted combination of feature maps

4\. Overlaying the generated heatmap onto the original MRI image



These visual explanations help verify that the models focus on meaningful tumor regions rather than unrelated image features.



\---



\## Grad-CAM Visualization Results



\### Example 1



| Original MRI | Grad-CAM Heatmap |

|---|---|

| !\[MRI](assets/xai2.1.png) | !\[GradCAM](assets/xai2.2.png) |



\### Example 2



| Original MRI | Grad-CAM Heatmap |

|---|---|

| !\[MRI](assets/xai3.1.png) | !\[GradCAM](assets/xai3.2.png) |



\### Example 3



| Original MRI | Grad-CAM Heatmap |

|---|---|

| !\[MRI](assets/xai4.png) | !\[GradCAM](assets/xai4.2.png) |



\---



\# Results



DenseNet121 transfer learning improved overall classification performance compared to the custom CNN, increasing test accuracy from \*\*79% to 88%\*\* and weighted F1-score from \*\*0.79 to 0.89\*\*.



\## Confusion Matrices



| Custom CNN | DenseNet121 Transfer Learning |

|---|---|

| !\[Custom CNN Confusion Matrix](assets/scratchCM.png) | !\[DenseNet121 Confusion Matrix](assets/transCM.png) |



\---



\## Custom CNN Performance



| Class | Precision | Recall | F1-Score |

|---|---|---|---|

| Glioma | 0.73 | 0.83 | 0.78 |

| Healthy | 0.94 | 0.85 | 0.89 |

| Meningioma | 0.64 | 0.53 | 0.58 |

| Pituitary | 0.83 | 0.92 | 0.87 |

| \*\*Weighted Avg\*\* | \*\*0.80\*\* | \*\*0.80\*\* | \*\*0.79\*\* |



\---



\## DenseNet121 Performance



| Class | Precision | Recall | F1-Score |

|---|---|---|---|

| Glioma | 0.88 | 0.87 | 0.88 |

| Healthy | 0.96 | 0.96 | 0.96 |

| Meningioma | 0.75 | 0.79 | 0.77 |

| Pituitary | 0.93 | 0.90 | 0.92 |

| \*\*Weighted Avg\*\* | \*\*0.89\*\* | \*\*0.89\*\* | \*\*0.89\*\* |



\---



\## Performance Analysis



The custom CNN performed well overall but struggled with the \*\*meningioma tumor class\*\*, achieving an F1-score of only \*\*0.58\*\*. The recall score of \*\*0.53\*\* shows that many true meningioma MRI scans were incorrectly classified as another category.



DenseNet121 transfer learning significantly improved meningioma classification:



\- Precision: \*\*0.64 → 0.75\*\*

\- Recall: \*\*0.53 → 0.79\*\*

\- F1-Score: \*\*0.58 → 0.77\*\*



The Healthy class also showed an important improvement. Precision increased from \*\*0.94 to 0.96\*\*, meaning that when the model predicted an MRI scan as healthy, it was correct 96% of the time. This reduced cases where tumor scans were incorrectly classified as healthy.



Healthy recall also improved from \*\*0.85 to 0.96\*\*, showing that the model became better at identifying true healthy MRI scans.



Overall, DenseNet121 demonstrated the benefits of transfer learning by improving feature extraction, classification accuracy, and reliability across MRI categories.



\---



\## Technologies Used



\- Python

\- PyTorch

\- TorchVision

\- DenseNet121

\- Grad-CAM

\- NumPy

\- Pandas

\- Matplotlib

\- Scikit-learn

\- Jupyter Notebook

\- LaTeX



\---



\## Future Improvements



Future work includes:



\- Exploring additional Explainable AI techniques such as Integrated Gradients and Score-CAM

\- Improving tumor localization with segmentation-based approaches

\- Testing additional architectures such as Vision Transformers

\- Expanding evaluation using larger medical imaging datasets



\---



\## Author



\*\*Brendan Lauterborn\*\*



MS Computer Science  

Deep Learning / Computer Vision Projectg / Computer Vision Project


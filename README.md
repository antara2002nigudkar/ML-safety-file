### Exercise 3 — Training and Evaluation of the Three Perception Models

[📓 Exercise 3](./Exercise%203%20ML_Safety.ipynb)

In Exercise 3, three separate image-classification models were trained using the CARLA perception dataset:

* **Pedestrian model**
* **Traffic-light model**
* **Vehicle model**

Each model was trained for **3 epochs** using PyTorch.

The exercise included evaluating the trained models using several standard classification metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-score**

These metrics were used to evaluate not only the overall number of correct predictions, but also how well each model detected its target class.

Recall was particularly important from a safety perspective because a **false negative**, such as failing to detect a pedestrian, can have significantly greater consequences than a false positive.

The exercise therefore established the baseline performance of the three perception models before carrying out the additional safety analyses in later exercises.

---

### Exercise 4 — Data Exploration and Model Evaluation

[📓 Exercise 4](./Exercise%204%20ML_Safety.ipynb)

Exercise 4 focused on **exploring and understanding the dataset** before using the models for further safety evaluation.

The data exploration included examining:

* Dataset structure
* Image samples
* Image dimensions
* Class labels
* Label distributions
* Training and validation data
* Dataset characteristics relevant to model performance

The exercise also evaluated the trained models using **confusion matrices**.

A separate confusion matrix was generated for each of the three models:

* Pedestrian
* Traffic light
* Vehicle

Confusion matrices were used to examine the types of classification errors made by each model rather than relying only on a single overall accuracy value.

This provided a clearer understanding of:

* Correct classifications
* False positives
* False negatives
* Class-specific performance
* Which types of errors may be safety-relevant

The data exploration and confusion-matrix analysis provided the basis for understanding the model behavior before applying further calibration and safety techniques.

---

### Exercise 5 — Temperature Scaling and Model Calibration

[📓 Exercise 5](./Exercise%205%20ML_Safety.ipynb)

Exercise 5 investigated **model confidence and calibration**.

Although a model can achieve good classification performance, its predicted probabilities may not accurately represent the true likelihood of being correct. This is particularly important in safety-critical systems where downstream decisions may depend on model confidence.

#### Temperature Scaling

Temperature scaling was applied to the model logits:

[
p(y|x)=\operatorname{softmax}\left(\frac{f(x)}{T}\right)
]

The temperature (T) was optimized on the validation set using negative log-likelihood (NLL).

A grid search was used to evaluate different temperature values and select the temperature producing the lowest validation NLL.

The resulting temperatures were:

| Model      | Best T |
| ---------- | -----: |
| Pedestrian |    3.0 |
| Traffic    |    0.5 |
| Vehicle    |    1.4 |

Temperature scaling was then used to produce calibrated model probabilities for the later safety analysis.

---

### Exercise 6 — Grad-CAM Explainability

[📓 Exercise 6](./ML_Safety_Ex_6.ipynb)

[📓 Exercise 6 — Continued](./ML_Safety_Ex_6%28continued%29.ipynb)

Exercise 6 focused on **model interpretability using Grad-CAM**.

**Grad-CAM (Gradient-weighted Class Activation Mapping)** was used to visualize which regions of an input image contributed most strongly to the model's prediction.

Grad-CAM heatmaps were generated for the perception models to investigate whether the models were focusing on meaningful image regions.

The analysis compared model attention across different examples, including correctly and incorrectly classified images.

This was useful for identifying possible:

* Relevant object-focused behavior
* Background dependence
* Spurious correlations
* Model failure modes
* Changes in model attention under different conditions

For example, when the model correctly identifies a pedestrian or vehicle, the Grad-CAM visualization can show whether the model is actually focusing on the relevant object rather than unrelated background features.

Grad-CAM therefore provides an additional qualitative form of evidence alongside the quantitative metrics such as accuracy, precision, recall, F1-score and confusion matrices.

Exercise 7 focuses on the calibration of the three perception models:

Pedestrian
Traffic
Vehicle
Temperature Scaling

Temperature scaling was applied to model logits:

[
p(y|x)=\operatorname{softmax}\left(\frac{f(x)}{T}\right)
]

A grid search over temperature values was used to minimize validation negative log-likelihood.

The selected temperatures were:

Model	Best T
Pedestrian	3.0
Traffic	0.5
Vehicle	1.4
Expected Calibration Error
Model	ECE Before	ECE After
Pedestrian	0.1551	0.0743
Traffic	0.2370	0.2203
Vehicle	0.0538	0.0018

Temperature scaling reduced ECE for all three models in the reported experiments.

Reliability Diagrams

Reliability diagrams were used to visually compare model confidence with observed accuracy.

These diagrams complement ECE by showing where the model is overconfident or underconfident across different confidence ranges.

Cost-Optimal Decision Making

For pedestrian detection, the safety costs were:

(C_{FN}=100)
(C_{FP}=1)
(\tau^*\approx0.0099)

The total loss was calculated as:

[
L=C_{FN}#FN+C_{FP}#FP
]

The results were:

Configuration	Threshold	Total Loss
Uncalibrated	0.5	72,718
Uncalibrated	0.0099	15,156
Calibrated	0.5	72,718
Calibrated	0.0099	2,686

The calibrated model with (\tau^*=0.0099) produced the lowest total loss.

This demonstrates why safety-critical classification thresholds should consider the relative costs of false negatives and false positives rather than simply using 0.5.

### Exercise 8 — Adversarial Robustness with FGSM

Exercise 8 investigated the robustness of the three perception models against **adversarial examples** using the **Fast Gradient Sign Method (FGSM)**.

The models were evaluated using different perturbation strengths:

* (\epsilon = 0.01)
* (\epsilon = 0.05)
* (\epsilon = 0.10)

The purpose was to investigate how adversarial perturbations affect the ability of the models to correctly detect their target classes.

The main metric considered was **recall**, both on clean inputs and adversarially perturbed inputs.

The reported results were:

| Model         | Recall (Clean) | Recall (Adversarial) |
| ------------- | -------------: | -------------------: |
| Pedestrian    |         0.1914 |               0.8948 |
| Traffic light |         0.5166 |               0.4398 |
| Vehicle       |         0.7115 |               0.6717 |

The results show that the effect of the adversarial perturbation differs between models. The traffic-light and vehicle models show a reduction in recall under the reported adversarial evaluation, while the pedestrian result shows a different behavior.

These results were used to investigate the robustness of the perception models and demonstrate that adversarial evaluation is an important part of ML safety assessment.

---

### Exercise 9 — Out-of-Distribution Detection

Exercise 9 investigated the ability of the perception system to identify **out-of-distribution (OOD)** inputs.

The evaluation included images from conditions such as:

* **Night**
* Fog
* **Different towns / unseen environments**

These conditions were used to investigate whether the models could recognize when an input differs from the conditions represented by the normal operating data.

Two OOD detection approaches were investigated:

#### Maximum Softmax Probability (MSP)

The **Maximum Softmax Probability (MSP)** was calculated from the model's output probabilities.

MSP measures the highest predicted class probability and can be used as a simple confidence-based indicator for identifying potentially unfamiliar inputs.

#### K-Nearest Neighbors (KNN)

A **KNN-based OOD detection approach** was also evaluated.

Feature representations from the model were used to compare samples based on their proximity to known data. Samples that are distant from the known data distribution can provide an indication that an input may be out of distribution.

#### AUROC

The performance of both OOD detection approaches was evaluated using **Area Under the Receiver Operating Characteristic Curve (AUROC)**.

The comparison between MSP and KNN provides evidence about how effectively the methods distinguish familiar inputs from OOD inputs.

The Exercise 9 experiments therefore investigate whether an ML system can detect when its perception model is operating outside the conditions for which it has reliable evidence.

This is important for safety because an OOD detector can potentially trigger a safer fallback behavior when the perception model encounters unfamiliar conditions such as nighttime scenes or an unseen town.

---

## Summary of Exercises 8 and 9

| Exercise       | Main focus                        |
| -------------- | --------------------------------- |
| **Exercise 8** | FGSM adversarial robustness       |
| **Exercise 9** | OOD detection                     |
|                |Fog, Night and different-town images   
|                | Maximum Softmax Probability (MSP) |
|                | K-Nearest Neighbors (KNN)         |
|                | AUROC evaluation                  |

Together, these exercises extend the safety evaluation beyond normal classification performance by investigating **adversarial robustness** and the ability to recognize **out-of-distribution inputs**.


The exercise also connected interpretability to **machine-learning safety**, since understanding what a model relies on can help identify potential weaknesses that may not be visible from performance metrics alone.

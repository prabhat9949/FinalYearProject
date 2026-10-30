# Complete Guide to Machine Learning Models for Liver Disease Screening (Clinical + Ultrasound Models)

## Table of Contents
1. [Overview of All Three Models](#overview-of-all-three-models)
2. [Model 1: Logistic Regression](#model-1-logistic-regression)
3. [Model 2: Random Forest](#model-2-random-forest)
4. [Model 3: XGBoost](#model-3-xgboost)
5. [Model Comparison](#model-comparison)
6. [Which Model to Choose](#which-model-to-choose)
7. [Real-World Clinical Examples](#clinical-examples)
8. [Backend Architecture - Complete System Design](#backend-architecture---complete-system-design)
9. [Byra Ultrasound Vision Transformer (ViT-B/16)](#model-4-byra-ultrasound-vision-transformer-vit-b16)
10. [BEHSOF Ultrasound Vision Transformer (ViT-B/16)](#model-5-behsof-ultrasound-vision-transformer-vit-b16)
11. [Metric Definitions and Combined Performance Summary](#metric-definitions-precision-recall-f1-specificity-and-auc)
12. [Multimodal Evidence Aggregation](#multimodal-evidence-aggregation-how-the-model-outputs-are-combined)

---

# Overview of All Three Models

We are training **three different machine learning models** on structured clinical data (blood tests + demographics) to predict liver disease. Each model has different strengths, weaknesses, and complexity levels.

> **Note:** The project has since been extended with two ultrasound image models (Byra and BEHSOF ViT-B/16). They are documented in sections 9–12 below and should not be compared head-to-head with these three clinical models.

Think of them like:

```
Logistic Regression = Simple rule-based doctor
Random Forest = Team of experienced doctors voting
XGBoost = Super-specialized doctor with decades of experience
```

| Model | Complexity | Speed | Interpretability | Accuracy |
|-------|-----------|-------|------------------|----------|
| Logistic Regression | ⭐ Very Simple | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| Random Forest | ⭐⭐⭐ Medium | ⭐⭐⭐ Medium | ⭐⭐⭐ Fair | ⭐⭐⭐⭐ Very Good |
| XGBoost | ⭐⭐⭐⭐⭐ Very Complex | ⭐⭐ Slower | ⭐⭐ Poor | ⭐⭐⭐⭐⭐ Excellent |

---

# Model 1: Logistic Regression

## What is Logistic Regression?

**Logistic Regression** is the **simplest classification model** — it's like asking a doctor to draw a straight line to separate "healthy" from "diseased" patients.

### Real-World Analogy

Imagine you're a doctor looking at blood test results:

```
Simple Rule: "If bilirubin > 1.5, patient likely has disease"
```

Logistic Regression learns a **linear rule** like this automatically from data.

---

## How Does It Work? (Step by Step)

### **Step 1: Start with Features**

Our input: **10 biochemical markers**
```
Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
Albumin, Albumin_and_Globulin_Ratio
```

### **Step 2: Assign Weights to Each Feature**

The model learns: "How much does each feature matter?"

```
Weight (Age) = 0.03              # Small importance
Weight (AST) = 0.45              # High importance
Weight (Bilirubin) = 0.52         # Very high importance
Weight (Albumin) = -0.38          # Negative (high albumin = healthy)
... (10 total weights)
```

### **Step 3: Calculate Disease Risk Score**

For each patient, combine all features with their weights:

```
Disease Risk Score = 
    0.03 × Age + 
    0.45 × AST + 
    0.52 × Bilirubin + 
    (-0.38) × Albumin + 
    ... (more terms)
```

### **Step 4: Convert Score to Probability**

Logistic Regression uses a special function (sigmoid) to convert any score into a probability between 0 and 1:

```
Probability = 1 / (1 + e^(-Risk Score))

If Risk Score = 0     → Probability = 0.5 (50% chance of disease)
If Risk Score = +2    → Probability = 0.88 (88% chance of disease)
If Risk Score = -2    → Probability = 0.12 (12% chance of disease)
```

### **Step 5: Make Prediction**

```
If Probability > 0.5  → Predict: Diseased
If Probability ≤ 0.5  → Predict: Healthy
```

---

## Visual Example: How Logistic Regression Separates Classes

```
Bilirubin Level
     ^
  5  |  ✗ Diseased
     |  ✗✗  ✗
  4  | ✗✗ ╱─ Decision Boundary (0.5 probability)
     |  ✗╱
  3  | ╱  ○ Healthy
     |╱    ○ ○
  2  |     ○ ○
     |      ○
  1  |  ○    ○○
     |___________________> AST Level
     0  1   2   3   4   5
```

**Logistic Regression draws a STRAIGHT LINE** to separate the two classes.

---

## Advantages of Logistic Regression

1. **✅ Highly Interpretable**
   - Doctors can understand WHY it made a prediction
   - "Patient has high disease risk because bilirubin is 5.2 and AST is elevated"
   - Each coefficient (weight) has a clear meaning

2. **✅ Very Fast**
   - Trains in milliseconds
   - Can predict on new patients instantly
   - Good for real-time screening

3. **✅ Computationally Lightweight**
   - Requires minimal CPU/memory
   - Can run on simple hardware
   - Perfect for mobile/embedded systems

4. **✅ Requires Less Data**
   - Works well even with small datasets
   - Fewer hyperparameters to tune

5. **✅ No Overfitting Risk**
   - Simple model, less likely to memorize noise
   - Generalizes well to new patients

---

## Disadvantages of Logistic Regression

1. **❌ Only Captures Linear Patterns**
   - Real diseases are complex (non-linear relationships)
   - Example: Low bilirubin + high albumin = healthy
   - High bilirubin + low albumin = diseased
   - LR might miss this interaction

2. **❌ Assumes Feature Independence**
   - Doesn't understand interactions between features
   - Doesn't realize "AST and ALT together matter more"

3. **❌ Poor Performance on Complex Data**
   - Lower accuracy than advanced models
   - For liver disease screening: ~78-82% recall (our testing showed 65% on imbalanced data)

4. **❌ Struggles with Multicollinearity**
   - If features are correlated (like AST and ALT), gets confused
   - Weights become unstable

---

## Mathematical Formula (For Reference)

```
Logistic Regression Probability:

P(Disease = 1) = 1 / (1 + e^(-z))

where z = β₀ + β₁×X₁ + β₂×X₂ + ... + β₁₀×X₁₀

β₀ = intercept (baseline risk)
β₁, β₂, ... β₁₀ = feature weights (learned from data)
X₁, X₂, ... X₁₀ = feature values (your 10 biochemical markers)
e = Euler's number (2.718...)
```

---

## Expected Performance on Our Data

```
Logistic Regression Results (with 50-50 balanced training data):

Recall:        85-87%     ✓ Good (catches most diseases)
Precision:     79-82%     ✓ Acceptable (low false alarms)
F1-Score:      0.83-0.85  ✓ Good balance
ROC-AUC:       0.88-0.90  ✓ Good discrimination
Training Time: < 1 second ⭐ Very fast
```

---

---

# Model 2: Random Forest

## What is Random Forest?

**Random Forest** is like asking **100 different doctors** to vote on each patient's diagnosis. Each doctor sees a different view of the data and makes their own decision. The final diagnosis is determined by majority vote.

### Real-World Analogy

Imagine you bring a patient to:
- **Doctor A** (sees bilirubin and albumin) → Says "Diseased"
- **Doctor B** (sees AST and ALT) → Says "Diseased"
- **Doctor C** (sees age and gender) → Says "Healthy"
- **Doctor D** (sees alkaline phosphatase) → Says "Diseased"

**Vote: 3 out of 4 say diseased → Final diagnosis: DISEASED ✓**

This is how Random Forest works!

---

## How Does It Work? (Step by Step)

### **Step 1: Create Multiple Decision Trees**

The model builds **100-300 decision trees** (you specify the number). Each tree is a different doctor seeing different features.

### **Step 2: Each Tree Gets Random Subset of Data**

For example:

```
Tree 1: Sees 250 random patients (with replacement)
Tree 2: Sees 250 different random patients
Tree 3: Sees another 250 random patients
... (100 trees total)
```

This randomness helps trees learn different patterns!

### **Step 3: Each Tree Gets Random Features**

Each tree also sees only **some features**:

```
Tree 1: Sees [Age, Bilirubin, AST, Albumin]
Tree 2: Sees [Gender, Alkaline_Phosphotase, ALT, Albumin_Ratio]
Tree 3: Sees [Age, Bilirubin, ALT, Total_Proteins]
... (each sees random subset)
```

Again, this diversity prevents overfitting!

### **Step 4: Each Tree Makes a Decision**

Each decision tree is like a flowchart:

```
Tree 1:
        Is Bilirubin > 1.5?
       /                    \
      YES                    NO
      /                        \
   Is AST > 40?            Is Albumin > 3.0?
   /         \              /          \
 YES         NO          YES           NO
  |           |           |            |
Disease   Healthy     Healthy       Disease
```

For a patient, the tree asks a series of yes/no questions and ends at a prediction.

### **Step 5: Ensemble Voting**

All trees vote:

```
Patient X blood test:
├─ Tree 1: Diseased (66% confidence)
├─ Tree 2: Diseased (72% confidence)
├─ Tree 3: Healthy (40% confidence)
├─ Tree 4: Diseased (81% confidence)
└─ ... (96 more trees)

Final prediction: DISEASED (88% of trees agree)
Final probability: 0.88 (average confidence)
```

---

## Visual Example: Decision Tree Inside Random Forest

```
                Root (All 333 samples)
                   |
          Is Total_Bilirubin > 1.0?
         /                        \
        YES (245)                 NO (88)
        /                            \
  Is AST > 40?                  Is Albumin > 3.5?
  /      \                       /        \
YES(180) NO(65)             YES(60)      NO(28)
 /        \                  /            \
🔴 D      🟢 H          🟢 H           🔴 D
(Disease) (Healthy)

🔴 = Diseased class
🟢 = Healthy class
```

Each path from top to bottom = one prediction rule!

---

## Why Multiple Trees? (The Wisdom of the Crowd)

One doctor can be wrong. But if 100 doctors vote, mistakes cancel out:

```
Scenario: Real patient is Diseased

Doctor A (saw limited data): Says Healthy ✗
Doctor B: Says Diseased ✓
Doctor C: Says Diseased ✓
Doctor D: Says Diseased ✓
... (96 more)

Vote: 95/100 say Diseased
Final: Diseased ✓ Correct!
```

---

## Advantages of Random Forest

1. **✅ Handles Non-Linear Patterns**
   - Can learn complex interactions
   - Example: "Low albumin + high bilirubin = disease" 
   - LR would struggle, RF captures this easily

2. **✅ Robust to Outliers**
   - One extreme value doesn't break the model
   - Each tree is built on random subset of data

3. **✅ Feature Importance Available**
   - Tells you which features matter most
   - "Bilirubin is 2.5x more important than age"

4. **✅ Less Hyperparameter Tuning**
   - Works well with default settings
   - Forgiving to tuning choices

5. **✅ Better Accuracy Than LR**
   - Expected recall: 85-88%
   - Expected precision: 79-82%

6. **✅ Handles Mixed Feature Types**
   - Works with numeric, categorical, or mixed data

---

## Disadvantages of Random Forest

1. **❌ Less Interpretable**
   - Can't easily explain why a specific prediction was made
   - "Why did it predict diseased?" → "100 trees voted"
   - Hard to show to doctors

2. **❌ Slower Than Logistic Regression**
   - Needs to ask 100 trees (not just a math equation)
   - Training: 5-30 seconds vs. < 1 second for LR
   - Prediction: milliseconds but still 100× slower

3. **❌ Risk of Overfitting**
   - If trees are too deep, can memorize training data
   - Needs careful hyperparameter tuning

4. **❌ Memory Intensive**
   - Stores 100+ trees in memory
   - Requires more storage/computing power than LR

5. **❌ Hyperparameter Sensitivity**
   - Need to tune: n_estimators, max_depth, min_samples_split
   - Too many trees = overfitting, too few = underfitting

---

## Key Hyperparameters

```python
RandomForestClassifier(
    n_estimators=200,          # Number of trees (100-500 typical)
    max_depth=10,              # Max tree depth (prevents overfitting)
    min_samples_split=5,       # Min samples to split node
    min_samples_leaf=2,        # Min samples in leaf node
    random_state=42            # For reproducibility
)
```

**What they mean:**
- **n_estimators**: More trees = better but slower
- **max_depth**: Deeper trees = more complex but risk overfitting
- **min_samples_split**: Higher value = simpler trees, less overfitting

---

## Expected Performance on Our Data

```
Random Forest Results (with 50-50 balanced training data):

Recall:        85-87%     ✓ Good
Precision:     79-81%     ✓ Acceptable
F1-Score:      0.83-0.84  ✓ Good
ROC-AUC:       0.87-0.89  ✓ Good discrimination
Training Time: 5-15 seconds ⚠️ Moderate
```

---

---

# Model 3: XGBoost

## What is XGBoost?

**XGBoost** (Extreme Gradient Boosting) is the **most advanced model**. Instead of voting independently, it learns from previous mistakes and improves incrementally.

### Real-World Analogy

Imagine training a doctor:

```
Day 1: Doctor makes 100 diagnoses, gets 20 wrong
       Review: "You missed patients with high ALT"

Day 2: Doctor focuses extra on ALT patterns
       Makes 5 new mistakes
       Review: "You were wrong on cases with low albumin"

Day 3: Doctor focuses on albumin + ALT combinations
       Makes 2 new mistakes
       ... keeps improving

After 200 days: Doctor is excellent (90%+ accuracy)
```

This is XGBoost! It **improves through iterations**, learning from mistakes.

---

## How Does It Work? (Step by Step)

### **Step 1: Start Simple**

The first tree is very simple (just a few rules):

```
Tree 1: Just asks "Is Bilirubin > 1.5?"
        Some patients correctly classified ✓
        Some patients misclassified ✗
```

### **Step 2: Calculate Errors (Residuals)**

The model calculates how wrong it was:

```
Patient A: Predicted Healthy, Actually Diseased → Error = +1 (missed disease)
Patient B: Predicted Diseased, Actually Healthy → Error = -1 (false alarm)
Patient C: Predicted Healthy, Actually Healthy  → Error = 0 (correct)
```

### **Step 3: Build Second Tree to Fix Errors**

Instead of independent voting (like Random Forest), the **second tree focuses on the ERRORS from Tree 1**:

```
Tree 2 specializes in: "Cases where Tree 1 was wrong"
       Asked specifically about low albumin + high ALT
       Fixes many of Tree 1's mistakes
```

### **Step 4: Repeat Process**

Builds 100-500 trees, each learning from previous trees' mistakes:

```
Tree 1:   Gets 80% of patients right
          ↓ Tree 2 focuses on remaining 20%
Tree 2:   Improves to 85% overall
          ↓ Tree 3 focuses on remaining 15%
Tree 3:   Improves to 88% overall
          ↓ ... continues
Tree 100: Final accuracy: 91%+
```

### **Step 5: Final Prediction = Sum of All Trees**

Unlike Random Forest (majority voting), XGBoost **sums predictions from all trees**:

```
Tree 1 prediction: +0.45 (leans toward disease)
Tree 2 prediction: +0.20
Tree 3 prediction: +0.15
... (97 more)
________________
Total: +3.50

Final probability = sigmoid(+3.50) = 0.97 = 97% chance of disease
Prediction: DISEASED
```

---

## Visual Example: Boosting Process

```
Initial Data (all equally important):
  ✓ ✓ ✗ ✓ ✗ ✓ ✗ ✓ ...
  Correct predictions ✓
  Misclassifications ✗

After Tree 1 (80% accuracy):
  ✓ ✓ ↑ ✓ ↑ ✓ ↑ ✓ ...
  Missed cases get HIGHER WEIGHT ↑↑↑

Tree 2 focuses on HIGH-WEIGHT cases:
  (More samples from the ✗ group)
  Learns patterns from mistakes

After Tree 2 (85% accuracy):
  ✓ ✓ ✓ ✓ ✗ ✓ ✗ ✓ ...
  Fewer mistakes, but some remain

Tree 3 focuses on NEW high-weight cases:
  ... continues until convergence
```

---

## Why XGBoost is So Powerful

**Key Insight:** Learning from mistakes is more efficient than voting!

```
Random Forest: "Ask 100 independent doctors"
XGBoost: "Train 1 doctor for 100 days, improving daily"

Random Forest: Gets diverse views (good)
XGBoost: Gets focused improvement on hard cases (better)
```

---

## Advantages of XGBoost

1. **✅ Highest Accuracy**
   - Expected recall: 87-90%
   - Expected precision: 80-84%
   - Best performance among the three models

2. **✅ Handles Complex Patterns**
   - Captures non-linear relationships
   - Learns feature interactions automatically
   - Better than Random Forest on complex data

3. **✅ Built-in Regularization**
   - Prevents overfitting through multiple mechanisms
   - learning_rate, max_depth, subsample all control complexity

4. **✅ Feature Importance**
   - Provides detailed feature importance
   - Shows which markers matter most

5. **✅ Handles Missing Values**
   - Automatically learns how to handle missing data
   - No need for imputation

6. **✅ Faster Than Random Forest**
   - Despite complexity, often trains faster
   - Focuses computation on important patterns

7. **✅ Industry Standard**
   - Used in 80%+ of Kaggle competitions
   - Proven in millions of production systems

---

## Disadvantages of XGBoost

1. **❌ Least Interpretable**
   - 300+ boosted trees stacked together
   - Hard to explain specific predictions
   - Black box for clinical interpretation

2. **❌ Complex Hyperparameters**
   - Many parameters to tune: learning_rate, max_depth, n_estimators, subsample, colsample_bytree, gamma, lambda
   - Requires experience to tune well

3. **❌ Risk of Overfitting**
   - If learning_rate is too high or too many trees
   - Needs careful cross-validation

4. **❌ Slower Training**
   - 30-60 seconds vs. 5-15 seconds for RF
   - But still acceptable for screening

5. **❌ Requires Careful Tuning**
   - Default settings often not optimal
   - Need GridSearchCV to find best parameters

6. **❌ Overkill for Simple Problems**
   - Uses advanced techniques for something LR could handle
   - Like using a fighter jet to deliver pizza!

---

## Key Hyperparameters

```python
XGBClassifier(
    n_estimators=200,         # Number of boosting rounds
    learning_rate=0.1,        # Step size (0.01-0.3 typical)
    max_depth=5,              # Tree depth (3-8 typical)
    subsample=0.8,            # Fraction of samples per tree
    colsample_bytree=1.0,     # Fraction of features per tree
    scale_pos_weight=2,       # Weight for positive class (for imbalance)
    random_state=42
)
```

**What they mean:**
- **learning_rate**: Lower = slower improvement but better generalization
- **max_depth**: Lower = simpler trees, less overfitting
- **subsample**: Lower = less data per tree, more regularization
- **scale_pos_weight**: Higher = emphasize catching diseases more

---

## Expected Performance on Our Data

```
XGBoost Results (with 50-50 balanced training data):

Recall:        88-91%     ✅ Excellent (above target)
Precision:     81-84%     ✓ Good
F1-Score:      0.85-0.87  ✅ Excellent
ROC-AUC:       0.90-0.92  ✅ Excellent discrimination
Training Time: 30-60 seconds ⚠️ Slower but acceptable
```

---

---

# Model Comparison

## Side-by-Side Comparison Table

| Aspect | Logistic Regression | Random Forest | XGBoost |
|--------|---|---|---|
| **How It Works** | Linear boundary | Voting ensemble | Sequential boosting |
| **Complexity** | ⭐ Simple | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Complex |
| **Interpretability** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Fair | ⭐⭐ Poor |
| **Training Speed** | ⭐⭐⭐⭐⭐ <1 sec | ⭐⭐⭐ 5-15 sec | ⭐⭐ 30-60 sec |
| **Prediction Speed** | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐ Medium | ⭐⭐⭐ Medium |
| **Expected Recall** | 85-87% | 85-87% | 88-91% |
| **Expected Precision** | 79-82% | 79-81% | 81-84% |
| **Expected ROC-AUC** | 0.88-0.90 | 0.87-0.89 | 0.90-0.92 |
| **Non-Linear Patterns** | ❌ No | ✅ Yes | ✅ Yes |
| **Feature Interactions** | ❌ No | ✅ Yes | ✅ Yes |
| **Robustness to Outliers** | ❌ No | ✅ Yes | ✅ Yes |
| **Hyperparameter Tuning** | Easy | Medium | Hard |
| **Risk of Overfitting** | Low | Medium | High (if not careful) |
| **Clinical Trust** | ✅ High (interpretable) | ⚠️ Medium | ⚠️ Low (black box) |
| **Production Ready** | ✅ Yes | ✅ Yes | ✅✅ Yes (most used) |

---

## Performance Comparison Chart

```
Recall Comparison (on 50-50 balanced test data):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Logistic Regression:  ████████████████████░░░░  85-87%
Random Forest:        ████████████████████░░░░  85-87%
XGBoost:              ████████████████████░░░░  88-91%  ← Winner

Target (≥85%):        ████████████████████
```

```
Precision Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Logistic Regression:  ██████████████████░░░░░░  79-82%
Random Forest:        ██████████████████░░░░░░  79-81%
XGBoost:              █████████████████░░░░░░░  81-84%

Target (≥70%):        ████████████████████
```

```
Speed Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Logistic Regression:  █████████████████████████  <1 second  ⭐ Fastest
Random Forest:        ███████████████░░░░░░░░░░  10 seconds
XGBoost:              ██████████░░░░░░░░░░░░░░░  45 seconds
```

---

## Real Clinical Scenario: Which Model to Choose?

### **Scenario 1: Rural Clinic with Limited Computing Power**

```
Requirement: Fast, runs on basic hardware, interpretable

Solution: ✅ Logistic Regression
Why:
  • Works on any computer (even smartphones)
  • Trains in <1 second
  • Doctors can understand coefficients
  • 85% recall still meets clinical targets
```

### **Scenario 2: Hospital with Modern Infrastructure**

```
Requirement: Best accuracy, some interpretability, fast enough

Solution: ✅ Random Forest
Why:
  • 30 seconds training is acceptable
  • Similar recall to LR but more reliable
  • Can provide feature importance
  • Handles complex disease patterns
```

### **Scenario 3: Research Institution with High Standards**

```
Requirement: Absolute best accuracy, willing to trade interpretability

Solution: ✅ XGBoost
Why:
  • Best recall (90%+) catches most diseases
  • Best ROC-AUC discriminates risk well
  • Can use SHAP for some interpretability
  • Production-proven technology
```

---

# Which Model to Choose?

## Decision Framework

```
START
  ↓
Is accuracy critical?
  ├─ YES → Need 90%+ recall → XGBoost ✅
  └─ NO → 85%+ is fine
         ↓
    Is interpretability required?
      ├─ YES (doctors must understand) → Logistic Regression ✅
      └─ NO (black box acceptable)
         ↓
      Is speed important?
        ├─ YES (real-time screening) → Logistic Regression ✅
        └─ NO → Random Forest or XGBoost ✅
```

## For Our Liver Disease Screening Project

**Recommendation: XGBoost**

**Why:**
1. ✅ Meets all performance targets (Recall ≥ 85%, Precision ≥ 70%)
2. ✅ Highest recall (90%+) = catches most diseased patients
3. ✅ 45-second training is acceptable for periodic retraining
4. ✅ Can use SHAP to partially explain predictions to clinicians
5. ✅ Industry standard (most hospitals use it)
6. ✅ Robust against outliers in blood tests

**But also keep Logistic Regression as backup**:
- If we need to explain predictions to doctors
- If computational resources are limited
- As a sanity check (LR should also get good recall)

---

# Clinical Examples

## Patient Case Studies: How Each Model Predicts

### **Patient 1: Mr. Singh (68-year-old male)**

**Blood Test Results:**
```
Age: 68
Gender: Male (1)
Total Bilirubin: 2.5 (high)
Direct Bilirubin: 1.8 (high)
Alkaline Phosphatase: 95 (high)
ALT: 58 (elevated)
AST: 72 (elevated)
Total Proteins: 6.5 (low)
Albumin: 2.8 (low)
Albumin/Globulin Ratio: 0.65 (low)
```

**Actual Status: HAS LIVER DISEASE (Class = 1)**

#### Logistic Regression's Decision:

```
Weights learned:
  Age: 0.02 × 68 = 1.36
  Bilirubin: 0.45 × 2.5 = 1.13
  AST: 0.38 × 72 = 27.36
  Albumin: -0.52 × 2.8 = -1.46
  ... (more terms)

Total Score: +15.20

Probability = 1/(1 + e^(-15.20)) = 0.999 = 99.9%

Prediction: DISEASED ✅ CORRECT
Confidence: 99.9%

Explanation: "Patient has very high AST, high bilirubin,
and low albumin - classic liver disease pattern"
```

#### Random Forest's Decision:

```
Tree 1 (sees bilirubin, AST, albumin):    Diseased (95%)
Tree 2 (sees age, alkaline phosphatase):  Diseased (88%)
Tree 3 (sees proteins, ratios):           Diseased (92%)
Tree 4 (sees gender, direct bilirubin):   Diseased (97%)
... (96 more trees mostly diseased)

Final Vote: 98 out of 100 trees say DISEASED

Prediction: DISEASED ✅ CORRECT
Confidence: 98%

Explanation: "Most trees recognize disease pattern"
```

#### XGBoost's Decision:

```
Tree 1 (bilirubin rule):  +0.68 (leans disease)
Tree 2 (corrects errors): +0.32
Tree 3 (AST focus):       +0.45
Tree 4 (albumin focus):   +0.28
... (96 more, average +0.34 each)

Total Score: +0.68 + 0.32 + 0.45 + 0.28 + (96×0.34) = +35.8

Probability = 1/(1 + e^(-35.8)) ≈ 100%

Prediction: DISEASED ✅✅ CONFIDENT
Confidence: 99.999%

Explanation: "All boosting rounds agree - clear disease pattern"
```

**Consensus: ALL THREE MODELS AGREE → DISEASED ✅**

---

### **Patient 2: Mrs. Patel (45-year-old female)**

**Blood Test Results:**
```
Age: 45
Gender: Female (0)
Total Bilirubin: 0.8 (normal)
Direct Bilirubin: 0.2 (normal)
Alkaline Phosphatase: 65 (normal)
ALT: 28 (normal)
AST: 32 (normal)
Total Proteins: 7.0 (normal)
Albumin: 3.5 (normal)
Albumin/Globulin Ratio: 1.1 (normal)
```

**Actual Status: NO LIVER DISEASE (Class = 0)**

#### Logistic Regression's Decision:

```
Score calculation:
  All features are normal/healthy
  Total Score: -2.50

Probability = 1/(1 + e^(-(-2.50))) = 0.076 = 7.6%

Prediction: HEALTHY ✅ CORRECT
Confidence: 92.4%
```

#### Random Forest's Decision:

```
All 100 trees see normal patterns
100 out of 100 trees say HEALTHY

Prediction: HEALTHY ✅ CORRECT
Confidence: 100%
```

#### XGBoost's Decision:

```
All boosting rounds see healthy pattern
Total Score: -8.50

Probability = 0.0002 = 0.02%

Prediction: HEALTHY ✅ CORRECT
Confidence: 99.98%
```

**Consensus: ALL THREE MODELS AGREE → HEALTHY ✅**

---

### **Patient 3: Mr. Desai (52-year-old male)** ⚠️ EDGE CASE

**Blood Test Results:**
```
Age: 52
Total Bilirubin: 1.3 (borderline)
Direct Bilirubin: 0.5 (borderline)
Alkaline Phosphatase: 82 (borderline)
ALT: 45 (borderline)
AST: 48 (borderline)
Total Proteins: 6.8 (borderline)
Albumin: 3.2 (borderline)
Albumin/Globulin Ratio: 0.95 (borderline)
```

**Actual Status: NO LIVER DISEASE (Class = 0)**

#### Logistic Regression's Decision:

```
Score: -0.15

Probability = 1/(1 + e^(-(-0.15))) = 0.463 = 46.3%

Prediction: HEALTHY (just barely)
Confidence: 53.7%

⚠️ RISKY: Very uncertain (46% is almost a coin flip!)
```

#### Random Forest's Decision:

```
Tree 1: Healthy (52%)
Tree 2: Diseased (58%)
Tree 3: Healthy (51%)
Tree 4: Diseased (62%)
... (mixed results)

Final Vote: 45 trees say healthy, 55 say diseased

Prediction: DISEASED
Confidence: 55%

⚠️ SLIGHTLY BIASED: Leans toward disease (false positive)
```

#### XGBoost's Decision:

```
Trees identify complex borderline pattern
Total Score: +0.32

Probability = 0.58 = 58%

Prediction: DISEASED
Confidence: 58%

⚠️ FALSE POSITIVE: Predicts disease when patient is healthy
This is acceptable though - better to flag and retest
than to miss actual disease
```

**Consensus: SPLIT (LR says healthy, RF & XGB say diseased)**

**What This Means for Clinical Decision:**

```
Patient score is BORDERLINE
Recommendation: RETEST in 3-6 months

Why? 
- If disease is developing, it will become obvious in retest
- If false alarm, repeat test will confirm patient is healthy
- Safety first: better to monitor than miss disease
```

---

## Summary: When Each Model Excels

```
Clear Disease Cases (high markers):
  ✅ All models agree → DISEASED
  Confidence: 99%+
  Action: Immediate specialist referral

Clear Healthy Cases (normal markers):
  ✅ All models agree → HEALTHY
  Confidence: 99%+
  Action: Routine follow-up in 1-2 years

Borderline Cases (mixed markers):
  ⚠️ Models disagree
  Confidence: 50-70%
  Action: Retest in 3-6 months
  Use SHAP to understand which features are conflicting
```

---

## Performance Summary for Your Project

Based on the three models trained on your liver disease data:

```
Test Set Results (with 50-50 balanced training data):

═══════════════════════════════════════════════════════
           Recall    Precision    F1      ROC-AUC
═══════════════════════════════════════════════════════
Target     ≥ 0.85    ≥ 0.70       N/A     ≥ 0.80
───────────────────────────────────────────────────────
LR         0.8495    0.8276       0.8384  0.8921  ✓
RF         0.8554    0.8019       0.8280  0.8734  ✓
XGBoost    0.8795    0.8292       0.8536  0.9124  ✅
═══════════════════════════════════════════════════════

✅ = Exceeds all targets
✓ = Meets all targets
```

**Final Recommendation: XGBoost is the winner!**

- Highest recall (87.95%) → catches most diseases
- Good precision (82.92%) → acceptable false alarm rate
- Best ROC-AUC (0.9124) → excellent risk discrimination
- All metrics exceed targets

---

## Next Steps: SHAP Explanation & Risk Stratification

Once we've selected XGBoost as our final model, we can:

1. **Use SHAP** to explain individual predictions to doctors
2. **Create Risk Tiers** (Low/Medium/High) based on probability scores
3. **Validate Clinical Performance** with medical professionals
4. **Deploy in Production** for real screening

---

# Backend Architecture - Complete System Design

## Overview

This section explains the theoretical foundations and architectural design of the backend system that serves machine learning predictions to the frontend application. The backend acts as a bridge between the React frontend and the trained ML models (Supervised, Unsupervised, and SHAP).

---

## 1. Architectural Pattern: Service-Oriented Layered Architecture

The backend follows a **layered architecture** pattern, which separates concerns into distinct tiers:

```
┌─────────────────────────────────────────┐
│   Presentation Layer (FastAPI Routes)   │  ← HTTP/REST + WebSocket APIs
├─────────────────────────────────────────┤
│   Business Logic Layer (predict.py)     │  ← Orchestration & Coordination
├─────────────────────────────────────────┤
│   Model Layer (Individual Predictors)   │  ← Supervised, Unsupervised, SHAP
├─────────────────────────────────────────┤
│   Data Access Layer (Model Loaders)     │  ← Model Persistence & Loading
└─────────────────────────────────────────┘
```

### Why This Architecture?

**Separation of Concerns:**
- Each layer has a single, well-defined responsibility
- API routes handle HTTP/WebSocket communication only
- Business logic handles orchestration without knowing HTTP details
- Model layer focuses purely on ML computations
- Data access layer handles model persistence

**Benefits:**
- **Testability**: Each layer can be tested independently
- **Maintainability**: Changes in one layer don't affect others
- **Scalability**: Layers can be scaled independently
- **Reusability**: Model layer can be used by different interfaces (CLI, API, etc.)

---

## 2. FastAPI Framework - Theoretical Rationale

### Why FastAPI?

**Asynchronous I/O Model:**

Traditional synchronous servers process requests one at a time:
```
Request 1 → Process (blocking) → Response
Request 2 → Wait... → Process (blocking) → Response
Request 3 → Wait... → Process (blocking) → Response
```

FastAPI uses asynchronous I/O, allowing concurrent request handling:
```
Request 1 → Process → (meanwhile)
Request 2 → Process → (meanwhile)
Request 3 → Process → Responses sent when ready
```

**Key Benefits:**
- **Concurrency**: Handle multiple requests simultaneously
- **Non-blocking**: While one prediction runs, server accepts other requests
- **Efficiency**: Single-threaded event loop uses resources efficiently
- **Scalability**: Can handle 1000s of concurrent connections

### Automatic API Documentation

FastAPI automatically generates OpenAPI/Swagger documentation:
- **Type hints** → Automatic validation and documentation
- **Pydantic models** → Request/response schemas with validation
- **Zero configuration** → Documentation available at `/docs` endpoint

### Type Safety with Pydantic

Pydantic provides runtime validation based on Python type hints:
```python
class PatientData(BaseModel):
    age: int = Field(ge=0, le=120)  # Must be 0-120
    totalBilirubin: float = Field(ge=0.0)  # Must be non-negative
```

**Benefits:**
- **Data Validation**: Invalid data rejected before reaching models
- **Error Messages**: Clear feedback on what's wrong
- **Documentation**: Schema automatically documented
- **Type Safety**: Reduces bugs from type mismatches

---

## 3. Unified Prediction Pipeline - Facade Pattern

### Design Pattern: Facade

The `predict.py` file implements the **Facade Pattern**, providing a unified interface to multiple subsystems:

```
Client Code
    ↓
[Facade: UnifiedPredictor] ← Single entry point
    ├──→ [Subsystem 1: Supervised Predictor]
    ├──→ [Subsystem 2: Unsupervised Predictor]
    └──→ [Subsystem 3: SHAP Explainer]
```

**Why Facade?**
- **Simplified Interface**: Client code doesn't need to know about all subsystems
- **Loose Coupling**: Subsystems can change without affecting clients
- **Centralized Logic**: All orchestration in one place

### Pipeline Execution Flow

```
Input Patient Data
    ↓
┌─────────────────────────────────────┐
│  1. Data Validation & Preprocessing │
│     - Type checking                 │
│     - Range validation              │
│     - Feature scaling               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. Parallel Model Execution        │
│     ┌─────────────┐                 │
│     │ Supervised  │ → Risk Score    │
│     └─────────────┘                 │
│     ┌─────────────┐                 │
│     │Unsupervised │ → Cluster       │
│     └─────────────┘                 │
│     ┌─────────────┐                 │
│     │   SHAP      │ → Explanations  │
│     └─────────────┘                 │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. Result Aggregation              │
│     - Combine outputs               │
│     - Format for frontend           │
└─────────────────────────────────────┘
    ↓
Response to Client
```

**Key Principles:**
- **Parallel Execution**: Independent models can run concurrently (using asyncio)
- **Idempotency**: Same input always produces same output
- **Determinism**: Fixed random seeds ensure reproducibility

---

## 4. WebSocket Communication Theory

### Why WebSockets for Bidirectional Communication?

**HTTP Request-Response Model (Limitations):**

```
Client → [Request] → Server
Client ← [Wait...] ← Server
Client → [Request] → Server (new connection needed)
```

**Problems:**
- **Connection Overhead**: Each request establishes new TCP connection
- **Unidirectional**: Server can't push updates to client
- **Stateless**: No persistent connection context

**WebSocket Model:**

```
Client ←→ [Persistent Connection] ←→ Server
   ↑                                      ↓
   └── Can send anytime ──────────────────┘
```

**Benefits:**
- **Persistent Connection**: One handshake, many messages
- **Bidirectional**: Both client and server can send anytime
- **Low Latency**: No connection overhead per message
- **Stateful**: Connection maintains context

### WebSocket Protocol Theory

**Handshake Phase:**
```
1. Client sends HTTP upgrade request with special headers
2. Server responds with "101 Switching Protocols"
3. Connection upgraded to WebSocket protocol
4. Full-duplex communication begins
```

**Frame-Based Communication:**
- Messages sent as **frames** (binary or text)
- FastAPI handles frame encoding/decoding automatically
- Supports message boundaries and fragmentation for large messages

**Connection Lifecycle:**
```
┌──────────┐      ┌──────────┐
│  Client  │      │  Server  │
└────┬─────┘      └────┬─────┘
     │                 │
     │─── Connect ────>│
     │<── Accepted ────│
     │                 │
     │─── Predict ────>│
     │                 │ [Processing...]
     │<── Result ──────│
     │                 │
     │─── Predict ────>│
     │<── Result ──────│
     │                 │
     │─── Close ──────>│
     │<── Closed ──────│
```

---

## 5. Model Persistence Strategy

### Why Save Models to Disk?

**Training vs. Inference Separation:**

```
Training Phase (Offline):
Data → [Train Models] → Save to Disk
    (Expensive, takes minutes/hours)

Inference Phase (Online):
Request → [Load Models] → Predict → Response
    (Fast, takes milliseconds)
```

**Benefits:**
- **Speed**: No retraining required on server startup
- **Consistency**: Same model used for all predictions
- **Resource Efficiency**: Training requires GPU/CPU-intensive operations
- **Version Control**: Can keep multiple model versions

### Model Serialization Theory

**Pickle/Joblib Serialization:**
- Python objects → Binary representation → Disk file
- Preserves complete object state: weights, parameters, configuration
- Deserialization reconstructs exact object in memory

**Model Artifacts Stored:**

```
XGBoost Model:
- Tree structures (decision rules)
- Leaf values (predictions at leaves)
- Feature weights and importance

Scaler Objects:
- Mean/Median values (for normalization)
- Scale factors (standard deviation/IQR)
- Transformation parameters

SHAP Explainer:
- Tree structure reference (TreeExplainer)
- Expected values (baseline prediction)
- Background data reference (if needed)

UMAP Reducer:
- Embedding parameters
- Manifold structure learned

KMeans Centroids:
- Cluster center coordinates
- Cluster labels mapping
```

---

## 6. Supervised Learning Predictor - Internal Theory

### XGBoost Prediction Process

**Mathematical Flow:**
```
Input Features (x₁, x₂, ..., xₙ)
    ↓
[Feature Scaling] → Normalized features (same as training)
    ↓
[Tree Ensemble Evaluation]
    ├─ Tree 1 → score₁
    ├─ Tree 2 → score₂
    ├─ Tree 3 → score₃
    └─ ... → scoreₙ (for all trees)
    ↓
[Sum all scores] → raw_score = Σ(scores)
    ↓
[Sigmoid Function] → probability = 1 / (1 + e^(-raw_score))
    ↓
Output: Probability (0 to 1)
```

**Decision Threshold:**
- Probability ≥ 0.5 → "Disease Risk"
- Probability < 0.5 → "No Disease Risk"

### Feature Scaling Importance

**Why RobustScaler is Used:**
- **Robust to Outliers**: Uses median and quartiles (not mean/std)
- **Formula**: `scaled = (x - Q₂) / (Q₃ - Q₁)` where Q = quartile
- **Preserves Relationships**: Relative distances between values maintained

**Consistency Principle:**
- **Critical**: Training data was scaled → Inference data MUST use same scaler
- **Different scaling = Invalid predictions**: Model assumptions break
- **Same scaler object**: Load exact scaler from training phase

---

## 7. Unsupervised Learning Predictor - Clustering Theory

### Clustering Pipeline Theory

**Step-by-Step Process:**

**1. Outlier Detection (Isolation Forest):**
```
Theory: Outliers are "isolated" points in feature space
- Normal points: Many neighbors nearby
- Outliers: Few/no neighbors nearby

Algorithm identifies and removes outliers
before clustering (ensures cleaner clusters)
```

**2. Dimensionality Reduction (UMAP):**
```
High-dimensional space (9 features)
    ↓
[UMAP Manifold Learning]
    ↓
Low-dimensional space (2 dimensions)

Theory: Preserves local neighborhood structure
- Points close in 9D remain close in 2D
- Makes clustering more effective and interpretable
- Non-linear dimensionality reduction
```

**3. Cluster Assignment (PSO-KMeans):**
```
UMAP Space (2D points)
    ↓
[Find closest centroid using PSO-optimized positions]
    ↓
Cluster Assignment (0 or 1)

Theory: PSO (Particle Swarm Optimization) finds
optimal centroid positions before KMeans
- Better initial placement than random
- Results in more meaningful clusters
```

**4. Cluster-to-Risk Mapping:**
```
Cluster Assignment → Analyze cluster characteristics
    ↓
Compare cluster mean biomarker values
    ↓
Map to risk level:
- Cluster with higher bilirubin → "High Risk"
- Cluster with normal values → "Low Risk"
```

**Distance to Centroid:**
- Measures how "central" patient is to cluster
- Low distance = Typical cluster member
- High distance = Atypical (may be edge case)

---

## 8. SHAP Values - Explainability Theory

### SHAP Value Theory

**Shapley Values (Game Theory Origin):**
```
Concept: Fair allocation of contribution among players

In ML context:
- Model prediction = "total payoff"
- Each feature = "player"
- SHAP value = "fair share" each feature contributes
```

**Additivity Property:**
```
Prediction = Base Value + Σ(SHAP_values)

Example:
Base Value (average prediction): 0.42
+ Total Bilirubin SHAP: +0.15
+ Albumin SHAP: -0.08
+ AST SHAP: +0.12
────────────────────────────────
Final Prediction: 0.61 (High Risk)
```

**TreeExplainer Algorithm:**
- **TreeSHAP**: Computes exact SHAP values for tree models
- **Efficient**: O(TL²D) complexity where T=trees, L=leaves, D=depth
- **Exact Values**: No approximation needed (unlike KernelSHAP)

### SHAP Output Interpretation

**Feature Contributions:**
- **Positive SHAP** → Pushes prediction toward disease
- **Negative SHAP** → Pushes prediction toward healthy
- **Magnitude** → Strength of contribution

**Clinical Application:**
- Shows **which biomarkers** drive the prediction
- Explains **why** patient is high/low risk
- Helps doctors **understand** model reasoning

---

## 9. Request-Response Data Flow Theory

### State Management

**Stateless Design (REST):**
```
Each request contains all necessary information
No server-side state stored between requests

Benefits:
- Scalability: Any server can handle any request
- Reliability: Server crash doesn't lose state
- Simplicity: No session management needed
```

**Request Lifecycle:**
```
1. Client → HTTP Request (JSON/Form data)
2. FastAPI → Route Handler (extract data)
3. Validation → Pydantic (ensure correctness)
4. Business Logic → UnifiedPredictor
5. Models → Process (compute predictions)
6. Response → Format & Return JSON
7. Client ← Receives results
```

### Error Handling Theory

**Fail-Fast Principle:**
- **Validate Early**: Catch errors at input boundary
- **Prevent Waste**: Invalid data doesn't reach models
- **Clear Feedback**: Return specific error messages

**Error Propagation:**
```
Model Error → Exception → Handler → HTTP 500 (Internal Server Error)
Validation Error → Pydantic → HTTP 422 (Unprocessable Entity)
Not Found → HTTP 404
Unauthorized → HTTP 401
```

---

## 10. Bulk Processing Theory

### Batch Processing Strategy

**Sequential vs. Parallel:**
```
Sequential (Simple):
Patient 1 → Predict → Result 1
Patient 2 → Predict → Result 2
Patient 3 → Predict → Result 3
Time: n × prediction_time

Parallel (Optimized):
Patient 1 ┐
Patient 2 ├→ [Batch Predict] → Results
Patient 3 ┘
Time: ~prediction_time (if batched)
```

**Memory Considerations:**
- Large CSV files → Process in chunks (streaming)
- Keep memory usage bounded (don't load entire CSV)
- Process chunk-by-chunk if needed

### Data Aggregation Theory

**Summary Statistics:**
```
Individual Results → Aggregate
    ↓
- Count by risk level (High/Medium/Low)
- Average probability
- Distribution analysis
- Export to CSV for download
```

---

## 11. CORS and Frontend Integration Theory

### Cross-Origin Resource Sharing (CORS)

**Problem:**
```
Frontend: http://localhost:5173 (Vite dev server)
Backend:  http://localhost:8000 (FastAPI)

Browser blocks requests (different origins = security risk)
```

**Solution:**
```
Backend adds CORS headers:
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

**How It Works:**
1. Browser sends **preflight request** (OPTIONS) before actual request
2. Server responds with **allowed origins/methods**
3. Browser checks if request is allowed
4. If allowed, sends actual request

### API Contract Theory

**Contract-First Design:**
- Define **request/response schemas** (Pydantic models)
- Frontend and backend both use same contract
- **Type safety**: Prevents mismatched data structures
- **Documentation**: Schemas serve as API documentation

---

## 12. Performance Optimization Theory

### Model Caching Strategy

**Lazy Loading:**
```
Models loaded on first use, then cached in memory
Reduces startup time
Memory used only when needed
```

**Loading Pattern:**
```python
# First request: Load models (slow, ~2-3 seconds)
# Subsequent requests: Use cached models (fast, ~10ms)
```

### Connection Pooling (Future Optimization)

```
Multiple requests → Connection pool → Backend
Reuses connections (reduces TCP handshake overhead)
```

---

## Summary: Core Theoretical Principles

1. **Separation of Concerns**: API, business logic, and ML separated into layers
2. **Facade Pattern**: Unified interface hides complexity of multiple subsystems
3. **Stateless Design**: Each request is independent (enables horizontal scaling)
4. **Type Safety**: Validation at boundaries prevents errors
5. **Model Persistence**: Train once, use many times (efficiency)
6. **Bidirectional Communication**: WebSockets enable real-time interaction
7. **Explainability**: SHAP values provide interpretability for clinical use
8. **Fail-Fast**: Validate early, fail clearly with helpful errors
9. **Scalability**: Stateless design enables horizontal scaling
10. **Consistency**: Same preprocessing pipeline as training (determinism)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                          │
│  (Doctor-Friendly Liver Disease Dashboard)                  │
└──────────────────┬──────────────────────────────────────────┘
                   │ HTTP/REST + WebSocket
                   ↓
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  API Routes Layer                                      │ │
│  │  - /api/predict/individual (POST)                      │ │
│  │  - /api/predict/bulk (POST)                            │ │
│  │  - /ws/predict (WebSocket)                             │ │
│  └────────────────────┬───────────────────────────────────┘ │
│                       ↓                                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Unified Predictor (predict.py)                        │ │
│  │  - Data validation & preprocessing                     │ │
│  │  - Orchestrates all models                            │ │
│  │  - Result aggregation                                  │ │
│  └─────┬──────────┬──────────┬────────────────────────────┘ │
│        ↓          ↓          ↓                               │
│  ┌─────────┐ ┌──────────┐ ┌─────────┐                      │
│  │Supervised│ │Unsupervised│ │  SHAP  │                      │
│  │Predictor│ │  Predictor  │ │Explainer│                     │
│  └────┬────┘ └──────┬─────┘ └────┬────┘                     │
│       │             │            │                           │
│       └─────────────┴────────────┘                           │
│                       ↓                                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Model Loader Layer                                    │ │
│  │  - Load from saved_models/ directory                  │ │
│  │  - Cache in memory                                     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                   │
                   ↓
        ┌──────────────────────┐
        │  saved_models/       │
        │  - xgboost_model.pkl │
        │  - scalers.pkl       │
        │  - shap_explainer.pkl│
        │  - umap_reducer.pkl  │
        └──────────────────────┘
```

---

---

# New Computer-Vision Models: Ultrasound-Based Liver Assessment

This section extends the original clinical-model guide with the two ultrasound image models added to the project. The clinical models use structured blood-test and demographic features. These two models instead process ultrasound images, and their outputs should be interpreted as image-model estimates for the specific dataset labels—not as definitive diagnoses.

## Model 4: Byra Ultrasound Vision Transformer (ViT-B/16)

### 1. Purpose and target

The Byra branch classifies liver ultrasound images into:

- **Class 0 — Normal**
- **Class 1 — Steatosis**

The source dataset is `data/ultrasound/dataset_liver_bmodes_steatosis_assessment_IJCARS.mat`, associated with Byra et al., *Transfer learning with deep convolutional neural network for liver steatosis assessment in ultrasound images* (IJCARS, 2018).

**Important:** this dataset supports normal-versus-steatosis image classification. It does **not** establish whether steatosis is alcohol-related or non-alcoholic, and the model should not be described as an alcohol-vs-nonalcoholic classifier.

### 2. Dataset and split

| Property | Recorded value |
|---|---:|
| Number of patients | 55 |
| Images per patient | 10 |
| Total images | 550 |
| Image dimensions | 434 × 636 pixels |
| Image format | Grayscale, `uint8` |
| Normal patients (class 0) | 17 |
| Steatosis patients (class 1) | 38 |
| Training split | 38 patients / 380 images |
| Validation split | 8 patients / 80 images |
| Held-out test split | 9 patients / 90 images |

The split is **patient-level**, so images from the same patient are kept in one partition. This is preferable to randomly splitting individual images because images from the same person can be very similar. The test set is nevertheless small (nine patients), so its results have substantial uncertainty.

### 3. Model architecture

The image model uses a **Vision Transformer (ViT-B/16)** initialized with pretrained ImageNet weights.

```text
Ultrasound image (grayscale)
          |
          v
Image preparation / resize / tensor conversion
          |
          v
Patch embedding (16 × 16 patches)
          |
          v
Learned positional embeddings + class token
          |
          v
Transformer encoder blocks
(self-attention + MLP + residual connections + normalization)
          |
          v
Classification representation
          |
          v
Fine-tuned classification head
          |
          v
Softmax probabilities
   ┌──────┴────────┐
   v               v
Normal          Steatosis
Class 0          Class 1
```

A ViT does not process the image as a sequence of convolutional feature maps in the same way as a conventional CNN. It divides the image into patches, embeds each patch as a token, adds positional information, and uses self-attention to learn relationships between patches. The classification head maps the learned image representation to the two target classes.

The recorded fine-tuning configuration updates the final three transformer blocks, the final normalization layer, and the classification head while keeping the earlier backbone layers frozen. The model has **21,266,690 trainable parameters out of 85,800,194 total parameters (24.79%)**.

### 4. Training and validation approach

- Pretrained initialization: ImageNet.
- Fine-tuning: last three transformer blocks, final normalization, and classification head.
- Split strategy: patient-level train/validation/test.
- Model selection: validation performance.
- Reported additional evaluation: five-fold cross-validation.
- Class imbalance: the dataset has more steatosis patients than normal patients, so per-class metrics matter in addition to overall accuracy.

The exact optimizer, learning-rate schedule, augmentations, and checkpoint-selection details should be taken from the training script/configuration. They should not be inferred from this overview alone.

### 5. Reported performance

#### Five-fold cross-validation

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.8909 (89.09%) | 0.1060 |
| Precision | 0.9194 (91.94%) | 0.0683 |
| Recall / Sensitivity | 0.9179 (91.79%) | 0.1127 |
| F1-score | 0.9171 (91.71%) | 0.0871 |
| ROC-AUC | 0.8988 (89.88%) | 0.1086 |

These are mean ± standard deviation across the recorded folds. The standard deviations—especially for accuracy and recall—show that performance varies between folds.

#### Held-out patient-level test set

| Metric | Value |
|---|---:|
| Accuracy | 0.7778 (77.78%) |
| Precision | 0.7500 (75.00%) |
| Recall / Sensitivity | 1.0000 (100.00%) |
| F1-score | 0.8571 (85.71%) |
| ROC-AUC | 0.7778 (77.78%) |
| Specificity | 0.3333 (33.33%) |

Recorded confusion matrix, with rows as actual classes and columns as predicted classes:

|  | Predicted Normal | Predicted Steatosis |
|---|---:|---:|
| Actual Normal | 1 | 2 |
| Actual Steatosis | 0 | 6 |

From this matrix, the model detected all six steatosis cases in this small test set, but it incorrectly flagged two of the three normal cases. Thus, the test recall of 100% comes with low specificity (33.33%) and should not be interpreted as perfect overall performance. One or two cases can change these percentages substantially because only nine patients were tested.

### 6. Example output

One recorded example for patient 11 was:

| Output | Value |
|---|---:|
| Predicted class | Normal |
| Normal probability | 99.414% |
| Steatosis probability | 0.605% |

These are model probabilities for that image/input, not a guarantee of clinical certainty. They should be displayed with the model name, target labels, and appropriate caution.

---

## Model 5: BEHSOF Ultrasound Vision Transformer (ViT-B/16)

### 1. Purpose and target

The BEHSOF image branch classifies ultrasound images into:

- **NAFLD**
- **Non-NAFLD**

The full BEHSOF dataset has a Figshare record with clinical metadata and ultrasound-image information. The current image-only copy used for this branch does not include reliable patient identifiers, which limits patient-level validation.

Dataset record: [BEHSOF advanced non-alcoholic fatty liver dataset (Figshare)](https://figshare.com/articles/dataset/BEHSOF_Advanced_Non-alcoholic_fatty_liver_dataset_with_clinical_metadata_and_ultrasound_images_for_Deep_learning_Models/26389069).

### 2. Dataset and preprocessing

| Property | Recorded value |
|---|---:|
| Images in image-only copy | 1,669 |
| NAFLD images | 1,517 |
| Non-NAFLD images | 152 |
| Original image dimensions | 1024 × 768 |
| Image type | RGB |
| Duplicate images identified | 23 |
| Unique images after duplicate removal | 1,646 |

The dataset is strongly imbalanced: NAFLD images substantially outnumber Non-NAFLD images. The recorded image-level split (random seed 42, after duplicate removal) is:

| Partition | Total images | Non-NAFLD | NAFLD |
|---|---:|---:|---:|
| Training | 1,152 | 106 | 1,046 |
| Validation | 247 | 23 | 224 |
| Test | 247 | 23 | 224 |

**Validation limitation:** this is an image-level split, not a patient-level split, because patient identifiers are unavailable in the image-only copy. If images from the same patient or acquisition source occur in multiple partitions, test performance can be optimistic. The results should therefore be treated as preliminary image-level performance, not proof of generalization to unseen patients or hospitals.

### 3. Model architecture

This branch also uses **ViT-B/16** with pretrained ImageNet weights, but it is a separate model trained for the BEHSOF dataset and its NAFLD/Non-NAFLD labels. It is not the same checkpoint as the Byra model.

```text
RGB ultrasound image (1024 × 768)
             |
             v
Duplicate removal + train/validation/test partition
             |
             v
Resize / normalize to model input (224 × 224)
             |
             v
Patch embedding (16 × 16 patches)
             |
             v
Positional embeddings + class token
             |
             v
ViT-B/16 transformer encoder
(self-attention + MLP blocks)
             |
             v
Classification head
             |
             v
Two-class scores / probabilities
       ┌─────┴──────┐
       v            v
    Non-NAFLD      NAFLD
```

Each 224 × 224 input produces a grid of 14 × 14 patches for a 16 × 16 patch size, plus the class token. Self-attention lets the model combine information across image regions before the classification head predicts the dataset label.

### 4. Recorded training configuration

| Setting | Value |
|---|---|
| Backbone | ImageNet-pretrained ViT-B/16 |
| Input size | 224 × 224 |
| Batch size | 8 |
| Maximum epochs | 15 |
| Early stopping patience | 4 |
| Classification-head learning rate | 5 × 10⁻⁵ |
| Backbone learning rate | 5 × 10⁻⁶ |
| Weight decay | 0.05 |
| Learning-rate schedule | Cosine scheduler |
| Best recorded validation ROC-AUC | 0.9728 at epoch 15 |

The use of a smaller learning rate for the pretrained backbone than for the head is intended to preserve useful pretrained representations while allowing the task-specific classifier to adapt. The best validation AUC is a model-selection result, not a test-set result.

### 5. Held-out image-level test performance

| Metric | Value |
|---|---:|
| Accuracy | 0.9717 (97.17%) |
| Precision | 0.9780 (97.80%) |
| Recall / Sensitivity | 0.9911 (99.11%) |
| Specificity | 0.7826 (78.26%) |
| F1-score | 0.9845 (98.45%) |
| Balanced accuracy | 0.8868 (88.68%) |
| ROC-AUC | 0.9589 (95.89%) |
| PR-AUC | 0.9925 (99.25%) |

Recorded confusion matrix, with rows as actual classes and columns as predicted classes:

|  | Predicted Non-NAFLD | Predicted NAFLD |
|---|---:|---:|
| Actual Non-NAFLD | 18 | 5 |
| Actual NAFLD | 2 | 222 |

The high accuracy and recall should be interpreted alongside the class imbalance. The model correctly identifies most NAFLD images, but five of 23 Non-NAFLD images are incorrectly classified as NAFLD. The balanced accuracy (88.68%) and specificity (78.26%) are useful complementary metrics because they expose errors on the minority class.

### 6. Why accuracy alone is insufficient

When 1,517 of 1,669 images are NAFLD, a classifier that heavily favors NAFLD can achieve high accuracy without handling the minority class well. Always report at least precision, recall/sensitivity, specificity, F1-score, ROC-AUC, and the confusion matrix. For imbalanced data, also inspect PR-AUC and balanced accuracy. The reported metrics do not remove the patient-level split limitation.

---

## Metric Definitions: Precision, Recall, F1, Specificity, and AUC

For binary classification, define:

- **TP (true positive):** positive cases correctly predicted positive.
- **TN (true negative):** negative cases correctly predicted negative.
- **FP (false positive):** negative cases incorrectly predicted positive.
- **FN (false negative):** positive cases incorrectly predicted negative.

| Metric | Formula | Interpretation |
|---|---|---|
| Accuracy | `(TP + TN) / (TP + TN + FP + FN)` | Fraction of all predictions that are correct |
| Precision (positive predictive value) | `TP / (TP + FP)` | Among predicted-positive cases, the fraction that are truly positive |
| Recall / Sensitivity | `TP / (TP + FN)` | Among actual-positive cases, the fraction detected |
| Specificity | `TN / (TN + FP)` | Among actual-negative cases, the fraction correctly ruled negative |
| F1-score | `2 × Precision × Recall / (Precision + Recall)` | Harmonic mean of precision and recall |
| ROC-AUC | Area under ROC curve | Ranking/discrimination across thresholds; not the same as accuracy |
| PR-AUC | Area under precision–recall curve | Precision/recall trade-off, particularly useful with class imbalance |
| Balanced accuracy | `(Sensitivity + Specificity) / 2` | Average performance on positive and negative classes |

A metric can be undefined if its denominator is zero; implementation libraries may return a warning or use a configured zero-division value. State the positive class and averaging convention (binary, macro, weighted, or micro) when reporting metrics. The tables above preserve the recorded values; verify the exact scoring configuration in the evaluation scripts before formal publication.

### How to interpret trade-offs in liver screening

- **High recall** is important when missing a possible case is costly, but high recall alone can come with many false positives.
- **High precision** means fewer false alarms among positive predictions, but does not show how many actual cases were missed.
- **Specificity** measures how well the model recognizes negative cases.
- **F1-score** balances precision and recall but does not include true negatives directly.
- **ROC-AUC** evaluates ranking over thresholds; it does not establish calibration or clinical utility.
- **Calibration** asks whether predictions assigned, for example, a 0.80 probability are positive about 80% of the time. A high AUC does not guarantee calibrated probabilities.

Do not describe a model as clinically validated solely because one test metric is high. Validation requires an appropriate split, a clearly defined reference standard, adequate sample size, external evaluation where possible, and clinical review.

---

## Combined Model Inventory and Performance Summary

The project now has three structured-data baseline classifiers and two ultrasound image classifiers. They solve related but **not identical** tasks and use different datasets; their scores must not be compared as if they were measured on the same patients or target.

| Model | Input / dataset | Target | Evaluation reported | Precision | Recall / Sensitivity | F1-score | ROC-AUC |
|---|---|---|---|---:|---:|---:|---:|
| Logistic Regression | ILPD structured clinical features | Liver-disease label | Recorded clinical test results | 0.8276 | 0.8495 | 0.8384 | 0.8921 |
| Random Forest | ILPD structured clinical features | Liver-disease label | Recorded clinical test results | 0.8019 | 0.8554 | 0.8280 | 0.8734 |
| XGBoost | ILPD structured clinical features | Liver-disease label | Recorded clinical test results | 0.8292 | 0.8795 | 0.8536 | 0.9124 |
| Byra ViT-B/16 | Byra ultrasound; patient-level held-out test | Normal vs. steatosis | 9 held-out patients / 90 images | 0.7500 | 1.0000 | 0.8571 | 0.7778 |
| BEHSOF ViT-B/16 | BEHSOF image-only copy | Non-NAFLD vs. NAFLD | Image-level held-out test | 0.9780 | 0.9911 | 0.9845 | 0.9589 |

For the Byra and BEHSOF branches, the displayed precision/recall/F1 values use the recorded positive-class convention in their evaluation output; verify the class-to-index mapping in the code before publication. Byra five-fold cross-validation results are separately reported above and should not be mixed with its held-out test values.

### Clinical-model interpretation

On the recorded ILPD metrics, XGBoost has the highest recall (0.8795), F1-score (0.8536), and ROC-AUC (0.9124) among the three clinical baselines. XGBoost also has the highest precision (0.8292), only slightly ahead of Logistic Regression (0.8276), so it leads on all four recorded metrics. These figures are project-reported results, not independently verified benchmarks. Confirm the dataset split, preprocessing, class balancing, threshold, and averaging settings in the training/evaluation scripts before using them in a paper.

### Do not rank all five models on one leaderboard

The clinical models predict a structured-data liver-disease label; the Byra model predicts normal versus steatosis on a small patient-level ultrasound test; the BEHSOF model predicts NAFLD versus Non-NAFLD on an image-level test. Different cohorts, labels, split methods, and reference standards make direct ranking misleading.

---

## Multimodal Evidence Aggregation: How the Model Outputs Are Combined

The application can accept structured patient data and an optional ultrasound image. Its orchestration layer can return clinical prediction output, unsupervised clustering output, SHAP explanation, ultrasound-model output, and a combined evidence score.

The currently recorded engineering formula is:

```text
S = 0.70 × Pclinical
  + 0.10 × PByra
  + 0.20 × PBEHSOF
```

where each `P` is a branch output used by the application. In the example previously recorded:

| Component | Example value | Weight | Weighted contribution |
|---|---:|---:|---:|
| Clinical model | 0.976625 | 0.70 | 0.683638 |
| Byra ultrasound model | 0.006050 | 0.10 | 0.000605 |
| BEHSOF ultrasound model | 0.591309 | 0.20 | 0.118262 |
| **Combined score** | — | **1.00 total weight** | **≈ 0.8025 (80.25/100)** |

The score is a **weighted evidence-aggregation score**, not a measured accuracy, not a validated probability of disease, and not a calibrated multimodal classifier. The branches were trained on different datasets and target definitions and are not paired observations from the same cohort. Before treating this score as a medical risk estimate, the project would need a common target label, patient-level paired data, a prespecified fusion-training procedure, calibration, and independent evaluation.

If the frontend labels the score as Low/Moderate/High using thresholds such as `<0.40`, `0.40–0.69`, and `≥0.70`, describe these as application-defined display bands—not clinically validated risk categories. Likewise, a UI label such as “agreement” must not be presented as diagnostic agreement unless the branch labels and thresholds have been harmonized and formally evaluated.

---

## Deployment and Reproducibility Notes for the Two Image Models

1. Keep the Byra and BEHSOF checkpoints separate: their input distributions and target labels differ.
2. Apply the same resize, channel conversion, normalization, and class-index mapping used during training.
3. Confirm whether the image model expects grayscale converted to three channels or a model-specific grayscale pipeline; do not silently assume these are interchangeable.
4. Load the exact saved checkpoint and its associated preprocessing configuration.
5. Record model version, checkpoint hash, threshold, predicted class, and class probabilities in evaluation logs.
6. Evaluate patient-level generalization. For BEHSOF, obtain patient identifiers or use a dataset version that reliably links images to patients, then split by patient before claiming generalization.
7. Report confusion matrices and minority-class metrics alongside accuracy.
8. Keep the output framed as decision support / research screening, not a standalone diagnosis.

## Updated Overall Architecture

```text
                         React / Doctor Dashboard
                                  |
                           FastAPI endpoint
                                  |
                         Unified prediction layer
                                  |
             ┌────────────────────┼────────────────────┐
             |                    |                    |
     Clinical branch       Unsupervised branch    SHAP explanation
     LR / RF / XGBoost     Isolation Forest       TreeExplainer
                           + UMAP + PSO-KMeans
             |
             | optional ultrasound upload
             v
       Image preprocessing
             |
        ┌────┴─────────┐
        |              |
   Byra ViT-B/16   BEHSOF ViT-B/16
   Normal vs.      Non-NAFLD vs.
   Steatosis       NAFLD
        |              |
        └──────┬───────┘
               |
      Evidence aggregation layer
      (engineering score; not calibrated)
               |
     Structured response to frontend
```

The exact routing and invocation behavior should be confirmed against the current backend implementation. In particular, an optional upload does not by itself prove that both image models are executed for every request; the route and predictor code determine that behavior.

## Research Reporting Checklist

Before publishing or presenting these results:

- [ ] Confirm the clinical-model split, preprocessing, balancing strategy, threshold, and averaging convention from the scripts.
- [ ] Confirm the positive class for each image model and the class-index mapping.
- [ ] Report patient-level Byra test results separately from five-fold cross-validation.
- [ ] Clearly identify BEHSOF results as image-level because patient identifiers are unavailable in the current copy.
- [ ] Include confusion matrices and class-wise performance, not only accuracy.
- [ ] Do not compare metrics across datasets as if the experiments were head-to-head.
- [ ] Describe the weighted multimodal score as an engineering aggregation until it is trained/calibrated and validated on paired patient-level data.
- [ ] Avoid diagnostic claims and include appropriate clinical oversight and external validation plans.

---

**This is our complete model understanding guide for the liver disease screening project!**

For any questions about specific models, hyperparameters, clinical interpretations, backend architecture, or the ultrasound models, refer to this document.

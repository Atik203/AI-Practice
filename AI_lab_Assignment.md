## Objective

The goal of this assignment is to implement Multiple Linear Regression from scratch, including:

- Data preprocessing and splitting
- Model parameter initialization
- Forward pass (matrix multiplication)
- Loss calculation
- Gradient descent parameter updates
- Model evaluation

## 1. Steps

### 1.1. Data Preprocess & Split

1.  **Drop unnecessary columns:**
    Remove columns that are not useful for prediction or contain identifiers: Examples include:
    - ID or Serial No.
      - these are just row identifiers and carry no predictive power.
    - Unnamed: 0
      - sometimes created accidentally when saving CSV.
    - Any duplicate columns or irrelevant textual descriptions.
2.  **Preprocess the dataset** according to the given guidelines (e.g., handle missing values, normalization, encoding if required).
3.  **Split the dataset** into training and testing sets:
    $(X_{train},y_{train})$, $(X_{test},y_{test})$

### 1.2. Model Function

We define and use a consistent set of equations that match the notebook (referenced as Eq. 1–Eq. 5 there):

**Hypothesis (Eq. 2 in notebook):**
$$\hat{y} = X\theta + b$$
where $\theta = [\theta_{1},\theta_{2},\dots,\theta_{n}]^{T}$ and $b$ is a scalar bias.

1. **Initialize parameters:** small random $\theta$ (e.g. $\mathcal{N}(0,0.01)$), and $b = 0$.
2. **Forward pass (training hypothesis):**
   $$\hat{y}_{\text{train}} = X_{\text{train}}\theta + b$$
3. **Mean Squared Error loss (Eq. 3):**
   $$L(\theta,b) = \frac{1}{m}\sum_{i=1}^{m}\big(\hat{y}_{i} - y_{i}\big)^{2}$$
   where $m$ is the number of training samples.
4. **Gradients (Eq. 4a, 4b):**
   $$\nabla_{\theta} L = \frac{2}{m} X_{\text{train}}^{T}(\hat{\mathbf{y}} - \mathbf{y}) \qquad (\text{Eq. 4a})$$
   $$\nabla_{b} L = \frac{2}{m}\sum_{i=1}^{m}(\hat{y}_{i} - y_{i}) \qquad (\text{Eq. 4b})$$

These expressions align exactly with the code lines annotated in the notebook.

### 1.3. Gradient Descent

**Parameter update rule (Eq. 5):**
$$\theta \leftarrow \theta - \alpha\nabla_{\theta} L, \qquad b \leftarrow b - \alpha\nabla_{b} L$$

This repeats until a stopping criterion is met (fixed iterations or convergence).

### 1.4. Evaluation

After training completes, we produce test predictions using the same hypothesis:
$$\hat{y}_{\text{test}} = X_{\text{test}}\theta_{\text{final}} + b_{\text{final}}$$

Then compute the test MSE (Eq. 6 in notebook):
$$\text{MSE}_{\text{test}} = \frac{1}{m_{\text{test}}}\sum_{i=1}^{m_{\text{test}}}\big(\hat{y}^{(\text{test})}_{i} - y^{(\text{test})}_{i}\big)^{2}$$

Optionally also compute $R^{2}$.

## Submission Instructions

- Submit your Jupyter Notebook or Python file.
- Include all plots (loss curve, predictions vs. actual).
- Clearly label each step in your code.

## Bonus (Optional)

Find the best learning rate for this problem (hyper-parameter tuning)

## Dataset for Testing

You may use the "Salary Dataset" from Kaggle, which has **no missing values** according to its description.
You can access it here: <https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set>
You can instruct students to download this for the assignment and use it for model training/testing.

df = pd.read_csv(path)

## Helper Code Snippet

Below is a clean reference implementation from scratch (NumPy only) mapping each line to the equations above:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data ----------------------------------------------------
df = pd.read_csv("concrete_data.csv")

# Feature / target split (assumes last column is target)
X = df.iloc[:, :-1].values  # shape (m, n)
y = df.iloc[:, -1].values.reshape(-1, 1)  # shape (m, 1)

# Train / test split ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardization (Eq. 1) -------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

m, n = X_train.shape

# Parameter initialization -------------------------------------
np.random.seed(42)
theta = np.random.normal(0, 0.01, size=(n, 1))  # θ ~ N(0, 0.01)
b = 0.0

learning_rate = 0.01
iterations = 1000
losses = []

for i in range(iterations):
    # Hypothesis (Eq. 2): ŷ = Xθ + b
    y_pred = X_train.dot(theta) + b

    # Residuals
    error = y_pred - y_train

    # Loss (Eq. 3)
    loss = np.mean(error ** 2)
    losses.append(loss)

    # Gradients (Eq. 4a, 4b)
    grad_theta = (2 / m) * X_train.T.dot(error)
    grad_b = (2 / m) * np.sum(error)

    # Update (Eq. 5)
    theta -= learning_rate * grad_theta
    b -= learning_rate * grad_b

# Evaluation (Eq. 6)
y_pred_test = X_test.dot(theta) + b
mse_test = np.mean((y_pred_test - y_test) ** 2)
print(f"Test MSE: {mse_test:.4f}")
```

## Why Some LaTeX Failed Earlier

- Missing braces: expressions like `\hat{y_i}` should be `\hat{y}_{i}`.
- Inline math inside code blocks was preceded by `$X=$` etc., which is invalid Python and confuses both parser and renderer.
- Inconsistent spacing around operators (e.g., `\hat{y_{i}}`) can render but is less clear than `\hat{y}_{i}`.
- Returned statement (`return X_train, ...`) appeared outside a function; removed for clarity.

## Consistency With Notebook

Equation labels now match notebook annotations:

- Eq. 1: Standardization (scaler)
- Eq. 2: Hypothesis
- Eq. 3: MSE loss
- Eq. 4a / 4b: Gradients
- Eq. 5: Parameter update
- Eq. 6: Test MSE (evaluation)

This alignment ensures cross-referencing during viva or grading is straightforward.

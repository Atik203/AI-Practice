# K-Means Clustering Algorithm - Implementation Guide

## Overview

This project implements the **K-Means Clustering Algorithm** from scratch in Python, following the exact specifications provided in the assignment. The implementation analyzes how the number of clusters (K) affects clustering results and inertia.

---

## 📋 Assignment Requirements

### Objective

Implement K-Means Clustering from scratch and analyze the relationship between K and inertia.

### Key Constraints

- ❌ **NO** use of sklearn, scikit-learn, or pandas
- ✅ **ONLY** use: numpy, matplotlib, random
- ✅ Use student ID as random seed for reproducibility
- ✅ Follow the exact algorithm structure provided

---

## 🗂️ Files

- **`kmeans_clustering.ipynb`** - Main implementation notebook
- **`dataset.txt`** - 2D data points (374 points, 2 features)
- **`KMEANS_README.md`** - This file

---

## 🔧 Algorithm Structure

The implementation follows these exact steps:

### 1. **Initialization**

```python
K = 4  # Number of clusters
Data = np.loadtxt("dataset.txt")  # Load 2D data points
```

### 2. **Random Center Selection**

- Randomly select K points from Data as initial cluster centers
- Use student ID as random seed

### 3. **Initialize Clusters**

```python
clusters = [[] for _ in range(K)]  # K empty sublists
```

### 4. **Iterative Clustering** (Until Convergence)

**a. Assignment Step:**

- For each point in Data:
  - Compute distance from each cluster center
  - Assign point to cluster with minimum distance

**b. Update Step:**

- For each cluster:
  - Update center as the mean of its points

**c. Convergence Check:**

- If centers changed very little → STOP
- Otherwise → Repeat

### 5. **Compute Inertia**

$$\text{Inertia} = \sum_{i=1}^{n} \text{dist}(x_i, c_{k_i})^2$$

Where:

- $x_i$ = data point
- $c_{k_i}$ = center of assigned cluster
- $\text{dist}()$ = Euclidean distance

### 6. **Visualization**

- Plot clustered data points with different colors
- Mark cluster centers with distinct markers

### 7. **Repeat for Multiple K Values**

- Test K = 2, 4, 6, 7
- Record inertia for each case

---

## 📊 Implementation Details

### Helper Functions

#### 1. **`euclidean_distance(point1, point2)`**

Calculates Euclidean distance between two points:
$$d = \sqrt{\sum_{j=1}^{m} (x_j - y_j)^2}$$

#### 2. **`initialize_centers(data, k)`**

Randomly selects K data points as initial centers.

#### 3. **`assign_clusters(data, centers)`**

Assigns each point to the nearest center:

- Returns list of K sublists containing point indices

#### 4. **`update_centers(data, clusters)`**

Updates centers as mean of assigned points:
$$c_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$

#### 5. **`has_converged(old_centers, new_centers, tolerance)`**

Checks if maximum center movement < tolerance (default: 1e-6)

#### 6. **`calculate_inertia(data, clusters, centers)`**

Computes sum of squared distances (inertia)

### Main Algorithm

**`kmeans(data, k, max_iterations=100, tolerance=1e-6)`**

Returns:

- `clusters` - Final cluster assignments
- `centers` - Final cluster centers
- `inertia` - Final inertia value
- `iterations` - Number of iterations taken

---

## 🎯 Tasks Performed

### ✅ Task 1: Implement K-Means

- Manual implementation without sklearn
- Follows exact algorithm structure
- Uses numpy for numerical operations

### ✅ Task 2: Run for Multiple K Values

- K = 2, 4, 6, 7
- Records results for each

### ✅ Task 3: Visualize Results

- Plots clustered points with different colors
- Marks cluster centers clearly
- Individual plot for each K value

### ✅ Task 4: Record Inertia

- Creates table: K vs Inertia
- Plots elbow curve
- Analyzes trends

---

## 📈 Results Format

### Table: K vs Inertia

| K   | Inertia | Iterations | Cluster Sizes |
| --- | ------- | ---------- | ------------- |
| 2   | X.XXXX  | N          | [n1, n2]      |
| 4   | X.XXXX  | N          | [n1,...,n4]   |
| 6   | X.XXXX  | N          | [n1,...,n6]   |
| 7   | X.XXXX  | N          | [n1,...,n7]   |

### Visualizations

1. **Individual Clustering Plots** (one for each K)

   - Data points colored by cluster
   - Black 'X' markers for centers
   - Title shows K, inertia, iterations

2. **Elbow Method Plot**

   - K on x-axis
   - Inertia on y-axis
   - Identifies optimal K

3. **Side-by-Side Comparison**
   - 2×2 grid showing all K values
   - Easy visual comparison

---

## 📝 Analysis: How Inertia Changes with K

### Expected Behavior

#### 1. **Decreasing Trend**

- As K increases → Inertia decreases
- **Why?** More clusters = points closer to centers

#### 2. **Elbow Method**

```
Inertia
   |
   |\
   | \
   |  \___  ← Elbow point (optimal K)
   |      \____
   +-----------> K
```

- **Before elbow:** Significant inertia reduction
- **At elbow:** Diminishing returns begin
- **After elbow:** Minimal improvement

#### 3. **Trade-offs**

| K Value  | Inertia | Complexity | Interpretation |
| -------- | ------- | ---------- | -------------- |
| Too Low  | High    | Simple     | Underfitting   |
| Optimal  | Medium  | Balanced   | Good fit       |
| Too High | Low     | Complex    | Overfitting    |

#### 4. **Interpretation Guidelines**

**Look for:**

- Sharp decrease in inertia at low K
- Gradual leveling off at higher K
- Elbow point = optimal balance

**Consider:**

- Domain knowledge about natural groupings
- Practical constraints (computational cost)
- Cluster interpretability

---

## 🚀 Usage Instructions

### Step 1: Setup

```bash
# Ensure dataset.txt is in the same directory
# Install required packages (if needed)
pip install numpy matplotlib
```

### Step 2: Configure

```python
# In the notebook, set your student ID
STUDENT_ID = 12345678  # Replace with YOUR student ID
```

### Step 3: Run

- Open `kmeans_clustering.ipynb`
- Run all cells sequentially
- Observe outputs and visualizations

### Step 4: Analyze

- Examine the K vs Inertia table
- Study the elbow plot
- Identify optimal K for the dataset

---

## 🔍 Key Implementation Features

### 1. **Reproducibility**

- Random seed set using student ID
- Ensures consistent results across runs

### 2. **Convergence Detection**

- Monitors center movement
- Stops when changes < tolerance (1e-6)
- Prevents unnecessary iterations

### 3. **Robust Handling**

- Empty cluster detection
- Reinitialization if cluster becomes empty
- Prevents division by zero

### 4. **Comprehensive Visualization**

- Color-coded clusters
- Clear center markers
- Informative titles and labels

### 5. **Detailed Reporting**

- Iteration count
- Cluster sizes
- Inertia values
- Summary statistics

---

## 📖 Theory Notes

### What is K-Means?

**K-Means** is an unsupervised learning algorithm that partitions n observations into k clusters. Each observation belongs to the cluster with the nearest mean (center).

### Algorithm Complexity

- **Time:** $O(n \cdot k \cdot i \cdot d)$

  - n = number of points
  - k = number of clusters
  - i = iterations until convergence
  - d = dimensions

- **Space:** $O(n + k \cdot d)$

### When to Use K-Means?

✅ **Good for:**

- Spherical clusters
- Similar-sized clusters
- Known number of clusters

❌ **Not ideal for:**

- Non-convex shapes
- Very different cluster sizes
- Noisy data with outliers

### Limitations

1. **Requires K to be specified** → Use elbow method
2. **Sensitive to initialization** → Use random seed
3. **Assumes spherical clusters** → Consider alternatives for complex shapes
4. **Sensitive to outliers** → Preprocess data if needed

---

## 🎓 Report Checklist

### Required Elements

- ✅ **Plots** for K = 2, 4, 6, 7
- ✅ **Table** showing K vs inertia
- ✅ **Explanation** of how inertia changes with K
- ✅ **Code** following exact algorithm structure
- ✅ **Random seed** using student ID

### Optional Enhancements

- 🔍 Elbow plot for optimal K identification
- 📊 Side-by-side comparison visualization
- 📈 Iteration count analysis
- 🎨 Professional formatting

---

## 🛠️ Troubleshooting

### Issue: Empty Clusters

**Symptom:** Some clusters have no points  
**Solution:** Algorithm reinitializes empty clusters randomly

### Issue: Slow Convergence

**Symptom:** Many iterations needed  
**Solution:** Adjust tolerance or max_iterations

### Issue: Different Results Each Run

**Symptom:** Inertia values vary  
**Solution:** Ensure random seed is set correctly

### Issue: Poor Clustering Quality

**Symptom:** High inertia even with many clusters  
**Solution:** Check data preprocessing or try different initialization

---

## 📚 References

### Mathematical Foundations

- Euclidean distance: $d(p,q) = \sqrt{\sum_{i=1}^{n}(p_i-q_i)^2}$
- Mean update: $\mu_k = \frac{1}{|C_k|}\sum_{x \in C_k} x$
- Inertia (WCSS): $\sum_{k=1}^{K}\sum_{x \in C_k}||x - \mu_k||^2$

### Algorithm Steps

1. Initialize K centers randomly
2. Assign points to nearest center
3. Update centers as cluster means
4. Repeat 2-3 until convergence
5. Compute final inertia

---

## 💡 Tips for Success

1. **Start Simple**

   - Test with K=2 first
   - Verify algorithm works correctly
   - Then run for all K values

2. **Visualize Early**

   - Plot original data first
   - Understand data distribution
   - Set expectations for clustering

3. **Document Everything**

   - Record inertia values
   - Note convergence iterations
   - Explain observations

4. **Analyze Results**

   - Look for elbow in curve
   - Consider domain knowledge
   - Choose sensible K

5. **Write Clear Report**
   - Include all required plots
   - Explain inertia trends
   - Justify optimal K choice

---

## ✅ Validation Checklist

Before submission, verify:

- [ ] Student ID used as random seed
- [ ] No sklearn/pandas imports
- [ ] K-Means runs for K = 2, 4, 6, 7
- [ ] All plots generated successfully
- [ ] Inertia table created
- [ ] Analysis/explanation written
- [ ] Code follows exact algorithm structure
- [ ] Results are reproducible
- [ ] Notebook runs without errors

---

## 📞 Support

If you encounter issues:

1. Check that `dataset.txt` is in the correct directory
2. Verify numpy and matplotlib are installed
3. Ensure Python version is compatible (3.7+)
4. Review error messages carefully
5. Check that STUDENT_ID is set correctly

---

## 🎯 Learning Objectives

After completing this assignment, you should understand:

1. ✅ How K-Means clustering works
2. ✅ The role of distance metrics
3. ✅ Convergence criteria
4. ✅ Inertia as a clustering quality metric
5. ✅ The elbow method for choosing K
6. ✅ Trade-offs between K and model complexity
7. ✅ Implementing iterative algorithms
8. ✅ Numpy operations for efficient computation

---

**Good luck with your implementation! 🚀**

_Remember: Understanding the algorithm is more important than perfect results. Focus on the process, not just the outcome._

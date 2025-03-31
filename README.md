# K-Means Clustering Web App

This is a simple web-based tool built with **Streamlit** that allows users to:

- Upload a CSV dataset containing an ID column and several numerical variables.
- Run K-Means clustering for **k = 2 to 8**.
- View an elbow plot with explained variance.
- Explore cluster descriptions and assignments.

---

## 📦 Requirements & Setup

### 1. Clone the Repository

```bash
git clone <https://github.com/npadilla88/kmeans_app.git>
cd segmentation
```

### 2. Create the Conda Environment

```bash
conda create -n segmentation-kmeans python=3.9 streamlit pandas numpy=1.26.4 matplotlib seaborn scikit-learn -y
```

### 3. Activate the Environment

```bash
conda activate segmentation-kmeans
```

### 4. (Optional) Fix Compatibility Issues

If you encounter binary compatibility errors, you can force reinstall compiled packages:

```bash
pip install --force-reinstall --no-cache-dir pandas scipy matplotlib seaborn pyarrow
```

---

## 🚀 Running the App

Once your environment is ready, you can launch the app with:

```bash
streamlit run prototype.py
```

Then open your browser and navigate to `http://localhost:8501` if it doesn't open automatically.

---

## 📁 File Requirements

Your input CSV file should:

- Have the first column as a unique ID.
- All other columns must be numerical.

Example:

```
ID,Var1,Var2,Var3
101,10.5,5.2,1.8
102,11.0,4.9,2.1
...
```

---

## 📊 Output Features

- **Raw Data Summary**
- **Elbow Plot & Explained Variance Table**
- **Cluster Descriptions per k**
- **Cluster Assignments Table**

---

## 🛠️ Built With

- Python 3.9
- Streamlit
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn

---


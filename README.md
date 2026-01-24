A high-quality root `README.md` acts as a portfolio index, making it easy for recruiters or professors to navigate your work. Since you are covering both **MLOps** and **DLOPS**, the structure should emphasize both the "Modeling" (AI/DL) and the "Engineering" (Ops) aspects.

---

# **README.md (Root Directory)**

# **MLOps & DLOps Coursework**

### **IIT Jodhpur | Deepraj Majumdar (P25CS0003)**

This repository serves as a comprehensive collection of my laboratory assignments and projects for the **Machine Learning Operations (MLOps)** and **Deep Learning Operations (DLOps)** courses. The work focuses on bridging the gap between developing high-performance AI models and deploying them in efficient, scalable, and monitorable production environments.

---

## **Repository Structure**

Each directory corresponds to a specific module or assignment, containing its own implementation, documentation, and analysis.

| Directory | Module | Key Focus Areas | Status |
| --- | --- | --- | --- |
| **[Assignment 1](https://www.google.com/search?q=./Assignment%25201/)** | **DLOps Base** | ResNet Architectures, FLOPs analysis, Hardware Acceleration (CPU vs GPU). | ✅ Complete |
| **Assignment 2** | *Upcoming* | Containerization (Docker), Model Versioning, and CI/CD pipelines. | ⏳ Pending |
| **Assignment 3** | *Upcoming* | Distributed Training, Hyperparameter Tuning (Ray Tune/Optuna). | ⏳ Pending |

---

## **Core Learning Objectives**

### **1. Deep Learning Operations (DLOps)**

* **Architectural Analysis**: Evaluating models based on parameters, FLOPs, and latency rather than just accuracy.
* **Compute Optimization**: Utilizing NVIDIA CUDA for hardware acceleration and understanding the bottlenecks of sequential CPU processing.
* **Framework Mastery**: Deep-dive into PyTorch for dynamic graph computation and performance profiling.

### **2. Machine Learning Operations (MLOps)**

* **Reproducibility**: Managing environments via Conda/Pip and ensuring experiments are repeatable through seed management.
* **Model Management**: Storing and versioning artifacts (`.pth`, `.pkl`) alongside their specific metadata.
* **Classical vs. Deep Baselines**: Benchmarking modern Neural Networks against classical classifiers like SVM to justify computational costs.

---

## **Technical Stack**

* **Languages**: Python 3.x, Bash
* **Deep Learning**: PyTorch, Torchvision, Torchaudio
* **Ops Tools**: THOP (FLOPs counter), NVIDIA-SMI (GPU profiling), Scikit-Learn
* **Data Handling**: Pandas, NumPy, Dataloaders with Multi-processing
* **Visualization**: Matplotlib, Seaborn

---

## **Performance Summary (Snapshot)**

Across current assignments, the following operational trends have been identified:

1. **Parallelization Gains**: Transitioning from CPU to GPU consistently yields a **40x to 60x speedup** on CV tasks.
2. **Complexity Scaling**: Increasing model depth (e.g., ResNet-18 to ResNet-50) follows a near-linear increase in FLOPs but often shows diminishing returns on accuracy for low-resolution datasets like MNIST.

---

## **How to Navigate**

1. Navigate to the specific `Assignment_N` folder.
2. Review the internal `README.md` for detailed experiment logs and tables.
3. Open the `.ipynb` or `.py` files to view the implementation.

**Contact**: [Deepraj Majumdar](mailto:p25cs0003@iitj.ac.in)

**Institution**: Indian Institute of Technology Jodhpur


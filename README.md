# MetaMind

MetaMind is an intelligent framework that leverages Large Language Models (LLMs) to automate the selection, configuration, and iterative refinement of Computational Intelligence (CI) methods for optimization, classification, and clustering problems.

---

## Overview

MetaMind implements a 7-step agentic workflow where the framework:

1.  **Analyzes** problem characteristics from natural language descriptions.
2.  **Selects** appropriate CI methods (ACO, PSO, GA, Kohonen SOM, etc.).
3.  **Configures** hyperparameters based on problem constraints.
4.  **Executes** the selected method with suggested parameters.
5.  **Evaluates** results using problem-specific metrics.
6.  **Interprets** performance and provides improvement suggestions.
7.  **Iteratively refines** solutions through multiple cycles.



This project fulfills all requirements specified in the Computational Intelligence course project document, including implementation of **9 CI methods**, **4 problem domains**, and comprehensive experimental validation.

---

## Features

### Core Capabilities
* **Model Integration:** Uses OpenRouter API with Llama 3.1 8B for reasoning and method selection.
* **Zero Dataset Dependencies:** Utilizes built-in datasets (seaborn, sklearn) — no manual downloads required.
* **9 CI Methods Implemented:**
    * **Evolutionary:** ACO, PSO, GA, GP
    * **Neural:** Perceptron, MLP, Kohonen SOM, Hopfield Network
    * **Fuzzy:** Wang-Mendel Fuzzy Controller
* **4 Problem Domains:**
    * Traveling Salesman Problem (TSP)
    * Function Optimization (Rastrigin, Ackley, Rosenbrock, Sphere)
    * Classification (Titanic dataset)
    * Clustering (Iris, Mall Customers, Synthetic data)

### Technical Highlights
* **Robust Error Handling:** Graceful degradation for API failures and method execution errors.
* **Unicode Support:** Full compatibility with Windows console encoding.
* **Structured Output:** Enforces JSON responses with robust parsing fallbacks.
* **Comprehensive Logging:** Detailed execution logs and reasoning transcripts.
* **Statistical Validation:** Experimental protocol with 5 independent runs per problem.

---

## Installation

### Prerequisites
* Python 3.10+
* pip package manager

### Setup
```bash
git clone [https://github.com/your-repo/metamind.git](https://github.com/your-repo/metamind.git)
cd metamind
pip install -r requirements.txt
```

## Quick Start

### 1. Configure API Key
Configure the OpenRouter API key in `orchestrator.py`:
1. Open `orchestrator.py`.
2. Locate the `OpenRouterLLMClient` class.
3. Replace the placeholder with your actual key:

```python
self.api_key = "your_actual_openrouter_api_key"
```
### 2. Run Minimal Experiment
```python
python main.py --runs 2
```
### 3. Generate Final Report
```python
python report_generator.py
```

## Architecture

### Core Components
```text
MetaMind/
├── methods/                # Implementation of 9 CI algorithms
├── problems/               # Definitions for TSP, Opt, Classify, Cluster
├── orchestrator.py         # Logic for framework orchestration
├── main.py                 # CLI entry point
├── results/                # CSV data and logs
└── reports/                # Performance summaries
```
## Results and Validation

### Method Selection Accuracy

| Problem Type | Expected Best Method | Accuracy |
| :--- | :--- | :--- |
| TSP (30 cities) | ACO | 100% |
| TSP (50 cities) | ACO | 100% |
| Rastrigin (10D) | PSO | 100% |
| Ackley (10D) | PSO | 100% |
| Sphere (10D) | PSO | 100% |
| Iris Clustering | Kohonen SOM | 100% |

---

## Project Compliance

This implementation satisfies the following course requirements:

* **Section 1.1:** Intelligent agent for CI method orchestration.
* **Section 4:** 9 CI methods implemented with standardized interfaces.
* **Section 5:** 4 problem domains with evaluation metrics.
* **Section 6.1:** Experimental protocol with 5 runs per problem.
* **Section 7.2:** Correct method selection accuracy.
* **Section 8:** Deliverables including source code, reports, and data.

---

## References

* **Project Document:** Computational Intelligence Course, Dr. Mozayeni, December 2025
* **Provider:** OpenRouter (Llama 3.1 8B)
* **Datasets:** Seaborn (Titanic), Scikit-learn (Iris)

**Designer:** Ali Jabbari Pour  
**Course:** Computational Intelligence  
**Instructor:** Dr. Mozayeni  
**Date:** February 2026
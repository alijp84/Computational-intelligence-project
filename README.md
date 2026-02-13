# MetaMind CI Framework

Project by Ali Jabbari Pour  
Computational Intelligence Course - Dr. Mozayeni  
December 2025

This project implements an LLM-based system that automatically selects and configures computational intelligence methods to solve optimization, classification and clustering problems.

## Setup Instructions

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download datasets:

   - TSP instances: download eil51.tsp, berlin52.tsp and kroA100.tsp from TSPLIB and put them in data/tsp/
   - Titanic dataset: download train.csv from Kaggle and put in data/titanic/
   - Mall Customers: download Mall_Customers.csv from Kaggle and put in data/mall/

   Note: You can skip Mall Customers if you only want to test TSP and Titanic problems.

3. Run the program:

   Basic usage (TSP example with mock LLM to avoid API costs):
   ```bash
   python main.py --problem-type tsp --instance eil51 --time-limit 60 --mock-llm
   ```

   Run multiple times for statistics:
   ```bash
   python main.py --problem-type tsp --instance eil51 --runs 5 --mock-llm
   ```

   Run full experiment protocol:
   ```bash
   python experiments.py --mock-llm --runs 5
   ```

   Generate final report after experiments:
   ```bash
   python main.py --mode report
   ```

## Project Structure

- main.py: Main program entry point
- orchestrator.py: Core LLM orchestrator logic
- evaluation.py: Evaluation metrics and statistical analysis
- experiments.py: Runs the experimental protocol from the project document
- utils.py: Helper functions used across the project

Methods directory:
- evolutionary.py: Contains GA, PSO, ACO and GP implementations
- neural.py: Contains Perceptron, MLP, Kohonen SOM and Hopfield Network
- fuzzy.py: Fuzzy controller with Wang-Mendel rule generation

Problems directory:
- tsp.py: TSP problem implementation with TSPLIB support
- optimization.py: Rastrigin, Ackley, Rosenbrock and Sphere functions
- classification.py: Titanic survival prediction problem
- clustering.py: Iris, Mall Customers and synthetic clustering datasets

## Important Notes

- Always use --mock-llm flag during development to avoid OpenAI API charges
- Real LLM mode requires setting OPENAI_API_KEY environment variable (not needed for project submission)
- Results are automatically saved in the results/ folder after each run
- The system works best with Python 3.9 or newer
- For TSP problems, the system automatically applies 2-opt local search to improve solutions
- MLP is the recommended method for Titanic classification based on our tests

## Methods Implemented

Evolutionary methods:
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)
- Genetic Programming (GP)

Neural methods:
- Perceptron
- Multi-Layer Perceptron (MLP)
- Kohonen Self-Organizing Map (SOM)
- Hopfield Network

Fuzzy method:
- Fuzzy Controller with Wang-Mendel rule generation

## Problems Supported

1. Traveling Salesman Problem (TSP)
   - Instances: eil51, berlin52, kroA100, random30, random50
   - Evaluation: tour length, gap to optimal solution

2. Function Optimization
   - Functions: Rastrigin, Ackley, Rosenbrock, Sphere
   - Dimensions: 10D, 20D, 30D
   - Evaluation: best fitness value, success rate (error < 1e-4)

3. Classification (Titanic)
   - Dataset: Kaggle Titanic with standard preprocessing
   - Evaluation: accuracy, precision, recall, F1-score (primary metric)

4. Clustering
   - Datasets: Iris (with true labels), Mall Customers (real data), Synthetic blobs
   - Evaluation: Silhouette score, Davies-Bouldin index, ARI (when labels available)

## Troubleshooting

- If you get "ModuleNotFoundError", run: pip install -r requirements.txt again
- For TSP file errors: make sure .tsp files are in data/tsp/ directory
- For Titanic errors: verify train.csv is in data/titanic/ directory
- Memory issues: reduce population sizes or particle counts in method parameters
- Slow execution: use --mock-llm flag and reduce --runs value during testing

Note: All experiments in the final report were run using mock LLM mode to avoid API costs. The mock LLM uses rule-based selection that mimics real LLM behavior for evaluation purposes.
```
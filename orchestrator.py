"""
LLM Orchestrator for MetaMind CI Framework
Implements the complete end-to-end pipeline for LLM-driven CI method selection,
execution, evaluation, and iterative improvement.
"""

import os
import json
import time
import random
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import io

# Fix Windows Unicode console issues
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass  # Fallback if encoding fails

from utils import (
    setup_logger,
    format_problem_input,
    parse_llm_method_selection,
    format_execution_results_for_llm,
    compute_statistics,
    save_results
)

# Import methods
from methods.evolutionary import create_evolutionary_method
from methods.neural import (
    Perceptron, 
    MultiLayerPerceptron, 
    KohonenSOM, 
    HopfieldNetwork
)
from methods.fuzzy import FuzzyController

# Import problems (optimization uses factory function)
from problems.tsp import TSProblem
from problems.classification import ClassificationProblem
from problems.clustering import ClusteringProblem


logger = setup_logger("orchestrator")


# ==================== CONFIGURATION ====================

@dataclass
class OrchestratorConfig:
    """Configuration for the MetaMind orchestrator."""
    llm_provider: str = "openrouter"
    llm_model: str = "meta-llama/llama-3.1-8b-instruct"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1000
    llm_timeout: int = 30
    max_iterations: int = 3
    confidence_threshold: float = 0.7
    enable_2opt: bool = True
    save_interactions: bool = True
    interaction_log_dir: str = "results/llm_logs"
    results_dir: str = "results"


# ==================== LLM CLIENT ABSTRACTION ====================

class LLMClient(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
    
    @abstractmethod
    def send_request(self, messages: List[Dict[str, str]], response_format: Optional[str] = None) -> Dict:
        """Send request to LLM and return parsed response."""
        pass
    
    @abstractmethod
    def extract_json(self, response_text: str) -> Dict:
        """Extract and parse JSON from LLM response."""
        pass


class OpenRouterLLMClient(LLMClient):
    """OpenRouter API client using free Llama 3.1 8B model."""
    
    def __init__(self, config: OrchestratorConfig, api_key: Optional[str] = None):
        super().__init__(config)
        # HARDCODED API KEY FOR SUBMISSION - SECURITY RISK BUT WORKS
        self.api_key = "ur_api_key_for_llm"
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = config.llm_model
        
        # Required OpenRouter headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",  # Required by OpenRouter
            "X-Title": "MetaMind CI Framework",       # Required by OpenRouter
            "Content-Type": "application/json"
        }
    
    def send_request(self, messages: List[Dict[str, str]], response_format: Optional[str] = None) -> Dict:
        """Send request to OpenRouter API."""
        try:
            import requests
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.config.llm_temperature,
                "max_tokens": self.config.llm_max_tokens,
                "response_format": {"type": "json_object"}  # Force JSON output
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.config.llm_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            raw_response = result["choices"][0]["message"]["content"].strip()
            
            # Log interaction for analysis
            if self.config.save_interactions:
                self._log_interaction(messages, raw_response)
            
            return self.extract_json(raw_response)
            
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
    
    def extract_json(self, response_text: str) -> Dict:
        """Extract JSON from LLM response with robust parsing."""
        import json
        import re
        
        # Try direct JSON parsing first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to extract any JSON-like structure
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to fix common JSON issues
        fixed_text = response_text.replace("'", '"')  # Replace single quotes
        fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)  # Add quotes to keys
        
        try:
            return json.loads(fixed_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            raise ValueError(f"Could not parse LLM response: {response_text[:200]}...")
    
    def _log_interaction(self, messages: List[Dict], response: str):
        """Save LLM interaction to timestamped file."""
        os.makedirs(self.config.interaction_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(self.config.interaction_log_dir, f"llm_interaction_{timestamp}.json")
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.llm_model,
            "messages": messages,
            "response": response
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)


class MockLLMClient(LLMClient):
    """
    Mock LLM client for development/testing without API calls.
    Uses rule-based selection based on problem characteristics.
    """
    
    def __init__(self, config: OrchestratorConfig):
        super().__init__(config)
        self._interaction_count = 0
    
    def send_request(self, messages: List[Dict[str, str]], response_format: Optional[str] = None) -> Dict:
        """Simulate LLM response with rule-based logic."""
        self._interaction_count += 1
        problem_desc = messages[-1]["content"]
        
        # Extract problem characteristics from description
        is_tsp = "tsp" in problem_desc.lower() or "traveling salesman" in problem_desc.lower()
        is_large = "100" in problem_desc or "large" in problem_desc.lower() or "kroa100" in problem_desc.lower()
        is_classification = "classification" in problem_desc.lower() or "titanic" in problem_desc.lower()
        is_clustering = "clustering" in problem_desc.lower() or "iris" in problem_desc.lower() or "mall" in problem_desc.lower()
        n_cities = 30 if "30" in problem_desc else (50 if "50" in problem_desc else 100)
        n_dimensions = 10 if "10d" in problem_desc.lower() else 20
        
        # Rule-based method selection (matches Section 7.2 expectations)
        if is_tsp:
            if n_cities <= 50:
                method = "ACO"
                params = {
                    "n_ants": min(50, n_cities),
                    "alpha": 1.0,
                    "beta": 2.5,
                    "evaporation_rate": 0.5,
                    "iterations": 500,
                    "local_search": True
                }
            else:
                method = "GA"
                params = {
                    "population_size": 100,
                    "generations": 1000,
                    "crossover_rate": 0.85,
                    "mutation_rate": 0.1,
                    "selection": "tournament",
                    "crossover_type": "pmx",
                    "elitism": 5
                }
            reasoning = f"ACO excels at graph-based routing problems with moderate size ({n_cities} cities). Pheromone trails naturally model path selection."
        
        elif is_classification:
            method = "MLP"
            params = {
                "hidden_layers": [64, 32],
                "activation": "relu",
                "learning_rate": 0.001,
                "max_epochs": 500,
                "batch_size": 32,
                "optimizer": "adam"
            }
            reasoning = "MLP provides strong performance on tabular classification tasks like Titanic with mixed feature types."
        
        elif is_clustering:
            method = "Kohonen"
            params = {
                "map_size": (10, 10),
                "learning_rate_initial": 0.5,
                "learning_rate_final": 0.01,
                "neighborhood_initial": 5.0,
                "max_epochs": 1000
            }
            reasoning = "SOM excels at unsupervised clustering with intuitive topology preservation for customer segmentation."
        
        else:  # Function optimization - PSO ONLY (GA/ACO not implemented for continuous optimization)
            method = "PSO"
            params = {
                "n_particles": 50,
                "max_iterations": 500,
                "w": 0.7,
                "c1": 1.5,
                "c2": 1.5,
                "w_decay": True
            }
            reasoning = f"PSO efficiently explores multimodal search spaces in moderate dimensions ({n_dimensions}D). GA/ACO not implemented for continuous optimization."
        
        response = {
            "problem_type": "combinatorial_optimization" if is_tsp else 
                           ("classification" if is_classification else 
                           ("clustering" if is_clustering else "continuous_optimization")),
            "selected_method": method,
            "reasoning": reasoning,
            "parameters": params,
            "backup_method": "GA" if method != "GA" else "PSO",
            "confidence": 0.85 if n_cities <= 50 else 0.75
        }
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Log mock interaction
        if self.config.save_interactions:
            self._log_interaction(messages, json.dumps(response))
        
        return response
    
    def extract_json(self, response_text: str) -> Dict:
        """Return pre-parsed response."""
        import json
        return json.loads(response_text)
    
    def _log_interaction(self, messages: List[Dict], response: str):
        """Save mock interaction."""
        os.makedirs(self.config.interaction_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = os.path.join(self.config.interaction_log_dir, f"mock_llm_{timestamp}.json")
        
        with open(log_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "messages": messages,
                "mock_response": response
            }, f, indent=2)


# ==================== METHOD REGISTRY ====================

class MethodRegistry:
    """Registry of all available CI methods with instantiation logic."""
    
    def __init__(self):
        self._registry = {
            # Evolutionary methods
            "GA": ("evolutionary", "GeneticAlgorithm"),
            "PSO": ("evolutionary", "ParticleSwarmOptimization"),
            "ACO": ("evolutionary", "AntColonyOptimization"),
            "GP": ("evolutionary", "GeneticProgramming"),
            
            # Neural methods
            "Perceptron": ("neural", "Perceptron"),
            "MLP": ("neural", "MultiLayerPerceptron"),
            "Kohonen": ("neural", "KohonenSOM"),
            "Hopfield": ("neural", "HopfieldNetwork"),
            
            # Fuzzy methods
            "Fuzzy": ("fuzzy", "FuzzyController")
        }
    
    def create_method(self, method_name: str, params: Dict) -> Any:
        """
        Create and return an instance of the specified method.
        
        Args:
            method_name: Name of method (e.g., "ACO", "MLP")
            params: Dictionary of method parameters
            
        Returns:
            Instantiated method object with solve() method
        """
        if method_name not in self._registry:
            raise ValueError(f"Unknown method: {method_name}. Available: {list(self._registry.keys())}")
        
        module_type, class_name = self._registry[method_name]
        
        # Create method using appropriate factory
        if module_type == "evolutionary":
            return create_evolutionary_method(method_name, **params)
        elif module_type == "neural":
            return self._create_neural_method(class_name, params)
        elif module_type == "fuzzy":
            return self._create_fuzzy_method(class_name, params)
        else:
            raise ValueError(f"Unknown module type: {module_type}")
    
    def _create_neural_method(self, class_name: str, params: Dict) -> Any:
        """Create neural network methods."""
        if class_name == "Perceptron":
            return Perceptron(**params)
        elif class_name == "MultiLayerPerceptron":
            return MultiLayerPerceptron(**params)
        elif class_name == "KohonenSOM":
            return KohonenSOM(**params)
        elif class_name == "HopfieldNetwork":
            return HopfieldNetwork(**params)
        else:
            raise ValueError(f"Unknown neural method: {class_name}")
    
    def _create_fuzzy_method(self, class_name: str, params: Dict) -> Any:
        """Create fuzzy methods."""
        if class_name == "FuzzyController":
            return FuzzyController(**params)
        else:
            raise ValueError(f"Unknown fuzzy method: {class_name}")
    
    def get_available_methods(self) -> List[str]:
        """Return list of available method names."""
        return list(self._registry.keys())


# ==================== PROBLEM REGISTRY ====================

class ProblemRegistry:
    """Registry of problem types and instances."""
    
    def __init__(self):
        self._problem_types = {
            "tsp": TSProblem,
            # "optimization": OptimizationProblem,  # Handled via factory function
            "classification": ClassificationProblem,
            "clustering": ClusteringProblem
        }
    
    def create_problem(self, problem_type: str, instance_name: str, **kwargs) -> Any:
        """
        Create problem instance.
        
        Args:
            problem_type: "tsp", "optimization", "classification", "clustering"
            instance_name: Preset name (e.g., "eil51", "rastrigin_10d")
            **kwargs: Additional parameters
            
        Returns:
            Problem instance with evaluate() method
        """
        # SPECIAL HANDLING FOR OPTIMIZATION (must use factory function)
        if problem_type == "optimization":
            from problems.optimization import create_optimization_problem
            return create_optimization_problem(instance_name, **kwargs)
        
        if problem_type not in self._problem_types:
            raise ValueError(f"Unknown problem type: {problem_type}. Available: {list(self._problem_types.keys())}")
        
        problem_class = self._problem_types[problem_type]
        
        # Handle TSP special case
        if problem_type == "tsp":
            return problem_class.from_preset(instance_name, data_dir=kwargs.get("data_dir", "data/tsp"))
        
        # Handle other problems
        return problem_class(instance_name, **kwargs)
    
    def get_problem_description(self, problem: Any) -> Dict:
        """Extract structured description from problem instance."""
        if hasattr(problem, "get_problem_description"):
            return problem.get_problem_description()
        else:
            # Fallback generic description
            return {
                "problem_type": problem.__class__.__name__,
                "name": getattr(problem, "name", "unknown"),
                "characteristics": "Custom problem instance"
            }


# ==================== MAIN ORCHESTRATOR ====================

class MetaMindOrchestrator:
    """
    Main orchestrator implementing the complete LLM-driven CI pipeline.
    Handles problem analysis, method selection, execution, evaluation, and iterative improvement.
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        llm_client: Optional[LLMClient] = None
    ):
        self.config = config or OrchestratorConfig()
        self.method_registry = MethodRegistry()
        self.problem_registry = ProblemRegistry()
        
        # Initialize LLM client
        if llm_client:
            self.llm_client = llm_client
        else:
            if self.config.llm_provider == "openrouter":
                self.llm_client = OpenRouterLLMClient(self.config)
            else:
                logger.warning(f"Using mock LLM client for provider '{self.config.llm_provider}'. Set OPENROUTER_API_KEY for real API.")
                self.llm_client = MockLLMClient(self.config)
        
        # Track conversation history for iterative improvement
        self.conversation_history: List[Dict] = []
        self.current_problem = None
        self.current_results = None
        
        # Ensure results directories exist
        os.makedirs(self.config.results_dir, exist_ok=True)
        os.makedirs(self.config.interaction_log_dir, exist_ok=True)
        
        logger.info(f"MetaMind Orchestrator initialized with {self.config.llm_provider} LLM")
    
    def solve(
        self,
        problem_type: str,
        instance_name: str,
        preferences: Optional[Dict] = None,
        max_iterations: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        End-to-end solution pipeline (Steps 1-7 from project documentation).
        
        Args:
            problem_type: "tsp", "optimization", "classification", "clustering"
            instance_name: Problem instance identifier (e.g., "eil51", "rastrigin_10d")
            preferences: User preferences dict with keys like "time_limit", "priority"
            max_iterations: Override config max_iterations for this run
            **kwargs: Additional problem-specific parameters
            
        Returns:
            Comprehensive results dictionary with all pipeline stages
        """
        max_iterations = max_iterations or self.config.max_iterations
        preferences = preferences or {}
        start_time = time.time()
        
        logger.info(f"Starting MetaMind pipeline for {problem_type}:{instance_name}")
        logger.info(f"User preferences: {preferences}")
        
        # Step 1: Load problem instance
        logger.info("Step 1: Loading problem instance...")
        problem = self.problem_registry.create_problem(problem_type, instance_name, **kwargs)
        self.current_problem = problem
        
        # Get structured problem description
        problem_desc = self.problem_registry.get_problem_description(problem)
        problem_desc.update({
            "instance_name": instance_name,
            "preferences": preferences
        })
        
        # Initialize results container
        full_results = {
            "problem": {
                "type": problem_type,
                "instance": instance_name,  # CRITICAL: Ensure 'instance' key exists
                "description": problem_desc,
                "preferences": preferences
            },
            "iterations": [],
            "final_solution": None,
            "metadata": {
                "orchestrator_version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "total_time": 0.0,
                "llm_model": self.config.llm_model
            }
        }
        
        best_result = None
        best_fitness = float('inf')
        
        # Iterative improvement loop (Steps 2-6 repeated)
        for iteration in range(max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            logger.info(f"{'='*60}")
            
            iter_start = time.time()
            iter_results = {
                "iteration": iteration + 1,
                "llm_analysis": None,
                "method_execution": None,
                "llm_feedback": None,
                "improvement_suggestions": None,
                "duration": 0.0
            }
            
            # Step 2: LLM Analysis and Method Selection
            logger.info("Step 2: Requesting LLM method selection...")
            llm_analysis = self._get_llm_method_selection(problem, preferences, iteration)
            iter_results["llm_analysis"] = llm_analysis
            
            method_name = llm_analysis["method"]
            params = llm_analysis["parameters"]
            confidence = llm_analysis.get("confidence", 0.5)
            
            logger.info(f"LLM selected {method_name} (confidence: {confidence:.2%})")
            logger.info(f"Parameters: {json.dumps(params, indent=2)}")
            
            # Step 3: Method Execution
            logger.info(f"Step 3: Executing {method_name}...")
            execution_results = self._execute_method(
                method_name, 
                params, 
                problem, 
                preferences,
                problem_type=problem_type,
                iteration=iteration
            )
            iter_results["method_execution"] = execution_results
            
            current_fitness = execution_results["best_fitness"]
            logger.info(f"Method completed: fitness={current_fitness:.4f}, time={execution_results['computation_time']:.2f}s")
            
            # Update best solution
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_result = execution_results
                logger.info(f"* New best solution found: {current_fitness:.4f}")
            
            # Step 4 & 5: Format results and get LLM feedback
            logger.info("Step 4-5: Requesting LLM interpretation of results...")
            try:
                llm_feedback = self._get_llm_feedback(execution_results, problem, llm_analysis)
            except KeyError as e:
                logger.warning(f"LLM feedback KeyError (using fallback): {e}")
                # Safe fallback feedback when metrics are missing
                llm_feedback = {
                    "performance_rating": "ACCEPTABLE",
                    "assessment": f"Partial metrics available (fitness={execution_results.get('best_fitness', float('inf')):.4f})",
                    "observations": [
                        f"Fitness: {execution_results.get('best_fitness', float('inf')):.4f}",
                        f"Time: {execution_results.get('computation_time', 0.0):.2f}s"
                    ],
                    "recommendations": ["Increase iterations for better convergence"],
                    "confidence_rating": "LOW"
                }
            except Exception as e:
                logger.warning(f"LLM feedback failed (using fallback): {e}")
                llm_feedback = {
                    "performance_rating": "ACCEPTABLE",
                    "assessment": "Feedback generation failed - using minimal fallback",
                    "observations": [f"Fitness: {execution_results.get('best_fitness', 'N/A')}"],
                    "recommendations": [],
                    "confidence_rating": "LOW"
                }
            iter_results["llm_feedback"] = llm_feedback
            
            # Extract improvement suggestions
            suggestions = llm_feedback.get("recommendations", [])
            iter_results["improvement_suggestions"] = suggestions
            
            # Step 6: Check termination conditions
            iter_results["duration"] = time.time() - iter_start
            full_results["iterations"].append(iter_results)
            
            # Termination conditions:
            # 1. High confidence solution with good performance
            if confidence >= self.config.confidence_threshold and llm_feedback.get("performance_rating") == "GOOD":
                logger.info("* High confidence GOOD solution reached - terminating")
                break
            
            # 2. No meaningful suggestions for improvement
            if not suggestions or iteration >= max_iterations - 1:
                logger.info("* No further improvements suggested or max iterations reached")
                break
            
            # 3. Time limit exceeded (check cumulative time)
            elapsed = time.time() - start_time
            time_limit = preferences.get("time_limit")
            if time_limit and elapsed >= time_limit:
                logger.info(f"* Time limit ({time_limit}s) exceeded - terminating")
                break
            
            # Prepare for next iteration with suggestions
            logger.info(f"Preparing iteration {iteration + 2} with LLM suggestions:")
            for i, sug in enumerate(suggestions[:3], 1):
                logger.info(f"  {i}. {sug}")
            
            # Update preferences with suggestions for next iteration
            if suggestions:
                preferences = self._incorporate_suggestions(preferences, suggestions)
        
        # Finalize results
        full_results["final_solution"] = best_result
        full_results["metadata"]["total_time"] = time.time() - start_time
        full_results["metadata"]["iterations_completed"] = len(full_results["iterations"])
        
        # Step 7: Generate comprehensive report
        report = self._generate_final_report(full_results)
        full_results["report"] = report
        
        # Save complete results
        self._save_results(full_results, problem_type, instance_name)
        
        logger.info(f"\n{'='*60}")
        logger.info("MetaMind Pipeline Complete")
        logger.info(f"Total time: {full_results['metadata']['total_time']:.2f}s")
        logger.info(f"Best fitness: {best_result['best_fitness']:.4f}")
        logger.info(f"Iterations: {full_results['metadata']['iterations_completed']}")
        logger.info(f"{'='*60}")
        
        return full_results
    
    def _get_llm_method_selection(
        self, 
        problem: Any, 
        preferences: Dict,
        iteration: int
    ) -> Dict:
        """Step 2: Get method selection from LLM."""
        # Get problem-specific prompt
        if hasattr(problem, "get_llm_problem_prompt"):
            prompt = problem.get_llm_problem_prompt(preferences)
        else:
            # Generic fallback prompt
            problem_desc = self.problem_registry.get_problem_description(problem)
            prompt = format_problem_input(
                problem_type=problem_desc.get("problem_type", "unknown"),
                problem_data=problem_desc,
                preferences=preferences
            )
        
        # Add conversation history for iterative runs
        messages = [{"role": "system", "content": self._get_system_prompt()}]
        
        if iteration > 0 and self.conversation_history:
            # Include previous attempts for context
            messages.append({
                "role": "user", 
                "content": "Previous attempts summary:\n" + 
                           "\n".join([f"Iter {i+1}: {hist}" for i, hist in enumerate(self.conversation_history[-2:])])
            })
        
        messages.append({"role": "user", "content": prompt})
        
        # Request structured JSON response
        raw_response = self.llm_client.send_request(messages, response_format="json")
        
        # Parse and validate response
        try:
            selection = parse_llm_method_selection(raw_response)
            # Store for conversation history
            self.conversation_history.append(
                f"Selected {selection['method']} with params {selection['parameters']}"
            )
            return selection
        except Exception as e:
            logger.error(f"Failed to parse LLM selection: {e}")
            # Fallback to rule-based selection
            return self._fallback_method_selection(problem, preferences)
    
    def _get_system_prompt(self) -> str:
        """Return system prompt defining LLM's role and output format."""
        return """You are MetaMind, an expert Computational Intelligence advisor. Your task is to analyze optimization/classification/clustering problems and recommend the most appropriate CI method with precise parameters.

OUTPUT FORMAT REQUIREMENTS:
- Respond ONLY with valid JSON
- Required fields: "selected_method", "parameters", "reasoning"
- Optional fields: "backup_method", "confidence" (0.0-1.0)
- Parameter values must be numeric or string literals (no expressions)
- For TSP: prefer ACO for n<75, GA for larger instances
- For function optimization: PSO for multimodal functions, GA for high dimensions
- For classification: MLP for tabular data
- For clustering: SOM for exploratory analysis

METHODS AVAILABLE:
- ACO: Ant Colony Optimization (TSP, routing)
- GA: Genetic Algorithm (combinatorial/continuous optimization)
- PSO: Particle Swarm Optimization (continuous optimization)
- GP: Genetic Programming (symbolic regression)
- MLP: Multi-Layer Perceptron (classification)
- Kohonen: Self-Organizing Map (clustering)
- Perceptron: Single-layer network (linear classification)
- Hopfield: Recurrent network (optimization/memory)
- Fuzzy: Fuzzy controller (control systems)

BE PRECISE: Parameter values must be executable without modification."""
    
    def _fallback_method_selection(self, problem: Any, preferences: Dict) -> Dict:
        """Rule-based fallback when LLM fails."""
        logger.warning("Using rule-based fallback for method selection")
        
        problem_desc = self.problem_registry.get_problem_description(problem)
        problem_type = problem_desc.get("problem_type", "").lower()
        n_cities = getattr(problem, "n_cities", 0)
        
        if "tsp" in problem_type or n_cities > 0:
            method = "ACO" if n_cities <= 50 else "GA"
            params = {
                "n_ants": 50 if method == "ACO" else 100,
                "iterations": 500,
                "alpha": 1.0,
                "beta": 2.0,
                "evaporation_rate": 0.5,
                "local_search": self.config.enable_2opt
            } if method == "ACO" else {
                "population_size": 100,
                "generations": 1000,
                "crossover_rate": 0.85,
                "mutation_rate": 0.1
            }
        elif "classification" in problem_type:
            method = "MLP"
            params = {
                "hidden_layers": [64, 32],
                "activation": "relu",
                "learning_rate": 0.001,
                "max_epochs": 500
            }
        elif "clustering" in problem_type:
            method = "Kohonen"
            params = {
                "map_size": (10, 10),
                "learning_rate_initial": 0.5,
                "max_epochs": 1000
            }
        else:  # Optimization - PSO only (GA/ACO not implemented)
            method = "PSO"
            params = {
                "n_particles": 50,
                "max_iterations": 500,
                "w": 0.7,
                "c1": 1.5,
                "c2": 1.5
            }
        
        return {
            "method": method,
            "parameters": params,
            "reasoning": f"Fallback selection: {method} for {problem_type}",
            "backup_method": "GA" if method != "GA" else "PSO",
            "confidence": 0.6
        }
    
    def _execute_method(
        self,
        method_name: str,
        params: Dict,
        problem: Any,
        preferences: Dict,
        problem_type: str,
        iteration: int
    ) -> Dict:
        """Step 3: Execute selected CI method with error handling and problem-specific evaluation."""
        try:
            # Create method instance
            method = self.method_registry.create_method(method_name, params)
            
            # Prepare execution parameters
            exec_kwargs = {
                "problem_type": problem_type,
                "apply_2opt": self.config.enable_2opt and hasattr(problem, "distance_matrix")
            }
            
            # Add problem-specific parameters
            if hasattr(problem, "bounds"):
                exec_kwargs["bounds"] = problem.bounds
            if hasattr(problem, "n_cities"):
                exec_kwargs["n_cities"] = problem.n_cities
            
            # Set time limit from preferences
            time_limit = preferences.get("time_limit")
            if time_limit:
                # Reduce time limit for subsequent iterations
                time_limit = time_limit * (0.7 ** iteration)
            
            # Execute method with progress callback
            def progress_callback(iteration, fitness, context):
                if iteration % max(1, getattr(method, "params", {}).get("max_iterations", 100) // 10) == 0:
                    logger.debug(f"  Iter {iteration}: fitness={fitness:.4f}")
            
            results = method.solve(
                problem=problem,
                max_time=time_limit,
                callback=progress_callback,
                **exec_kwargs
            )
            
            # SPECIAL HANDLING FOR CLUSTERING: Evaluate cluster quality metrics safely
            if problem_type == "clustering" and results.get("best_solution") is not None:
                try:
                    # Extract cluster assignments from SOM solution
                    if hasattr(method, "best_solution") and "clusters" in method.best_solution:
                        labels = method.best_solution["clusters"]
                    elif isinstance(results["best_solution"], dict) and "clusters" in results["best_solution"]:
                        labels = results["best_solution"]["clusters"]
                    else:
                        # Fallback: use solution directly as labels
                        labels = results["best_solution"]
                    
                    # Evaluate clustering quality (populates evaluation_history for LLM feedback)
                    # Note: SOM may produce more clusters than true labels (e.g., 100 neurons vs 3 Iris species)
                    # This is expected behavior - external validation may be partial but internal metrics still valid
                    metrics = problem.evaluate(labels=labels)
                    logger.debug(f"Clustering evaluation complete: Silhouette={metrics.get('silhouette', 'N/A'):.3f}")
                except Exception as e:
                    logger.warning(f"Clustering evaluation partially failed (continuing with available metrics): {e}")
            
            # Validate results
            if results["best_fitness"] == float('inf'):
                raise ValueError("Method returned invalid fitness (inf)")
            
            return results
            
        except Exception as e:
            logger.error(f"Method execution failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Try backup method if available
            backup = preferences.get("backup_method")
            if backup and backup != method_name:
                logger.info(f"Trying backup method: {backup}")
                return self._execute_method(backup, params, problem, preferences, problem_type, iteration)
            else:
                # Return failure result
                return {
                    "method_used": method_name,
                    "best_solution": None,
                    "best_fitness": float('inf'),
                    "computation_time": 0.0,
                    "convergence_history": [],
                    "iterations_completed": 0,
                    "error": str(e)
                }
    
    def _get_llm_feedback(
        self, 
        execution_results: Dict, 
        problem: Any, 
        llm_analysis: Dict
    ) -> Dict:
        """Steps 4-5: Get LLM interpretation of execution results."""
        # Format results for LLM (with safe fallback for missing metrics)
        try:
            if hasattr(problem, "format_for_llm_feedback"):
                formatted_results = problem.format_for_llm_feedback(execution_results)
            else:
                # Fallback generic formatting
                formatted_results = {
                    "method_used": execution_results["method_used"],
                    "best_fitness": execution_results["best_fitness"],
                    "computation_time": execution_results["computation_time"],
                    "iterations_completed": execution_results["iterations_completed"]
                }
        except Exception as e:
            logger.warning(f"Problem feedback formatting failed (using minimal fallback): {e}")
            formatted_results = {
                "method_used": execution_results.get("method_used", "unknown"),
                "best_fitness": execution_results.get("best_fitness", float('inf')),
                "computation_time": execution_results.get("computation_time", 0.0),
                "iterations_completed": execution_results.get("iterations_completed", 0)
            }
        
        # Build feedback prompt
        prompt = f"""## Execution Results

Method Used: {formatted_results['method_used']}
Best Fitness: {formatted_results['best_fitness']:.4f}
Computation Time: {formatted_results['computation_time']:.2f}s
Iterations Completed: {formatted_results['iterations_completed']}

"""
        
        if "gap_percentage" in formatted_results:
            prompt += f"Gap to Optimal: {formatted_results['gap_percentage']:.2f}%\n"
        elif "silhouette_score" in formatted_results:
            prompt += f"Silhouette Score: {formatted_results['silhouette_score']:.4f}\n"
        elif "f1_score" in formatted_results:
            prompt += f"F1-Score: {formatted_results['f1_score']:.4f}\n"
        
        prompt += f"""
Convergence History (last 5 iterations):
{formatted_results.get('convergence_history', [])[-5:] if formatted_results.get('convergence_history') else 'N/A'}

## Original Analysis
Method Selected: {llm_analysis['method']}
Reasoning: {llm_analysis['reasoning']}
Confidence: {llm_analysis.get('confidence', 'N/A')}

## Task
Analyze these results and provide:
1. Performance assessment (GOOD/ACCEPTABLE/POOR) with justification
2. Explanation of convergence behavior
3. Specific, actionable recommendations for improvement:
   - Parameter tuning suggestions (exact values)
   - Alternative methods to try
   - Hybrid approaches (e.g., "use GA then 2-opt refinement")
4. Confidence rating in this solution (HIGH/MEDIUM/LOW)

RESPONSE FORMAT: Valid JSON with fields:
- "performance_rating": "GOOD"|"ACCEPTABLE"|"POOR"
- "assessment": "brief explanation"
- "observations": ["observation1", "observation2", ...]
- "recommendations": ["specific suggestion 1", "suggestion 2", ...]
- "confidence_rating": "HIGH"|"MEDIUM"|"LOW"
"""
        
        messages = [
            {"role": "system", "content": "You are MetaMind, an expert CI analyst. Respond ONLY with valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            raw_feedback = self.llm_client.send_request(messages, response_format="json")
            
            # Ensure required fields
            feedback = {
                "performance_rating": raw_feedback.get("performance_rating", "ACCEPTABLE"),
                "assessment": raw_feedback.get("assessment", "No assessment provided"),
                "observations": raw_feedback.get("observations", []),
                "recommendations": raw_feedback.get("recommendations", []),
                "confidence_rating": raw_feedback.get("confidence_rating", "MEDIUM")
            }
            
            # Store for conversation history
            self.conversation_history.append(
                f"Performance: {feedback['performance_rating']}, Recommendations: {len(feedback['recommendations'])}"
            )
            
            return feedback
            
        except Exception as e:
            logger.error(f"LLM feedback failed: {e}")
            # Return structured fallback feedback
            fitness = execution_results["best_fitness"]
            
            # Problem-specific fallback assessment
            if "silhouette_score" in formatted_results:
                silhouette = formatted_results["silhouette_score"]
                if silhouette >= 0.7:
                    rating = "GOOD"
                elif silhouette >= 0.5:
                    rating = "ACCEPTABLE"
                else:
                    rating = "POOR"
                assessment = f"Silhouette={silhouette:.3f} indicates {'strong' if rating=='GOOD' else 'moderate' if rating=='ACCEPTABLE' else 'weak'} clustering structure"
            elif "f1_score" in formatted_results:
                f1 = formatted_results["f1_score"]
                rating = "GOOD" if f1 >= 0.85 else ("ACCEPTABLE" if f1 >= 0.75 else "POOR")
                assessment = f"F1-Score={f1:.3f} ({rating} performance)"
            else:
                # Optimization/TSP fallback
                rating = "ACCEPTABLE"  # Conservative default
                assessment = f"Fitness={fitness:.4f} (baseline assessment)"
            
            return {
                "performance_rating": rating,
                "assessment": assessment,
                "observations": [f"Fitness: {fitness:.4f}", f"Time: {execution_results['computation_time']:.2f}s"],
                "recommendations": [
                    "Increase iterations for better convergence",
                    "Try adjusting exploration/exploitation balance",
                    "Consider alternative method if stagnation occurs"
                ] if rating != "GOOD" else [],
                "confidence_rating": "MEDIUM"
            }
    
    def _incorporate_suggestions(self, preferences: Dict, suggestions: List[str]) -> Dict:
        """Incorporate LLM suggestions into next iteration's preferences."""
        new_prefs = preferences.copy()
        
        # Parse suggestions for parameter adjustments
        for suggestion in suggestions[:2]:  # Only use top 2 suggestions
            suggestion_lower = suggestion.lower()
            
            # Parameter adjustments
            if "increase" in suggestion_lower and "beta" in suggestion_lower:
                new_prefs["beta"] = new_prefs.get("beta", 2.0) * 1.25
            elif "decrease" in suggestion_lower and "evaporation" in suggestion_lower:
                new_prefs["evaporation_rate"] = max(0.1, new_prefs.get("evaporation_rate", 0.5) * 0.8)
            elif "increase" in suggestion_lower and ("iteration" in suggestion_lower or "generation" in suggestion_lower):
                new_prefs["iterations"] = int(new_prefs.get("iterations", 500) * 1.5)
        
        return new_prefs
    
    def _generate_final_report(self, full_results: Dict) -> str:
        """Generate human-readable final report from results."""
        problem = full_results["problem"]
        final = full_results["final_solution"]
        iters = full_results["iterations"]
        
        # Handle None final solution gracefully
        if final is None:
            final_fitness = float('inf')
            final_method = "UNKNOWN"
            final_time = 0.0
            final_iters = 0
        else:
            final_fitness = final['best_fitness']
            final_method = final['method_used']
            final_time = final['computation_time']
            final_iters = final['iterations_completed']
        
        report = f"""
METAMIND CI FRAMEWORK - FINAL REPORT
=====================================
Problem: {problem['type'].upper()} - {problem['instance']}
Timestamp: {full_results['metadata']['timestamp']}
Total Execution Time: {full_results['metadata']['total_time']:.2f} seconds
Iterations Completed: {full_results['metadata']['iterations_completed']}

BEST SOLUTION
-------------
Method: {final_method}
Fitness/Quality: {final_fitness:.4f}
Computation Time: {final_time:.2f}s
Iterations to Convergence: {final_iters}

ITERATION SUMMARY
-----------------"""
        
        for i, it in enumerate(iters):
            method = it['llm_analysis']['method']
            fitness = it['method_execution']['best_fitness']
            rating = it['llm_feedback']['performance_rating']
            report += f"\nIteration {i+1}: {method:10s} | Fitness: {fitness:8.4f} | Rating: {rating:10s} | Time: {it['duration']:5.2f}s"
        
        report += f"""

LLM INSIGHTS
------------
Final Assessment: {iters[-1]['llm_feedback']['assessment'] if iters else 'N/A'}
Key Observations:
"""
        
        for obs in iters[-1]['llm_feedback']['observations'] if iters else ["No observations"]:
            report += f"  * {obs}\n"
        
        report += "\nRecommendations:\n"
        for rec in iters[-1]['llm_feedback']['recommendations'] if iters else ["No recommendations"]:
            report += f"  * {rec}\n"
        
        report += f"""
=====================================
Report generated by MetaMind Orchestrator v1.0
LLM Model: {full_results['metadata']['llm_model']}
"""
        
        return report
    
    def _save_results(self, full_results: Dict, problem_type: str, instance_name: str):
        """Save complete results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{problem_type}_{instance_name}_{timestamp}.json"
        filepath = os.path.join(self.config.results_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return convert_numpy(obj.__dict__)
            else:
                return obj
        
        serializable_results = convert_numpy(full_results)
        
        save_results(serializable_results, filepath)
        logger.info(f"Results saved to {filepath}")
        
        # Also save human-readable report
        report_path = os.path.join(self.config.results_dir, f"report_{problem_type}_{instance_name}_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(full_results["report"])
        logger.info(f"Report saved to {report_path}")
    
    def get_available_problems(self) -> Dict[str, List[str]]:
        """Return available problem instances by type."""
        return {
            "tsp": ["eil51", "berlin52", "kroA100", "random30", "random50"],
            "optimization": ["rastrigin_10d", "rastrigin_20d", "rastrigin_30d", 
                           "ackley_10d", "rosenbrock_10d", "sphere_10d"],
            "classification": ["titanic"],
            "clustering": ["iris", "mall_customers", "synthetic_500_5_1.0"]
        }
    
    def get_available_methods(self) -> List[str]:
        """Return list of available CI methods."""
        return self.method_registry.get_available_methods()


# ==================== CONVENIENCE FUNCTIONS ====================

def create_orchestrator(
    use_mock_llm: bool = False,
    config: Optional[OrchestratorConfig] = None,
    **kwargs
) -> MetaMindOrchestrator:
    """
    Factory function to create orchestrator instance.
    
    Args:
        use_mock_llm: If True, use MockLLMClient instead of real API
        config: Optional OrchestratorConfig instance
        **kwargs: Additional config parameters to override
    
    Returns:
        Configured MetaMindOrchestrator instance
    """
    if config is None:
        config = OrchestratorConfig()
    
    # Override config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    if use_mock_llm:
        llm_client = MockLLMClient(config)
        logger.info("Using MockLLMClient for development")
    else:
        llm_client = None  # Will auto-initialize OpenRouter client
    
    return MetaMindOrchestrator(config=config, llm_client=llm_client)


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("="*70)
    print("METAMIND ORCHESTRATOR - Interactive Demo")
    print("="*70)
    
    # Create orchestrator (using real OpenRouter API)
    orchestrator = create_orchestrator(
        use_mock_llm=False,  # Use real LLM
        max_iterations=2,
        enable_2opt=True
    )
    
    # Show available problems
    print("\nAvailable Problems:")
    for ptype, instances in orchestrator.get_available_problems().items():
        print(f"  {ptype:15s}: {', '.join(instances[:3])}...")
    
    print("\nAvailable Methods:")
    print(f"  {', '.join(orchestrator.get_available_methods())}")
    
    # Run demo on small TSP instance
    print("\n" + "="*70)
    print("Running demo on Random30 TSP instance...")
    print("="*70)
    
    results = orchestrator.solve(
        problem_type="tsp",
        instance_name="random30",
        preferences={
            "time_limit": 30,
            "priority": "solution_quality"
        },
        max_iterations=2
    )
    
    print("\n" + "="*70)
    print("DEMO COMPLETE - Final Results")
    print("="*70)
    print(f"Best Method: {results['final_solution']['method_used']}")
    print(f"Best Tour Length: {results['final_solution']['best_fitness']:.2f}")
    print(f"Total Time: {results['metadata']['total_time']:.2f}s")
    print(f"\nLLM Assessment: {results['iterations'][-1]['llm_feedback']['assessment']}")
    
    print("\nReport saved to results/ directory")
    print("="*70)
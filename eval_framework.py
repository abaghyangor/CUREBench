"""
Bio-Medical AI Competition Starter Kit

A simple framework for evaluating models on bio-medical datasets.
Perfect for getting started quickly in the competition.

Key Features:
- Easy model loading (ChatGPT, Local models, Custom models)
- Simple dataset loading
- Automatic evaluation and scoring
- Submission file generation

Usage:
    framework = CompetitionKit()
    framework.load_model("gpt-4o-mini")
    results = framework.evaluate("quick_test")
    framework.sa        elif question_type == "open_ended":
            # For open-ended, only return response, use NOTAVALUE for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE"  # Use NOTAVALUE instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()ubmission(results, "my_submission.json")
"""

import json
import os
import sys
import logging
import argparse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import csv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class EvaluationResult:
    """Simple container for evaluation results"""
    dataset_name: str
    model_name: str
    accuracy: float
    correct_predictions: int
    total_examples: int
    predictions: List[Dict]  # Changed from List[str] to List[Dict]
    reasoning_traces: List[str] = None  # Add reasoning traces
    details: Optional[Dict] = None


# Model Classes
class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load(self, **kwargs):
        """Load the model"""
        pass
    
    @abstractmethod
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Run inference on the model
        
        Returns:
            Tuple of (response, messages) where messages is the complete conversation history
        """
        pass


class ChatGPTModel(BaseModel):
    """ChatGPT/OpenAI model wrapper"""
    
    def load(self, **kwargs):
        """Load ChatGPT model"""


        api_key = os.getenv("AZURE_OPENAI_API_KEY_O1")
        api_version = "2024-12-01-preview" #"2025-03-01-preview"

        if not api_key:
            raise ValueError(f"API key not found in environment. Please set the appropriate environment variable.")
        
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        from openai import AzureOpenAI
        print("Initializing AzureOpenAI client with endpoint:", azure_endpoint)
        print("Using API version:", api_version)
        self.model_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    
    def inference(self, prompt: str, max_tokens: int = 1024, question_type: str = "multi_choice") -> Tuple[str, List[Dict]]:
        """ChatGPT inference"""
        messages = [{"role": "user", "content": prompt}]

            # Set appropriate token limits based on question type
        if "multi_choice" in question_type:
            completion_tokens = 300  # Concise answers, optimized for multiple choice
        elif "open_ended" in question_type:
            completion_tokens = 1000  # For detailed answers
        
        responses = self.model_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=completion_tokens,
                temperature=0.0,
                seed=42,
                top_p=1.0,
            )
        # print("\033[94m" + str(responses) + "\033[0m")
        response = responses.choices[0].message.content
        
        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": response}]
        
        return response, complete_messages


class LocalModel(BaseModel):
    """Local HuggingFace model wrapper"""
    
    def load(self, **kwargs):
        """Load local HuggingFace model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                **kwargs
            )
            logger.info(f"Loaded local model: {self.model_name}")
        except ImportError as e:
            logger.error(f"Failed to import local model dependencies: {e}")
            raise
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Local model inference"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        print("messages:", messages)
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors='pt', enable_thinking=False
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            temperature=0.4,
            top_p=0.9,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)
        print("response_text:", response_text)
        # Create complete conversation history
        complete_messages = messages + [{"role": "assistant", "content": response_text}]
        
        return response_text, complete_messages


class CustomModel(BaseModel):
    """Custom model wrapper for user-defined models"""
    
    def __init__(self, model_name: str, model_instance, inference_func):
        super().__init__(model_name)
        self.model = model_instance
        self._inference_func = inference_func
    
    def load(self, **kwargs):
        """Custom models are already loaded"""
        logger.info(f"Using custom model: {self.model_name}")
    
    def inference(self, prompt: str, max_tokens: int = 1024) -> Tuple[str, List[Dict]]:
        """Custom model inference"""
        try:
            # For custom models, we'll create a simple message structure
            messages = [{"role": "user", "content": prompt}]
            
            response = self._inference_func(self.model, prompt, max_tokens)
            
            # Create complete conversation history
            complete_messages = messages + [{"role": "assistant", "content": response}]
            
            return response, complete_messages
        except Exception as e:
            logger.error(f"Custom model inference error: {e}")
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "Error occurred"}
            ]
            return "Error occurred", error_messages

class MultiAgentModel(BaseModel):
    """Multi-LLM router-based agent system for CUREBench"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.router_system = None
        self.debug_stats = {
            "total_calls": 0,
            "agent_usage": {},
            "llm_usage": {},
            "routing_decisions": []
        }
    
    def load(self, **kwargs):
        """Load the multi-agent router system"""
        logger.info(f"🚀 Loading multi-agent system: {self.model_name}")
        
        # Initialize the router system
        self.router_system = CUREBenchMultiLLMRouter()
        
        # Debug: Log available configurations
        available_llms = list(self.router_system.llm_configs.keys())
        logger.info(f"✅ Available LLMs: {available_llms}")
        
        # Debug: Log agent-LLM mapping
        mapping = self.router_system.get_agent_llm_mapping()
        logger.info("🎯 Agent-LLM Mapping:")
        for agent, llm in mapping.items():
            logger.info(f"  {agent}: {llm}")
        
        # Verify multi-LLM setup
        if len(available_llms) == 1:
            logger.warning(f"⚠️  WARNING: Only 1 LLM available ({available_llms[0]}). Multi-agent system will use single LLM for all agents.")
        else:
            logger.info(f"✅ Multi-LLM setup verified with {len(available_llms)} different models")
    
    def inference(self, prompt: str, max_tokens: int = 1024, question_type: str = "multi_choice") -> Tuple[str, List[Dict]]:
        """Multi-agent inference with debugging"""
        import asyncio
        
        self.debug_stats["total_calls"] += 1
        call_id = self.debug_stats["total_calls"]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 MULTI-AGENT CALL #{call_id}")
        logger.info(f"📝 Question Type: {question_type}")
        logger.info(f"❓ Question Preview: {prompt[:100]}...")
        logger.info(f"{'='*60}")
        
        try:
            # Run the multi-agent router system
            answer, reasoning = asyncio.run(
                self.router_system.process_question(prompt, question_type)
            )
            
            # Extract routing info from reasoning for debugging
            self._extract_and_log_routing_info(reasoning, call_id)
            
            # Create conversation history format
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reasoning}
            ]
            
            logger.info(f"✅ Call #{call_id} completed successfully")
            self._log_debug_summary()
            
            return reasoning, messages
            
        except Exception as e:
            logger.error(f"❌ Multi-agent system error in call #{call_id}: {e}")
            error_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"Error in multi-agent system: {e}"}
            ]
            return f"Error in multi-agent system: {e}", error_messages
    
    def _extract_and_log_routing_info(self, reasoning: str, call_id: int):
        """Extract and log routing decisions for debugging"""
        import re
        
        # Extract agent and LLM info
        agent_match = re.search(r"Selected Agent: (\w+)", reasoning)
        llm_match = re.search(r"Used LLM: (\w+)", reasoning)
        confidence_match = re.search(r"Routing Confidence: ([\d.]+)", reasoning)
        
        selected_agent = agent_match.group(1) if agent_match else "UNKNOWN"
        used_llm = llm_match.group(1) if llm_match else "UNKNOWN"
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0
        
        # Update statistics
        self.debug_stats["agent_usage"][selected_agent] = self.debug_stats["agent_usage"].get(selected_agent, 0) + 1
        self.debug_stats["llm_usage"][used_llm] = self.debug_stats["llm_usage"].get(used_llm, 0) + 1
        
        # Store routing decision
        routing_info = {
            "call_id": call_id,
            "agent": selected_agent,
            "llm": used_llm,
            "confidence": confidence
        }
        self.debug_stats["routing_decisions"].append(routing_info)
        
        # Log current routing decision
        logger.info(f"🎯 ROUTING DECISION #{call_id}:")
        logger.info(f"  └─ Agent: {selected_agent}")
        logger.info(f"  └─ LLM: {used_llm}")
        logger.info(f"  └─ Confidence: {confidence:.2f}")
    
    def _log_debug_summary(self):
        """Log periodic summary of multi-agent usage"""
        total = self.debug_stats["total_calls"]
        
        # Log every 10 calls
        if total % 10 == 0:
            logger.info(f"\n📊 MULTI-AGENT SUMMARY (after {total} calls):")
            
            logger.info("🤖 Agent Usage:")
            for agent, count in sorted(self.debug_stats["agent_usage"].items()):
                percentage = (count / total) * 100
                logger.info(f"  {agent}: {count} calls ({percentage:.1f}%)")
            
            logger.info("🔧 LLM Usage:")
            for llm, count in sorted(self.debug_stats["llm_usage"].items()):
                percentage = (count / total) * 100
                logger.info(f"  {llm}: {count} calls ({percentage:.1f}%)")
            
            # Check for fallback issues
            fallback_count = self.debug_stats["agent_usage"].get("GENERAL_MEDICINE", 0)
            if fallback_count > total * 0.5:
                logger.warning(f"⚠️  HIGH FALLBACK RATE: {fallback_count}/{total} calls using GENERAL_MEDICINE")
            
            logger.info(f"{'='*40}")

class CompetitionKit:
    """
    Simple competition framework - everything you need in one class!
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the competition kit
        
        Args:
            output_dir: Directory to save results and submissions
            config_path: Path to configuration file containing dataset configs
        """
        self.model = None
        self.model_name = None
        
        self.config = json.load(open(config_path, 'r')) if config_path else {}
        
        self.output_dir = self.config.get('output_dir', 'results')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load dataset configurations from config file or use defaults
        self.datasets = self._load_dataset_configs(self.config)
    
    def load_model(self, model_name: str, model_type: str = "auto", **kwargs):
        """
        Load a model for evaluation
        
        Args:
            model_name: Name/path of the model (e.g., "gpt-4o-mini", "multiagent-router")
            model_type: Type of model ("chatgpt", "local", "custom", "multiagenet", "auto" for auto-detection)
            **kwargs: Additional model configuration
        """
        self.model_name = model_name
        
        # Auto-detect model type if not specified
        if model_type == "auto":
            model_type = self._detect_model_type(model_name)
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        
        if model_type == "chatgpt":
            self.model = ChatGPTModel(model_name)
        elif model_type == "local":
            self.model = LocalModel(model_name)
        elif model_type == "multiagent":
            self.model = MultiAgentModel(model_name)
        elif model_type == "custom":
            # For custom models, user should provide model_instance and inference_func
            model_instance = kwargs.get("model_instance")
            inference_func = kwargs.get("inference_func")
            if not model_instance or not inference_func:
                raise ValueError("Custom model requires 'model_instance' and 'inference_func' parameters")
            self.model = CustomModel(model_name, model_instance, inference_func)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load the model
        self.model.load(**kwargs)
    
    def _load_dataset_configs(self, config) -> Dict:
        """
        Load dataset configurations from config file or return defaults
        
        Args:
            config: Configuration dictionary

        Returns:
            Dictionary of dataset configurations
        """
        if not config:
            print("Not config provided, existing.")
            exit(1)

        # Check if config has a single dataset configuration
        if 'dataset' in config:
            dataset_config = config['dataset']
            dataset_name = dataset_config.get('dataset_name', 'treatment')
            # Create a dictionary with the dataset name as key
            return {dataset_name: dataset_config}
        else:
            # If no dataset in config, return defaults
            print("Not config found, existing.")
            exit(1)

    def _detect_model_type(self, model_name: str) -> str:
        """Auto-detect model type based on model name"""
        if any(name in model_name.lower() for name in ["gpt", "chatgpt", "openai", 'o1', 'o3', 'o4']):
            return "chatgpt"
        elif any(name in model_name.lower() for name in ["multiagent", "multi-agent", "router", "agent"]):
            return "multiagent"
        else:
            return "local"
    
    def evaluate(self, dataset_name: str) -> EvaluationResult:
        """
        Evaluate model on a dataset
        
        Args:
            dataset_name: Name of dataset to evaluate on
            
        Returns:
            EvaluationResult object with scores and predictions
        """
        if not self.model:
            raise ValueError("No model loaded. Call load_model() first.")
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.datasets.keys())}")
        
        dataset_config = self.datasets[dataset_name]
        logger.info(f"Evaluating on {dataset_name}: {dataset_config['description']}")
        
        # Load dataset
        dataset = self._load_dataset(dataset_config)
        
        # Store dataset examples for later use in save_submission
        self._last_dataset_examples = dataset
        
        # Run evaluation
        predictions = []
        reasoning_traces = []  # Store reasoning traces
        total_count = len(dataset)
        # Track accuracy only for non-open-ended questions
        accuracy_correct_count = 0
        accuracy_total_count = 0
        
        logger.info(f"Running evaluation on {total_count} examples...")
        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                # Get prediction and reasoning trace
                prediction, reasoning_trace = self._get_prediction_with_trace(example)
                predictions.append(prediction)
                reasoning_traces.append(reasoning_trace)
                
                # Check if correct based on question type
                is_correct = False
                question_type = example["question_type"]
                expected_answer = example.get("answer")
                """ 
                Debugging Logs 

                print(f"Question type: {question_type}")
                print(f"Raw response: '{prediction}'")
                print(f"Extracted choice: '{prediction['choice']}'")
                print(f"Reasoning Trace: {reasoning_trace}")
                print(f"Expected: '{expected_answer}'")
                print("---")
                """
                
                if question_type == "multi_choice" or question_type == "open_ended_multi_choice":
                    # For multiple choice, compare the choice field
                    if expected_answer !='':
                        is_correct = prediction["choice"] == expected_answer
                    else:
                        is_correct = False
                    # Count for accuracy calculation (exclude open_ended)
                    accuracy_total_count += 1
                    if is_correct:
                        accuracy_correct_count += 1
                elif question_type == "open_ended":
                    # For open-ended, compare the open_ended_answer field but don't count in accuracy, we have internal evaluation for open-ended questions
                    if expected_answer !='':
                        is_correct = prediction["open_ended_answer"] == expected_answer
                    else:
                        is_correct = False
                
                # Log progress
                if (i + 1) % 10 == 0:
                    current_acc = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0
                    logger.info(f"Progress: {i+1}/{total_count}, Accuracy: {current_acc:.2%} (excluding open-ended)")
                    
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                error_prediction = {
                    "choice": "NOTAVALUE",  # Use NOTAVALUE instead of empty string
                    "open_ended_answer": "Error"
                }
                predictions.append(error_prediction)
                reasoning_traces.append("Error occurred during inference")
        
        # Calculate final accuracy (excluding open-ended questions)
        accuracy = accuracy_correct_count / accuracy_total_count if accuracy_total_count > 0 else 0.0
        
        result = EvaluationResult(
            dataset_name=dataset_name,
            model_name=self.model_name,
            accuracy=accuracy,
            correct_predictions=accuracy_correct_count,  # Use accuracy-specific count
            total_examples=accuracy_total_count,  # Use accuracy-specific count
            predictions=predictions,
            reasoning_traces=reasoning_traces  # Include reasoning traces
        )
        
        logger.info(f"Evaluation completed: {accuracy:.2%} accuracy ({accuracy_correct_count}/{accuracy_total_count}) - excluding open-ended questions")
        logger.info(f"Total examples processed: {total_count} (including {total_count - accuracy_total_count} open-ended questions)")
        
        return result
    
    def _load_dataset(self, dataset_config: Dict) -> List[Dict]:
        """Load dataset based on configuration"""
        from dataset_utils import build_dataset
        from torch.utils.data import DataLoader
        
        # Build dataset
        dataset = build_dataset(
            dataset_config.get("dataset_path"),
        )
        
        # Convert to list of dictionaries for easier processing
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        dataset_list = []
        
        for batch in dataloader:
            question_type = batch[0][0]
            
            if question_type == "multi_choice":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                })
            elif question_type == "open_ended_multi_choice":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                    "meta_question": batch[4][0],
                })
            elif question_type == "open_ended":
                dataset_list.append({
                    "question_type": batch[0][0],
                    "id": batch[1][0],
                    "question": batch[2][0],
                    "answer": batch[3][0],
                })
        
        return dataset_list

    
    def _get_prediction_with_trace(self, example: Dict) -> Tuple[Dict, str]:
        """Get model prediction and reasoning trace for a single example"""
        question = example["question"]
        question_type = example["question_type"]
        
        # Format prompt
        if question_type == "multi_choice":
            prompt = f"""You are a clinical expert in pharmaceutical therapeutics with specialized knowledge in drug mechanisms, safety profiles, contraindications, dosage protocols, and patient-specific treatment considerations.

{question}

Provide only the letter (A, B, C, D, or E):"""
            
        elif question_type == "open_ended_multi_choice" or question_type == "open_ended":
            prompt = f"""You are a clinical expert specializing in drug decision-making and treatment planning, with deep expertise in therapeutic reasoning across drug labeling, safety assessment, dosage optimization, contraindications, and patient-specific considerations.

Question: {question}

Provide comprehensive clinical reasoning that covers relevant mechanisms, guidelines, safety considerations, and practical applications:"""
        
        # Get model response and messages using the model's inference method
        response, reasoning_trace = self.model.inference(prompt, question_type=question_type)
        
        # Initialize prediction dictionary
        prediction = {
            "choice": "",  # Use empty string instead of None
            "open_ended_answer": ""  # Use empty string instead of None
        }
        
        # Extract answer from response
        if question_type == "multi_choice":
            # For multiple choice, extract the letter
            choice = self._extract_multiple_choice_answer(response)
            # Ensure choice is never None or NULL
            prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            prediction["open_ended_answer"] = response.strip()  # Keep full response too
        elif question_type == "open_ended_multi_choice":
            # First get the detailed response
            prediction["open_ended_answer"] = response.strip()
            
            # Then use meta question to get choice, if available
            if "meta_question" in example:
                meta_prompt = f"{example['meta_question']}Agent's answer: {response.strip()}\n\nMulti-choice answer:"
                meta_response, meta_reasoning = self.model.inference(meta_prompt, question_type="multi_choice")
                # Combine reasoning traces
                reasoning_trace += meta_reasoning
                # Extract the letter choice
                choice = self._extract_multiple_choice_answer(meta_response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
            else:
                # If no meta_question, try to extract choice directly from the response
                choice = self._extract_multiple_choice_answer(response)
                # Ensure choice is never None or NULL
                prediction["choice"] = choice if choice and str(choice).upper() not in ['NONE', 'NULL'] else ""
        elif question_type == "open_ended":
            # For open-ended, only return response, use N/A for choice to avoid empty string issues
            prediction["choice"] = "NOTAVALUE" # Use N/A instead of empty string to avoid NULL validation issues
            prediction["open_ended_answer"] = response.strip()
        
        return prediction, reasoning_trace
    
    def _extract_multiple_choice_answer(self, response: str) -> str:
        """Extract letter answer from model response"""
        if not response or response is None:
            return "A"
            
        response = response.strip().upper()
        
        # Look for letter at the beginning
        if response and len(response) >=1 and response[0] in ['A', 'B', 'C', 'D', 'E']:
            return response[0]
        
        # Look for "The answer is X" patterns
        import re
        patterns = [
            r'FINAL ANSWER:\s*([A-E])',
            r'SPECIALIST ANALYSIS.*?FINAL ANSWER:\s*([A-E])',
            r'ANALYSIS.*?([A-E])\s*$',
            r'^([A-E])$',
            r'^([A-E])[^A-Z]*$',
            r'(?:ANSWER IS|ANSWER:|IS)\s*([A-E])',
            r'([A-E])\)',
            r'OPTION\s*([A-E])',
            r'\b([A-E])\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        # Final fallback: look for any valid letter in response
        valid_letters = [c for c in response if c in 'ABCDE']
        if valid_letters:
            return valid_letters[-1] if valid_letters else "A"
        
        # Default to empty string if nothing found (to avoid None values in CSV)
        return ""
    
    def save_submission(self, results: List[EvaluationResult], filename: str = "submission.csv", 
                       metadata: Dict = None, dataset_examples: List[Dict] = None,
                       config_path: str = None, args: argparse.Namespace = None):
        """
        Save results in competition submission format as CSV file with metadata JSON and zip package
        
        Args:
            results: List of evaluation results
            filename: Output CSV filename (will be used for CSV inside zip)
            metadata: User-provided metadata dictionary containing model info, track, etc.
            dataset_examples: Original dataset examples to extract question IDs and reasoning traces
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        import pandas as pd
        import zipfile
        
        # Get metadata from various sources with priority order
        metadata = self.get_metadata(config_path, args, metadata)
        
        # Create submission data for CSV
        submission_data = []
        
        # Process each result to create the CSV format
        for result in results:
            # Get the corresponding dataset examples if provided
            examples = dataset_examples if dataset_examples else []
            
            for i, (prediction, example) in enumerate(zip(result.predictions, examples)):
                # Use stored reasoning trace if available, convert to simple text format
                reasoning_trace = json.dumps(result.reasoning_traces[i])
                # if result.reasoning_traces and i < len(result.reasoning_traces):
                #     trace = result.reasoning_traces[i]
                #     if isinstance(trace, list) and len(trace) > 0:
                #         # Convert list of messages to a simple text format
                #         text_parts = []
                #         for msg in trace:
                #             if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                #                 role = msg['role']
                #                 content = msg['content'].replace('\n', ' ').replace('\r', '').replace('"', "'")
                #                 text_parts.append(f"{role}: {content}")
                #         reasoning_trace = " | ".join(text_parts)
                #     else:
                #         # Fallback to string representation
                #         reasoning_trace = str(trace).replace('\n', ' ').replace('\r', '').replace('"', "'")
                
                # Clean up text fields to avoid CSV formatting issues
                prediction_text = prediction.get("open_ended_answer", "") or ""  # Ensure not None
                if not prediction_text or prediction_text.strip() == "":
                    prediction_text = "No prediction available"

                
                # Ensure choice is clean and never NULL
                choice_raw = prediction.get("choice", "")
                if choice_raw is None or str(choice_raw).upper() in ['NULL', 'NONE', 'NAN']:
                    choice_clean = "NOTAVALUE"  # Use NOTAVALUE instead of empty string
                elif str(choice_raw).strip() == "":
                    choice_clean = "NOTAVALUE"  # Replace empty strings with NOTAVALUE to avoid NULL validation issues
                else:
                    choice_clean = str(choice_raw).strip()
                
                # Ensure reasoning trace is not null
                if not reasoning_trace or reasoning_trace == "null" or reasoning_trace.strip() == "":
                    reasoning_trace = "No reasoning available"
                
                # Create CSV row - let pandas handle the escaping
                row = {
                    "id": str(example.get("id", str(i)) or f"unknown_{i}"),
                    "prediction": str(prediction_text),
                    "choice": str(choice_clean),
                    "reasoning": str(reasoning_trace)
                }
                
                # Debug: Log if choice is NULL-like
                if str(choice_clean).upper() in ['NULL', 'NONE', 'NAN'] or str(choice_clean).strip() == "":
                    logger.warning(f"Found NULL-like or empty choice for row {row['id']}: '{choice_clean}' - replacing with NOTAVALUE")
                    row["choice"] = "NOTAVALUE"
                
                submission_data.append(row)
        
        # Create DataFrame and save CSV with proper quoting and NaN handling
        df = pd.DataFrame(submission_data)
        
        # Convert all columns to string to avoid type issues
        for col in df.columns:
            df[col] = df[col].astype(str)
        
        # Aggressive null value cleaning
        null_replacements = {
            'id': 'unknown_id',
            'prediction': 'No prediction available',
            'choice': 'NOTAVALUE',  # Use NOTAVALUE for choice instead of empty string
            'reasoning': 'No reasoning available'
        }
        
        # Replace all possible null-like values
        for col in df.columns:
            # Replace pandas null values
            df[col] = df[col].fillna(null_replacements.get(col, 'NOTAVALUE'))
            
            # Replace string representations of null
            null_like_values = ['nan', 'NaN', 'None', 'null', 'NULL', '<NA>', 'nat', 'NaT']
            for null_val in null_like_values:
                df[col] = df[col].replace(null_val, null_replacements.get(col, 'NOTAVALUE'))
            
            # Special handling for choice column - ensure it's never empty or null-like
            if col == 'choice':
                df[col] = df[col].replace('NOTAVALUE', 'NOTAVALUE')  # Keep NOTAVALUE as is for choice
                # Replace any null-like values with NOTAVALUE
                for null_val in null_like_values:
                    df[col] = df[col].replace(null_val, 'NOTAVALUE')
                # Replace empty strings with NOTAVALUE for choice column
                df[col] = df[col].replace('', 'NOTAVALUE')
                df[col] = df[col].replace(' ', 'NOTAVALUE')  # Also replace whitespace-only
            
            # Replace empty strings (except for choice column which can be empty)
            if col != 'choice' and col in null_replacements:
                df[col] = df[col].replace('', null_replacements[col])
                df[col] = df[col].replace(' ', null_replacements[col])  # Also replace whitespace-only
        
        csv_path = os.path.join(self.output_dir, filename)
        
        # Validate DataFrame before saving
        logger.info(f"Creating CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Final validation - check for any remaining nulls
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Still found {null_count} nulls in column {col}")
        
        # Check for any problematic data
        for idx, row in df.head().iterrows():
            logger.debug(f"Sample row {idx}: id={row['id']}, choice='{row['choice']}', prediction_len={len(str(row['prediction']))}, reasoning_len={len(str(row['reasoning']))}")
        
        # Final safety check: ensure choice column has no NULL values or empty strings
        logger.info("Performing final NULL check on choice column...")
        null_patterns = ['NULL', 'null', 'None', 'NaN', 'nan', '<NA>', 'nat', 'NaT', 'NOTAVALUE']
        for pattern in null_patterns:
            count_before = (df['choice'] == pattern).sum()
            if count_before > 0:
                logger.warning(f"Found {count_before} instances of '{pattern}' in choice column, replacing with NOTAVALUE")
                df['choice'] = df['choice'].replace(pattern, 'NOTAVALUE')
        
        # Replace empty strings with NOTAVALUE to avoid NULL validation issues
        empty_count = (df['choice'] == '').sum()
        if empty_count > 0:
            logger.warning(f"Found {empty_count} empty strings in choice column, replacing with NOTAVALUE")
            df['choice'] = df['choice'].replace('', 'NOTAVALUE')
        
        # Also replace any remaining pandas nulls in choice column
        null_mask = df['choice'].isnull()
        if null_mask.sum() > 0:
            logger.warning(f"Found {null_mask.sum()} pandas null values in choice column, replacing with NOTAVALUE")
            df.loc[null_mask, 'choice'] = 'NOTAVALUE'
        

        # Use proper CSV parameters for robust handling of complex data
        df.to_csv(csv_path, index=False, na_rep='NOTAVALUE', quoting=1)  # index=False to avoid pandas index issues
        logger.info(f"Successfully saved CSV to {csv_path}")
    
        # Create metadata JSON file
        metadata_filename = "meta_data.json"
        metadata_path = os.path.join(self.output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create ZIP file with CSV and metadata
        zip_filename = filename.replace('.csv', '.zip')
        zip_path = os.path.join(self.output_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV file to zip
            zipf.write(csv_path, filename)
            # Add metadata JSON to zip
            zipf.write(metadata_path, metadata_filename)
        
        # Calculate and log overall accuracy
        total_correct = sum(r.correct_predictions for r in results)
        total_examples = sum(r.total_examples for r in results)
        overall_accuracy = total_correct / total_examples if total_examples > 0 else 0.0
        
        logger.info(f"CSV submission saved to: {csv_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info(f"Submission package saved to: {zip_path}")
        logger.info(f"Overall accuracy (excluding open-ended questions): {overall_accuracy:.2%} ({total_correct}/{total_examples})")
        
        return zip_path
    
    def save_submission_with_metadata(self, results: List[EvaluationResult], 
                                     metadata: Dict = None, filename: str = "submission.csv",
                                     config_path: str = None, args: argparse.Namespace = None):
        """
        Convenient method to save submission with user-provided metadata as CSV with zip package
        
        Args:
            results: List of evaluation results
            metadata: User-provided metadata dictionary with fields like:
                - model_name: Name of the model
                - model_type: Type of model wrapper used  
                - track: "internal_reasoning" or "agentic_reasoning"
                - base_model_type: "API" or "OpenWeighted"
                - base_model_name: Name of the base model
                - dataset: Dataset name
                - additional_info: Any additional information
            filename: Output CSV filename
            config_path: Path to configuration file containing metadata
            args: Command line arguments containing metadata
        """
        # Use the stored dataset examples from the last evaluation
        dataset_examples = getattr(self, '_last_dataset_examples', [])
        
        return self.save_submission(results, filename, metadata, dataset_examples, config_path, args)
    
    def list_datasets(self):
        """List available datasets"""
        print("Available Datasets:")
        print("-" * 50)
        for name, config in self.datasets.items():
            print(f"  {name}: {config['description']}")

    def load_metadata_from_config(self, config_path: str) -> Dict:
        """
        Load metadata from configuration file
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Metadata dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        _, ext = os.path.splitext(config_path)
        
        with open(config_path, 'r') as f:
            if ext.lower() in ['.json']:
                config = json.load(f)
            elif ext.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML is required for YAML config files. Install with: pip install PyYAML")
            else:
                raise ValueError(f"Unsupported config file format: {ext}")
        
        # Extract metadata from config
        metadata = config.get('metadata', config.get('meta_data', {}))
        
        # Validate required fields
        required_fields = ['model_name', 'track', 'base_model_type', 'base_model_name', 'dataset']
        for field in required_fields:
            if field not in metadata:
                logger.warning(f"Required metadata field '{field}' not found in config")
        
        return metadata
    
    def parse_metadata_from_args(self, args: argparse.Namespace) -> Dict:
        """
        Parse metadata from command line arguments
        
        Args:
            args: Parsed command line arguments
            
        Returns:
            Metadata dictionary
        """
        metadata = {}
        
        # Map argument names to metadata fields
        arg_mapping = {
            'model_name': 'model_name',
            'model_type': 'model_type',
            'track': 'track',
            'base_model_type': 'base_model_type',
            'base_model_name': 'base_model_name',
            'dataset': 'dataset',
            'additional_info': 'additional_info'
        }
        
        for arg_name, meta_field in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                metadata[meta_field] = getattr(args, arg_name)
        
        return metadata
    
    def get_metadata(self, config_path: str = None, args: argparse.Namespace = None, 
                    fallback_metadata: Dict = None) -> Dict:
        """
        Get metadata from various sources with priority order:
        1. Command line arguments (highest priority)
        2. Configuration file
        3. Fallback metadata provided
        4. Default metadata (lowest priority)
        
        Args:
            config_path: Path to configuration file
            args: Parsed command line arguments
            fallback_metadata: Fallback metadata dictionary
            
        Returns:
            Final metadata dictionary
        """
        # Start with default metadata
        metadata = {
            "model_name": self.model_name or "unknown",
            "model_type": type(self.model).__name__ if self.model else "Unknown",
            "track": "internal_reasoning",
            "base_model_type": "API",
            "base_model_name": self.model_name or "unknown",
            "dataset": "unknown",
            "additional_info": "Generated using eval_framework"
        }
        
        # Override with fallback metadata if provided
        if fallback_metadata:
            metadata.update(fallback_metadata)
        
        # Override with config file metadata if provided
        if config_path:
            try:
                config_metadata = self.load_metadata_from_config(config_path)
                metadata.update(config_metadata)
                logger.info(f"Loaded metadata from config file: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_path}: {e}")
        
        # Override with command line arguments if provided (highest priority)
        if args:
            arg_metadata = self.parse_metadata_from_args(args)
            metadata.update(arg_metadata)
            if arg_metadata:
                logger.info(f"Applied metadata from command line arguments")
        
        return metadata
    

# === MULTI-LLM ROUTER SYSTEM ===
# Add this entire section at the END of your eval_framework.py file (before the metadata parser functions)

import asyncio
import re
from typing import Dict
from dataclasses import dataclass

@dataclass
class AgentRoute:
    agent_name: str
    confidence: float
    reasoning: str

@dataclass  
class LLMConfig:
    client: any
    model_name: str
    api_type: str
    max_tokens: int = 800
    temperature: float = 0.1
    endpoint: str = None
    api_key: str = None

class CUREBenchMultiLLMRouter:
    def __init__(self):
        self.llm_configs = {}
        self.setup_llm_configs()
    
    def setup_llm_configs(self):
        """Setup different LLM configurations with debugging"""
        logger.info("🔧 Setting up LLM configurations...")
        
        # Azure OpenAI GPT-4o
        if os.getenv("AZURE_OPENAI_API_KEY_O1"):
            logger.info("✅ Configuring Azure OpenAI GPT-4o")
            from openai import AzureOpenAI
            gpt4o_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY_O1"),
                api_version="2024-12-01-preview",
            )
            
            self.llm_configs["gpt4o"] = LLMConfig(
                client=gpt4o_client,
                model_name="gpt-4o",
                api_type="azure_openai",
                max_tokens=600,
                temperature=0.01
            )
        else:
            logger.warning("⚠️  Azure OpenAI GPT-4o not configured (missing AZURE_OPENAI_API_KEY_O1)")
        
        # Azure OpenAI GPT-4o-mini
        if os.getenv("AZURE_OPENAI_API_KEY_O1"):
            logger.info("✅ Configuring Azure OpenAI GPT-4o-mini")
            gpt4o_mini_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY_O1"),
                api_version="2024-12-01-preview",
            )
            
            self.llm_configs["gpt4o_mini"] = LLMConfig(
                client=gpt4o_mini_client,
                model_name="gpt-4o-mini",
                api_type="azure_openai",
                max_tokens=300,
                temperature=0.1
            )
        else:
            logger.warning("⚠️  Azure OpenAI GPT-4o-mini not configured")
        
        # Azure AI Foundry DeepSeek
        if os.getenv("AZURE_DEEPSEEK_API_KEY") and os.getenv("AZURE_DEEPSEEK_ENDPOINT"):
            logger.info("✅ Configuring Azure DeepSeek")
            from openai import OpenAI
            azure_deepseek_client = OpenAI(
                base_url=os.getenv("AZURE_DEEPSEEK_ENDPOINT"),
                api_key=os.getenv("AZURE_DEEPSEEK_API_KEY")
            )
            
            self.llm_configs["deepseek_azure"] = LLMConfig(
                client=azure_deepseek_client,
                model_name="DeepSeek-R1",
                api_type="azure_deepseek",
                max_tokens=1000,
                temperature=0.2
            )
        else:
            logger.info("ℹ️  Azure DeepSeek not configured (optional)")
        
        # Direct DeepSeek API
        if os.getenv("DEEPSEEK_API_KEY"):
            logger.info("✅ Configuring Direct DeepSeek")
            from openai import OpenAI
            direct_deepseek_client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )
            
            self.llm_configs["deepseek_direct"] = LLMConfig(
                client=direct_deepseek_client,
                model_name="deepseek-reasoner",
                api_type="deepseek_direct",
                max_tokens=400,
                temperature=0.2
            )
        else:
            logger.info("ℹ️  Direct DeepSeek not configured (optional)")
        
        # Summary
        configured_llms = list(self.llm_configs.keys())
        logger.info(f"🎉 LLM configuration complete: {len(configured_llms)} models configured")
        for llm in configured_llms:
            logger.info(f"  ✓ {llm}")
        
        if len(configured_llms) == 0:
            logger.error("❌ NO LLMs CONFIGURED! Check your environment variables.")
        elif len(configured_llms) == 1:
            logger.warning(f"⚠️  Only 1 LLM configured. Multi-agent system will fallback to single model.")

    def get_agent_llm_mapping(self) -> Dict[str, str]:
        """Define which LLM to use for each agent type"""
        
        # Check which DeepSeek option is available
        deepseek_option = None
        if "deepseek_direct" in self.llm_configs:
            deepseek_option = "deepseek_direct"  # Prefer Azure version
        elif "deepseek_azure" in self.llm_configs:
            deepseek_option = "deepseek_azure"  # Fallback to direct
        
        mapping = {
            "PLANNER": "gpt4o_mini",  # Cost-effective routing
            "DRUG_BRANDS": "gpt4o",  # Factual accuracy  
            "CLINICAL_GUIDELINES": "gpt4o",  # Guidelines and protocols
            "TREATMENT_PLANNING": "gpt4o",  # Complex reasoning
            "PHARMACOLOGY": "gpt4o",  # Mechanism understanding  
            "DIAGNOSTICS": "gpt4o_mini",  # Analytical reasoning
            "DOSAGE_CALCULATION": "gpt4o",  # Precise calculations
            "CONTRAINDICATIONS": "gpt4o",  # Safety information
            "STORAGE_HANDLING": "gpt4o_mini",  # Simple factual info
            "ADVERSE_EVENT": "gpt4o",  # Complex safety reasoning
            "DRUG_INGREDIENTS": "gpt4o",      # Factual formulation data
            "CLINICAL_STUDIES": "gpt4o",      # Statistical analysis
            "TOXICOLOGY": "gpt4o",         # Complex safety reasoning  
            "PATIENT_POPULATIONS": "gpt4o",    # Specialized population dosing
            "GENERAL_MEDICINE": "gpt4o",  # Reliable fallback
        }
        
        # Clean up None values and fallback to available models
        available_llms = list(self.llm_configs.keys())
        if available_llms:
            fallback_llm = available_llms[0]
            for agent, preferred_llm in mapping.items():
                if preferred_llm is None or preferred_llm not in self.llm_configs:
                    mapping[agent] = fallback_llm
        
        return mapping
        
    async def planner_agent(self, question: str) -> AgentRoute:
        """Central planner with debugging"""
        logger.info("🎯 Running planner agent for routing decision...")
        
        prompt = f"""You are a medical AI planner. Route this question to the most appropriate specialist.

QUESTION: {question}

AVAILABLE SPECIALISTS:
1. PHARMACOLOGY - Drug mechanisms, interactions, pharmacokinetics
2. TREATMENT_PLANNING - Treatment protocols, therapy selection, clinical management  
3. DIAGNOSTICS - Lab tests, imaging, diagnostic procedures
4. DRUG_BRANDS - Specific brand names, formulations, brand information
5. CLINICAL_GUIDELINES - Evidence-based guidelines, protocols, standards
6. DOSAGE_CALCULATION - Dosing, administration, pediatric/adult dosing
7. CONTRAINDICATIONS - Safety considerations, when NOT to use treatments
8. STORAGE_HANDLING - Storage conditions, temperature, handling
9. ADVERSE_EVENT - Predicting potential adverse effects, side effects, drug interactions
10. PATIENT_POPULATIONS - Usage in pregnancy, pediatric, geriatric, special populations  
11. CLINICAL_STUDIES - Clinical trial data, research findings, statistical analysis
12. DRUG_INGREDIENTS - Active/inactive ingredients, formulation components, composition
13. TOXICOLOGY - Toxicity profiles, carcinogenesis, mutagenesis, safety studies
14. GENERAL_MEDICINE - Broad medical knowledge, general concepts

Choose the SINGLE most relevant specialist. Respond with:
SELECTED_AGENT: [AGENT_NAME]
CONFIDENCE: [0.1-1.0]
REASONING: [Brief explanation]"""

        mapping = self.get_agent_llm_mapping()
        planner_llm = mapping["PLANNER"]
        logger.info(f"🔧 Using {planner_llm} for routing decision")
        
        llm_config = self.llm_configs[planner_llm]
        response = await self._get_completion(prompt, llm_config, max_tokens=300)
        
        agent_name = self._extract_selected_agent(response)
        confidence = self._extract_confidence(response)
        reasoning = self._extract_reasoning(response)
        
        logger.info(f"🎯 Planner decision: {agent_name} (confidence: {confidence:.2f})")
        logger.info(f"📝 Reasoning: {reasoning[:100]}...")
        
        return AgentRoute(agent_name, confidence, reasoning)
    
    async def run_specialist_agent(self, agent_name: str, question: str, question_type: str) -> Tuple[str, str]:
        """Run specialist agent with debugging"""
        mapping = self.get_agent_llm_mapping()
        llm_key = mapping.get(agent_name, "gpt4o")
        
        logger.info(f"🤖 Running {agent_name} agent with {llm_key}")
        
        if llm_key not in self.llm_configs:
            logger.error(f"❌ LLM {llm_key} not available! Available: {list(self.llm_configs.keys())}")
            # Fallback to first available LLM
            fallback_llm = list(self.llm_configs.keys())[0]
            logger.warning(f"⚠️  Falling back to {fallback_llm}")
            llm_key = fallback_llm
        
        llm_config = self.llm_configs[llm_key]
        prompt = self._get_agent_prompt(agent_name, question)
        
        logger.info(f"🔧 Using {llm_config.model_name} ({llm_config.api_type})")
        response = await self._get_completion(prompt, llm_config)
        answer = self._extract_answer(response)
        
        logger.info(f"✅ {agent_name} completed, extracted answer: {answer}")
        
        return answer, response

    def _get_agent_prompt(self, agent_name: str, question: str) -> str:
        """Get specialized prompt for each agent type"""
        
        base_prompts = {
            "PHARMACOLOGY": f"""You are a clinical pharmacologist specializing in drug mechanisms, interactions, and pharmacokinetics.
Example:
QUESTION: Which mechanism best describes how omeprazole reduces gastric acid?
Choices:
A. H2 receptor blockade
B. Irreversible inhibition of the H+/K+ ATPase in parietal cells
C. Neutralizes acid by buffering
D. Stimulates gastric motility

Reasoning: Omeprazole is a proton pump inhibitor that irreversibly inhibits the H+/K+ ATPase on gastric parietal cells, reducing acid secretion.
FINAL ANSWER: B

Focus on molecular mechanisms, drug interactions, pharmacokinetic principles, and safety profiles.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}
""",


            "TREATMENT_PLANNING": f"""You are a clinical specialist in evidence-based treatment planning and therapeutic management.
Example:
QUESTION: A patient with non-ST elevation myocardial infarction (NSTEMI) arrives within hours of chest pain. Which is the most appropriate immediate management?
Choices:
A. Immediate thrombolysis
B. Initiate antiplatelet therapy, anticoagulation, and risk-stratify for PCI
C. Start high-dose corticosteroids
D. Discharge with outpatient follow-up

Reasoning: NSTEMI is managed with antiplatelet agents and anticoagulation and then risk stratified for invasive management; thrombolysis is for certain STEMI cases.
FINAL ANSWER: B

Focus on treatment protocols, therapeutic decision-making, clinical management strategies, and patient optimization.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

            "DIAGNOSTICS": f"""You are a diagnostic medicine specialist with expertise in laboratory medicine, imaging, and procedures.
Example:
QUESTION: What is the best initial diagnostic test to confirm a suspected deep vein thrombosis in a symptomatic leg?
Choices:
A. Chest X-ray
B. Venous duplex (compression) ultrasound of the leg
C. CT abdomen
D. D-dimer alone

Reasoning: Compression duplex ultrasound of the leg veins is the preferred initial imaging for suspected DVT in a symptomatic limb.
FINAL ANSWER: B

Focus on test interpretation, diagnostic accuracy, clinical correlation, and diagnostic procedures.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

            "DRUG_BRANDS": f"""You are a pharmaceutical specialist with expertise in brand-name medications and formulations.
Example:
QUESTION: Which brand name corresponds to acetaminophen in many countries?
Choices:
A. Tylenol
B. Lipitor
C. Advair
D. Crestor

Reasoning: Tylenol is a widely used brand name for acetaminophen/paracetamol.
FINAL ANSWER: A

Focus on brand identification, formulation differences, brand-specific indications, and manufacturer guidelines.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

            "CLINICAL_GUIDELINES": f"""You are a clinical guidelines specialist with expertise in evidence-based practice standards.
Example:
QUESTION: Per common outpatient guidelines, first-line empirical therapy for uncomplicated community-acquired pneumonia in a previously healthy adult is:
Choices:
A. Macrolide monotherapy (e.g., azithromycin)
B. IV vancomycin
C. Amphotericin B
D. High-dose corticosteroids

Reasoning: For otherwise healthy outpatients, macrolide monotherapy is guideline-consistent empirical therapy.
FINAL ANSWER: A

Focus on professional guidelines, evidence-based recommendations, standard of care, and regulatory requirements.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

            "DOSAGE_CALCULATION": f"""You are a clinical dosing specialist with expertise in dosage calculations and administration.
Example:
QUESTION: A medication is dosed at 5 mg/kg. For a 70 kg adult, what is the correct single dose?
Choices:
A. 200 mg
B. 250 mg
C. 350 mg
D. 400 mg

Reasoning: 5 mg/kg × 70 kg = 350 mg.
FINAL ANSWER: C

Focus on dosing calculations, patient-specific adjustments, administration routes, and safety margins.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

            "CONTRAINDICATIONS": f"""You are a medication safety specialist with expertise in contraindications and risk assessment.
Example:
QUESTION: Which of the following is an absolute contraindication to ACE inhibitor use?
Choices:
A. Controlled hypertension
B. Pregnancy
C. Mild dehydration
D. Uncomplicated hyperlipidemia

Reasoning: ACE inhibitors are contraindicated in pregnancy due to fetal renal/teratogenic risk.
FINAL ANSWER: B

Focus on absolute/relative contraindications, special populations, risk factors, and safety warnings.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

            "STORAGE_HANDLING": f"""You are a pharmaceutical storage specialist with expertise in drug stability and handling.
Example:
QUESTION: How should unopened insulin vials generally be stored?
Choices:
A. Room temperature indefinitely
B. Refrigerated at 2–8°C until first use
C. Frozen for long-term storage
D. Kept in direct sunlight

Reasoning: Unopened insulin vials are typically refrigerated (2–8°C); freezing and sunlight must be avoided.
FINAL ANSWER: B

Focus on storage conditions, temperature requirements, stability, and handling guidelines.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

 "ADVERSE_EVENT": f"""You are a drug safety specialist with expertise in adverse effects and drug interactions.
Example:
QUESTION: Which antibiotic is most strongly associated with dose-related nephrotoxicity and ototoxicity?
Choices:
A. Penicillin
B. Macrolides
C. Aminoglycosides (e.g., gentamicin)
D. Tetracyclines

Reasoning: Aminoglycosides are known for nephrotoxicity and ototoxicity, particularly at higher doses or prolonged use.
FINAL ANSWER: C

Focus on potential adverse effects, drug-drug interactions, contraindicated combinations, and safety profiles.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

        "PATIENT_POPULATIONS": f"""You are a clinical specialist in special population pharmacotherapy.
Example:
QUESTION: Which analgesic is generally considered acceptable during pregnancy when clinically indicated?
Choices:
A. Acetaminophen (paracetamol)
B. Isotretinoin
C. Warfarin
D. Methotrexate

Reasoning: Acetaminophen is commonly regarded as safe for use in pregnancy; warfarin, methotrexate, and isotretinoin are contraindicated.
FINAL ANSWER: A

Focus on pediatric, geriatric, pregnancy, and special population considerations for drug therapy.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

        "CLINICAL_STUDIES": f"""You are a clinical research specialist with expertise in interpreting trial data and research findings.
Example:
QUESTION: In a randomized trial, a p-value of 0.03 for the primary endpoint indicates:
Choices:
A. The observed effect is unlikely due to chance at the 0.05 level
B. The effect size is clinically large
C. The study is invalid
D. The null hypothesis is proven true

Reasoning: A p-value of 0.03 is less than 0.05, indicating statistical significance at the 5% level (does not by itself speak to clinical size).
FINAL ANSWER: A

Focus on clinical trial results, statistical analysis, research methodology, and evidence interpretation.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

        "DRUG_INGREDIENTS": f"""You are a pharmaceutical formulation specialist with expertise in drug composition.
Example:
QUESTION: Which component is the active ingredient in standard ibuprofen tablets?
Choices:
A. Lactose
B. Ibuprofen
C. Magnesium stearate
D. Microcrystalline cellulose

Reasoning: Ibuprofen is the active pharmaceutical ingredient; the others are excipients.
FINAL ANSWER: B

Focus on active ingredients, inactive components, excipients, and formulation characteristics.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

        "TOXICOLOGY": f"""You are a toxicology specialist with expertise in drug toxicity and safety studies.
Example:
QUESTION: Acetaminophen overdose primarily causes toxicity to which organ via NAPQI formation?
Choices:
A. Lungs
B. Liver
C. Heart
D. Pancreas

Reasoning: Acetaminophen overdose causes hepatotoxicity via the reactive metabolite NAPQI which depletes glutathione.
FINAL ANSWER: B

Focus on toxicity profiles, carcinogenicity, mutagenicity, teratogenicity, and safety studies.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

""",

            "GENERAL_MEDICINE": f"""You are a general medicine expert with broad medical knowledge.
Example:
QUESTION: A 68-year-old with resting tremor, bradykinesia, and rigidity most likely has:
Choices:
A. Parkinson disease
B. Essential tremor
C. Huntington disease
D. Myasthenia gravis

Reasoning: The triad of resting tremor, bradykinesia, and rigidity is characteristic of Parkinson disease.
FINAL ANSWER: A

Provide comprehensive medical analysis covering relevant principles, pathophysiology, and clinical reasoning.
End with: FINAL ANSWER: [LETTER]

QUESTION: {question}

"""
        }
        
        return base_prompts.get(agent_name, base_prompts["GENERAL_MEDICINE"])
    
    async def process_question(self, question: str, question_type: str) -> Tuple[str, str]:
        """Main processing function that coordinates all agents"""
        
        # Step 1: Get routing decision
        route = await self.planner_agent(question)
        
        # Step 2: Run specialist with appropriate LLM
        answer, reasoning = await self.run_specialist_agent(route.agent_name, question, question_type)
        
        # Step 3: Format response with routing info
        llm_mapping = self.get_agent_llm_mapping()
        used_llm = llm_mapping.get(route.agent_name, "unknown")
        
        final_reasoning = f"""=== MULTI-LLM ROUTER ANALYSIS ===

ROUTING DECISION:
• Selected Agent: {route.agent_name}
• Used LLM: {used_llm}
• Routing Confidence: {route.confidence:.2f}
• Routing Reasoning: {route.reasoning}

SPECIALIST ANALYSIS ({used_llm.upper()}):
{reasoning}

FINAL ANSWER: {answer}"""
        
        return answer, final_reasoning
    
    async def _get_completion(self, prompt: str, llm_config: LLMConfig, max_tokens: int = None) -> str:
        """Get completion with API debugging"""
        if max_tokens is None:
            if 'STORAGE_HANDLING' in prompt or 'DRUG_BRANDS' in prompt:
                max_tokens = 200
            elif "TREATMENT_PLANNING" in prompt or "PHARMACOLOGY" in prompt:
                max_tokens = 500  # Complex reasoning  
            else:
                max_tokens = llm_config.max_tokens
        
        # Ensure minimum tokens for proper responses
        max_tokens = max(max_tokens, 350)
        
        logger.debug(f"🔌 API call to {llm_config.api_type} - {llm_config.model_name}")
        logger.debug(f"📊 Tokens: {max_tokens}, Temp: {llm_config.temperature}")
        
        try:
            if llm_config.api_type == "azure_openai":
                response = llm_config.client.chat.completions.create(
                    model=llm_config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_tokens,
                    temperature=llm_config.temperature
                )
                result = response.choices[0].message.content.strip()
                logger.debug(f"✅ {llm_config.api_type} API success - {len(result)} chars returned")
                return result
            
            elif llm_config.api_type in ["azure_deepseek", "deepseek_direct"]:
                response = llm_config.client.chat.completions.create(
                    model=llm_config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=llm_config.temperature
                )
                result = response.choices[0].message.content.strip()
                logger.debug(f"✅ {llm_config.api_type} API success - {len(result)} chars returned")
                return result
            
            else:
                raise ValueError(f"Unknown API type: {llm_config.api_type}")
                
        except Exception as e:
            logger.error(f"❌ API error for {llm_config.api_type} ({llm_config.model_name}): {e}")
            return f"Error in {llm_config.api_type} API call: {e}"
        
    def _extract_selected_agent(self, response: str) -> str:
            """Extract selected agent from planner response"""
            patterns = [
                r"SELECTED_AGENT:\s*([A-Z_]+)",
                r"Agent:\s*([A-Z_]+)",
                r"Route to:\s*([A-Z_]+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    agent = match.group(1).upper()
                    valid_agents = [
                        "PHARMACOLOGY", "TREATMENT_PLANNING", "DIAGNOSTICS", 
                        "DRUG_BRANDS", "CLINICAL_GUIDELINES", "DOSAGE_CALCULATION",
                        "CONTRAINDICATIONS", "STORAGE_HANDLING", "ADVERSE_EVENT",
                        "PATIENT_POPULATIONS", "CLINICAL_STUDIES", "DRUG_INGREDIENTS", 
                        "TOXICOLOGY", "GENERAL_MEDICINE"
                    ]
                    if agent in valid_agents:
                        return agent
            
            return "GENERAL_MEDICINE"
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        patterns = [
            r"CONFIDENCE:\s*([0-9\.]+)",
            r"confidence[:\s]*([0-9\.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    conf = float(match.group(1))
                    return min(conf, 1.0) if conf <= 1.0 else conf / 100.0
                except:
                    continue
        
        return 0.8
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from planner response"""
        patterns = [
            r"REASONING:\s*(.+?)(?=\n\n|\n[A-Z]+:|$)",
            r"Reasoning:\s*(.+?)(?=\n\n|\n[A-Z]+:|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return "Automated routing decision"
    
    def _extract_answer(self, response: str) -> str:
        """Extract final answer from specialist response"""
        if not response:
            return "A"
            
        response_upper = response.strip().upper()
        
        # Priority 1: Explicit FINAL ANSWER (take LAST occurrence)
        final_patterns = [
            r'FINAL ANSWER:\s*([A-E])',
            r'FINAL:\s*([A-E])',
            r'ANSWER:\s*([A-E])'
        ]
        
        for pattern in final_patterns:
            matches = re.findall(pattern, response_upper)
            if matches:
                return matches[-1]  # CRITICAL: Take LAST match
        
        # Priority 2: Standalone letters at end of response
        lines = response_upper.strip().split('\n')
        for line in reversed(lines[-3:]):  # Check last 3 lines
            line = line.strip()
            if re.match(r'^[A-E]\.?$', line):
                return line[0]
        
        # Priority 3: Any valid letter (take last)
        valid_letters = re.findall(r'\b([A-E])\b', response_upper)
        return valid_letters[-1] if valid_letters else "A"

# === END OF MULTI-LLM ROUTER SYSTEM ===

def create_metadata_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser for metadata
    
    Returns:
        ArgumentParser with metadata-related arguments
    """
    parser = argparse.ArgumentParser(description='Evaluation Framework with Metadata Support')
    
    # Model information
    parser.add_argument('--model-name', type=str, help='Name of the model')
    parser.add_argument('--model-type', type=str, help='Type of model wrapper')
    parser.add_argument('--base-model-name', type=str, help='Name of the base model')
    parser.add_argument('--base-model-type', type=str, choices=['API', 'OpenWeighted'], 
                       help='Type of base model (API or OpenWeighted)')
    
    # Track information
    parser.add_argument('--track', type=str, choices=['internal_reasoning', 'agentic_reasoning'],
                       default='internal_reasoning', help='Competition track')
    
    # Dataset and submission info
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--additional-info', type=str, help='Additional information about the submission')
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON or YAML)')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='competition_results', 
                       help='Output directory for results')
    parser.add_argument('--output-file', type=str, default='submission.csv', 
                       help='Output CSV filename for submission (will be packaged in zip)')
    
    # Evaluation settings
    parser.add_argument('--subset-size', type=int, help='Limit evaluation to N examples')
    
    return parser


def load_config_file(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)


def load_and_merge_config(args):
    """Load config file and merge values into args. Command line args take precedence."""
    if not args.config:
        return args
    
    config = load_config_file(args.config)
    
    # First, handle the metadata section specially - merge its contents directly
    if 'metadata' in config:
        metadata = config['metadata']
        for key, value in metadata.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    # Then handle all other config values, flattening nested structures
    def add_config_to_args(config_dict, prefix=''):
        for key, value in config_dict.items():
            if key in ['metadata', 'dataset']:  # Skip metadata and dataset as we handle them specially
                continue
            attr_name = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                add_config_to_args(value, attr_name)
            elif not hasattr(args, attr_name) or getattr(args, attr_name) is None:
                setattr(args, attr_name, value)
    
    add_config_to_args(config)
    return args

def test_multiagent_routing():
    """Test function to verify multi-agent routing is working"""
    from eval_framework import CompetitionKit
    
    print("\n" + "="*60)
    print("🧪 TESTING MULTI-AGENT ROUTING")
    print("="*60)
    
    # Initialize framework
    framework = CompetitionKit()
    framework.load_model("multiagent-router", model_type="multiagent")
    
    # Test different question types to verify routing
    test_questions = [
        {
            "question": "What is the recommended storage temperature for insulin?",
            "expected_agent": "STORAGE_HANDLING",
            "question_type": "multi_choice"
        },
        {
            "question": "Which brand name drug contains acetaminophen as active ingredient?",
            "expected_agent": "DRUG_BRANDS",
            "question_type": "multi_choice"
        },
        {
            "question": "What is the mechanism of action of metformin in diabetes treatment?",
            "expected_agent": "PHARMACOLOGY",
            "question_type": "open_ended_multi_choice"
        },
        {
            "question": "What are the contraindications for prescribing warfarin?",
            "expected_agent": "CONTRAINDICATIONS",
            "question_type": "open_ended_multi_choice"
        }
    ]
    
    print(f"\nTesting {len(test_questions)} routing scenarios...")
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n--- Test {i}/{len(test_questions)} ---")
        print(f"Question: {test['question'][:60]}...")
        print(f"Expected Agent: {test['expected_agent']}")
        
        try:
            response, messages = framework.model.inference(
                test['question'], 
                question_type=test['question_type']
            )
            
            # Check if routing worked as expected
            if test['expected_agent'] in response:
                print(f"✅ Correct routing detected!")
            else:
                print(f"❓ Different routing - check logs above")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Print final statistics
    if hasattr(framework.model, 'debug_stats'):
        stats = framework.model.debug_stats
        print(f"\n📊 FINAL ROUTING STATISTICS:")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Agents used: {list(stats['agent_usage'].keys())}")
        print(f"LLMs used: {list(stats['llm_usage'].keys())}")
    
    print("\n" + "="*60)
    print("🏁 MULTI-AGENT TESTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    test_multiagent_routing()
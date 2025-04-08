import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from collections import defaultdict
import logging
import re
import json
from tqdm import tqdm
import os
from datasets import load_dataset, load_metric

logger = logging.getLogger(__name__)

class FaroeseEvaluator:
    """
    Evaluator for Faroese language models.
    
    Provides metrics and evaluation utilities specific to Faroese language.
    """
    
    def __init__(
        self,
        metrics: List[str] = ["foqa", "perplexity", "bleu"],
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize Faroese evaluator.
        
        Args:
            metrics: List of metrics to compute
            device: Device to run evaluation on
            cache_dir: Cache directory for metrics and datasets
        """
        self.metrics = metrics
        self.device = device
        self.cache_dir = cache_dir
        
        # Initialize metrics
        self.metric_fns = {}
        for metric in self.metrics:
            if metric == "perplexity":
                # No need to load a metric for perplexity - compute directly
                pass
            elif metric == "bleu":
                self.metric_fns["bleu"] = load_metric("bleu", cache_dir=cache_dir)
            elif metric == "rouge":
                self.metric_fns["rouge"] = load_metric("rouge", cache_dir=cache_dir)
            elif metric == "foqa":
                # FoQA is our custom Faroese QA metric
                # Implemented in this class
                pass
            else:
                logger.warning(f"Unknown metric: {metric}")
    
    def evaluate(
        self,
        model,
        tokenizer,
        dataset,
        metric_names: Optional[List[str]] = None,
        batch_size: int = 8,
        max_length: int = 2048,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on specified metrics.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer for the model
            dataset: Dataset with prompts and references
            metric_names: Metrics to compute (if None, use self.metrics)
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
            gen_kwargs: Generation kwargs for the model
            
        Returns:
            Dictionary of metric name to score
        """
        if metric_names is None:
            metric_names = self.metrics
        
        # Check metrics are valid
        for metric in metric_names:
            if metric not in self.metrics and metric != "perplexity":
                raise ValueError(f"Unknown metric: {metric}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Prepare generation kwargs
        if gen_kwargs is None:
            gen_kwargs = {
                "max_new_tokens": 256,
                "do_sample": False,
                "temperature": 1.0,
                "num_beams": 1,
            }
        
        results = {}
        
        # Calculate each metric
        for metric in metric_names:
            if metric == "perplexity":
                # Calculate perplexity
                perplexity = self.calculate_perplexity(model, tokenizer, dataset, batch_size, max_length)
                results["perplexity"] = perplexity
            elif metric == "bleu":
                # Calculate BLEU score
                bleu_score = self.calculate_bleu(model, tokenizer, dataset, batch_size, max_length, gen_kwargs)
                results["bleu"] = bleu_score
            elif metric == "foqa":
                # Calculate FoQA score
                foqa_score = self.calculate_foqa(model, tokenizer, dataset, batch_size, max_length, gen_kwargs)
                results["foqa"] = foqa_score
            elif metric == "rouge":
                # Calculate ROUGE scores
                rouge_scores = self.calculate_rouge(model, tokenizer, dataset, batch_size, max_length, gen_kwargs)
                for key, value in rouge_scores.items():
                    results[f"rouge_{key}"] = value
        
        return results
    
    def calculate_perplexity(
        self,
        model,
        tokenizer,
        dataset,
        batch_size: int = 8,
        max_length: int = 2048,
    ) -> float:
        """
        Calculate perplexity on dataset.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            dataset: Dataset with text field
            batch_size: Batch size
            max_length: Maximum sequence length
            
        Returns:
            Perplexity score
        """
        nlls = []
        total_length = 0
        
        # Process dataset in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Calculating perplexity"):
            batch = dataset[i:i + batch_size]
            
            # Get input texts and encode
            if "text" in batch:
                texts = batch["text"]
            elif "response" in batch:
                texts = batch["response"]
            else:
                raise ValueError("Dataset must have 'text' or 'response' field")
            
            encodings = tokenizer(
                texts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            
            # Compute loss
            with torch.no_grad():
                outputs = model(**encodings, labels=encodings.input_ids)
                neg_log_likelihood = outputs.loss
            
            # Get number of tokens
            tokens = encodings.input_ids.ne(tokenizer.pad_token_id).sum().item()
            
            # Accumulate negative log likelihoods and token counts
            nlls.append(neg_log_likelihood.item() * tokens)
            total_length += tokens
        
        # Calculate perplexity
        avg_nll = sum(nlls) / total_length
        perplexity = torch.exp(torch.tensor(avg_nll)).item()
        
        return perplexity
    
    def calculate_bleu(
        self,
        model,
        tokenizer,
        dataset,
        batch_size: int = 8,
        max_length: int = 2048,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate BLEU score on dataset.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            dataset: Dataset with prompt and reference fields
            batch_size: Batch size
            max_length: Maximum sequence length
            gen_kwargs: Generation kwargs
            
        Returns:
            BLEU score
        """
        if gen_kwargs is None:
            gen_kwargs = {"max_new_tokens": 256}
        
        predictions = []
        references = []
        
        # Process dataset in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Calculating BLEU"):
            batch = dataset[i:i + batch_size]
            
            # Get prompts and generate responses
            prompts = batch["prompt"] if "prompt" in batch else batch["input"]
            
            # Tokenize and generate
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length - gen_kwargs.get("max_new_tokens", 256),
                return_tensors="pt",
            ).to(self.device)
            
            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **gen_kwargs,
                )
            
            # Decode generated texts
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Extract references
            refs = batch["reference"] if "reference" in batch else batch["response"]
            
            # Store predictions and references
            for gen, ref in zip(generated_texts, refs):
                predictions.append(gen)
                references.append([ref])  # BLEU expects list of references for each prediction
        
        # Calculate BLEU score
        bleu_score = self.metric_fns["bleu"].compute(
            predictions=[prediction.split() for prediction in predictions],
            references=[[reference[0].split()] for reference in references],
        )["bleu"]
        
        return bleu_score
    
    def calculate_rouge(
        self,
        model,
        tokenizer,
        dataset,
        batch_size: int = 8,
        max_length: int = 2048,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores on dataset.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            dataset: Dataset with prompt and reference fields
            batch_size: Batch size
            max_length: Maximum sequence length
            gen_kwargs: Generation kwargs
            
        Returns:
            Dictionary of ROUGE scores
        """
        if gen_kwargs is None:
            gen_kwargs = {"max_new_tokens": 256}
        
        predictions = []
        references = []
        
        # Process dataset in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Calculating ROUGE"):
            batch = dataset[i:i + batch_size]
            
            # Get prompts and generate responses
            prompts = batch["prompt"] if "prompt" in batch else batch["input"]
            
            # Tokenize and generate
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length - gen_kwargs.get("max_new_tokens", 256),
                return_tensors="pt",
            ).to(self.device)
            
            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **gen_kwargs,
                )
            
            # Decode generated texts
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Extract references
            refs = batch["reference"] if "reference" in batch else batch["response"]
            
            # Store predictions and references
            predictions.extend(generated_texts)
            references.extend(refs)
        
        # Calculate ROUGE scores
        rouge_scores = self.metric_fns["rouge"].compute(
            predictions=predictions,
            references=references,
        )
        
        # Extract scores (e.g., rouge1, rouge2, rougeL)
        results = {}
        for key, value in rouge_scores.items():
            results[key] = value.mid.fmeasure
        
        return results
    
    def calculate_foqa(
        self,
        model,
        tokenizer,
        dataset,
        batch_size: int = 8,
        max_length: int = 2048,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Calculate Faroese Question Answering score (FoQA).
        
        This metric evaluates both factual correctness and linguistic quality
        of Faroese-language responses to questions.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            dataset: FoQA dataset with questions and answers
            batch_size: Batch size
            max_length: Maximum sequence length
            gen_kwargs: Generation kwargs
            
        Returns:
            FoQA score (average of correctness and linguistic scores)
        """
        if gen_kwargs is None:
            gen_kwargs = {"max_new_tokens": 256}
        
        predictions = []
        references = []
        questions = []
        
        # Process dataset in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc="Calculating FoQA"):
            batch = dataset[i:i + batch_size]
            
            # Get questions
            query = batch["question"] if "question" in batch else batch["prompt"]
            questions.extend(query)
            
            # Tokenize and generate
            inputs = tokenizer(
                query,
                padding=True,
                truncation=True,
                max_length=max_length - gen_kwargs.get("max_new_tokens", 256),
                return_tensors="pt",
            ).to(self.device)
            
            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **gen_kwargs,
                )
            
            # Decode generated texts
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Extract references
            refs = batch["answer"] if "answer" in batch else batch["response"]
            
            # Store predictions and references
            predictions.extend(generated_texts)
            references.extend(refs)
        
        # Calculate correctness score based on content overlap
        correctness_scores = []
        for pred, ref in zip(predictions, references):
            correctness = self._calculate_content_overlap(pred, ref)
            correctness_scores.append(correctness)
        
        # Calculate linguistic quality score for Faroese
        linguistic_scores = []
        for pred in predictions:
            linguistic = self._evaluate_faroese_quality(pred)
            linguistic_scores.append(linguistic)
        
        # Overall FoQA score is average of correctness and linguistic scores
        correctness_avg = sum(correctness_scores) / len(correctness_scores)
        linguistic_avg = sum(linguistic_scores) / len(linguistic_scores)
        foqa_score = (correctness_avg + linguistic_avg) / 2
        
        # Create detailed results for logging
        detailed_results = {
            "correctness": correctness_avg,
            "linguistic": linguistic_avg,
            "foqa": foqa_score,
            "examples": [
                {
                    "question": q,
                    "prediction": p,
                    "reference": r,
                    "correctness": c,
                    "linguistic": l,
                }
                for q, p, r, c, l in zip(
                    questions[:10],  # Just show first 10 examples
                    predictions[:10],
                    references[:10],
                    correctness_scores[:10],
                    linguistic_scores[:10],
                )
            ],
        }
        
        # Log detailed results
        logger.info(f"FoQA Detailed Results: {json.dumps(detailed_results, indent=2, ensure_ascii=False)}")
        
        return foqa_score
    
    def _calculate_content_overlap(self, prediction: str, reference: str) -> float:
        """
        Calculate semantic content overlap between prediction and reference.
        
        This is a simple implementation based on word overlap.
        A more sophisticated approach would use Faroese-specific word embeddings.
        
        Args:
            prediction: Predicted text
            reference: Reference text
            
        Returns:
            Overlap score between 0 and 1
        """
        # Normalize and tokenize
        pred_words = set(self._normalize_faroese(prediction).split())
        ref_words = set(self._normalize_faroese(reference).split())
        
        # Calculate overlap (Jaccard similarity)
        if not pred_words or not ref_words:
            return 0.0
        
        intersection = pred_words.intersection(ref_words)
        union = pred_words.union(ref_words)
        
        return len(intersection) / len(union)
    
    def _evaluate_faroese_quality(self, text: str) -> float:
        """
        Evaluate linguistic quality of Faroese text.
        
        This checks for proper use of Faroese characters and common patterns.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Quality score between 0 and 1
        """
        # This is a simple heuristic implementation
        # A full implementation would use Faroese grammar rules and spell checking
        
        # Check presence of Faroese-specific characters
        faroese_chars = "ðøáíóúýæÐØÁÍÓÚÝÆ"
        has_faroese_chars = any(c in text for c in faroese_chars)
        
        # Check for common Faroese word patterns
        common_faroese_patterns = [
            r"\bfør", r"\bík", r"\bæ\w+", r"\bø\w+", r"\bð\w+",
            r"\btó", r"\bmen", r"\bogso", r"\bhelst", r"\bikki"
        ]
        
        pattern_matches = sum(1 for pattern in common_faroese_patterns if re.search(pattern, text))
        pattern_score = min(1.0, pattern_matches / 5)  # Normalize to max of 1.0
        
        # Combine scores, giving more weight to Faroese characters
        if has_faroese_chars:
            return 0.7 + 0.3 * pattern_score
        else:
            return 0.3 * pattern_score
    
    def _normalize_faroese(self, text: str) -> str:
        """
        Normalize Faroese text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Standardize Faroese characters
        replacements = {
            "dh": "ð",
            "oe": "ø",
            "aa": "á",
            "ii": "í",
            "oo": "ó",
            "uu": "ú",
            "yy": "ý",
            "ae": "æ",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text 
#!/usr/bin/env python3
"""
Evaluate RAG Pipeline using Q&A Dataset

This script evaluates the RAG pipeline using a Q&A dataset with reference answers.
It uses LLM-as-a-judge to assess answer quality and tracks retrieval metrics.

Usage:
    python eval_rag_pipeline.py --qa-dataset path/to/qa.json [--persist-dir PERSIST_DIR] [--output results.json]
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import time

# Ensure parent directory is in path for module imports when run directly
_PARENT_DIR = str(Path(__file__).parent.parent.absolute())
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from golden.golden_retriever import Golden_Retriever
from core.config import RAG_PERSIST_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result for a single Q&A pair."""
    idx: int
    question: str
    retrieved_docs_count: int
    llm_score: float  # 0-100 scale
    llm_reasoning: str
    reference_answer: str
    retrieved_context: str
    eval_time_seconds: float
    error: str = None


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all Q&A pairs."""
    total_questions: int
    avg_llm_score: float
    median_llm_score: float
    avg_retrieval_count: int
    questions_with_errors: int
    total_eval_time_seconds: float


def load_qa_dataset(qa_path: str) -> List[Dict[str, Any]]:
    """
    Load Q&A dataset from JSON file.

    Args:
        qa_path: Path to Q&A JSON file

    Returns:
        List of Q&A dictionaries
    """
    logger.info(f"Loading Q&A dataset from {qa_path}")
    with open(qa_path, 'r') as f:
        qa_data = json.load(f)

    logger.info(f"Loaded {len(qa_data)} Q&A pairs")
    return qa_data


def load_rag_retriever(
    persist_dir: str = RAG_PERSIST_DIR,
) -> Golden_Retriever:
    """
    Load the RAG retriever from disk.

    Args:
        persist_dir: Directory where vector database is persisted

    Returns:
        Loaded Golden_Retriever instance
    """
    logger.info(f"Loading RAG retriever from {persist_dir}")

    db = Golden_Retriever.load(
        folder_path=persist_dir,
        similarity_fn="cosine"
    )

    doc_count = db._collection.count()
    logger.info(f"Loaded vector database with {doc_count:,} chunks")

    return db


def retrieve_context(
    retriever: Golden_Retriever,
    query: str,
    k: int = 5
) -> Tuple[str, int]:
    """
    Retrieve relevant context for a query.

    Args:
        retriever: Golden_Retriever instance
        query: Query string
        k: Number of results to return

    Returns:
        Tuple of (concatenated context, number of docs retrieved)
    """
    results = retriever.similarity_search(query, k=k)

    # Concatenate retrieved documents
    context_parts = []
    for i, doc in enumerate(results, 1):
        context_parts.append(f"[Document {i}]\n{doc.page_content}\n")

    context = "\n".join(context_parts)
    return context, len(results)


def llm_as_judge_prompt(question: str, reference_answer: str, retrieved_context: str) -> str:
    """
    Create a prompt for LLM-as-a-judge evaluation.

    Args:
        question: The question asked
        reference_answer: The reference/ground truth answer
        retrieved_context: The context retrieved by RAG

    Returns:
        Evaluation prompt string
    """
    prompt = f"""You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems. Your task is to assess whether the retrieved context contains sufficient information to answer the question accurately when compared to a reference answer.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{retrieved_context}

EVALUATION CRITERIA:
1. **Relevance**: Does the retrieved context contain information relevant to the question?
2. **Completeness**: Does the context contain all the key information present in the reference answer?
3. **Accuracy**: Is the information in the context consistent with the reference answer?
4. **Coverage**: Are all important points from the reference answer covered in the retrieved documents?

Please provide:
1. A score from 0-100 where:
   - 90-100: Excellent - Retrieved context fully covers all aspects of the reference answer
   - 70-89: Good - Retrieved context covers most key points with minor gaps
   - 50-69: Fair - Retrieved context covers some key points but has significant gaps
   - 30-49: Poor - Retrieved context has limited relevant information
   - 0-29: Very Poor - Retrieved context is mostly irrelevant or missing critical information

2. A brief reasoning explaining your score (2-3 sentences)

Format your response as JSON:
{{
  "score": <0-100>,
  "reasoning": "<your reasoning here>"
}}
"""
    return prompt


def evaluate_with_llm(
    question: str,
    reference_answer: str,
    retrieved_context: str,
    model: str = "gpt-4",
    use_anthropic: bool = True
) -> Tuple[float, str]:
    """
    Evaluate the retrieved context using LLM as a judge.

    Args:
        question: The question
        reference_answer: Reference answer
        retrieved_context: Retrieved context from RAG
        model: Model to use for evaluation
        use_anthropic: Whether to use Anthropic Claude (vs OpenAI)

    Returns:
        Tuple of (score, reasoning)
    """
    prompt = llm_as_judge_prompt(question, reference_answer, retrieved_context)

    try:
        if use_anthropic:
            # Use Anthropic Claude
            import anthropic
            client = anthropic.Anthropic()

            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response_text = message.content[0].text

        else:
            # Use OpenAI
            from openai import OpenAI
            client = OpenAI()

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for RAG systems. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )

            response_text = response.choices[0].message.content

        # Parse JSON response
        # Try to extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        score = float(result["score"])
        reasoning = result["reasoning"]

        return score, reasoning

    except Exception as e:
        logger.error(f"Error in LLM evaluation: {e}")
        return 0.0, f"Error during evaluation: {str(e)}"


def evaluate_single_qa(
    qa_item: Dict[str, Any],
    retriever: Golden_Retriever,
    top_k: int = 5,
    use_llm_judge: bool = True,
    use_anthropic: bool = True
) -> EvalResult:
    """
    Evaluate a single Q&A pair.

    Args:
        qa_item: Q&A dictionary with 'idx', 'prompt', 'response'
        retriever: Golden_Retriever instance
        top_k: Number of documents to retrieve
        use_llm_judge: Whether to use LLM as judge
        use_anthropic: Whether to use Anthropic Claude

    Returns:
        EvalResult object
    """
    start_time = time.time()

    idx = qa_item["idx"]
    question = qa_item["prompt"]
    reference_answer = qa_item["response"]

    logger.info(f"Evaluating Q&A #{idx}: {question[:100]}...")

    try:
        # Retrieve context
        retrieved_context, num_docs = retrieve_context(retriever, question, k=top_k)

        # Evaluate with LLM
        if use_llm_judge and reference_answer.strip() and reference_answer != "I am not sure.":
            llm_score, llm_reasoning = evaluate_with_llm(
                question,
                reference_answer,
                retrieved_context,
                use_anthropic=use_anthropic
            )
        else:
            # Skip LLM evaluation if no reference answer
            llm_score = -1.0
            llm_reasoning = "Skipped: No reference answer available"

        eval_time = time.time() - start_time

        return EvalResult(
            idx=idx,
            question=question,
            retrieved_docs_count=num_docs,
            llm_score=llm_score,
            llm_reasoning=llm_reasoning,
            reference_answer=reference_answer[:500],  # Truncate for output
            retrieved_context=retrieved_context[:500],  # Truncate for output
            eval_time_seconds=eval_time,
            error=None
        )

    except Exception as e:
        logger.error(f"Error evaluating Q&A #{idx}: {e}")
        eval_time = time.time() - start_time

        return EvalResult(
            idx=idx,
            question=question,
            retrieved_docs_count=0,
            llm_score=0.0,
            llm_reasoning="",
            reference_answer=reference_answer[:500],
            retrieved_context="",
            eval_time_seconds=eval_time,
            error=str(e)
        )


def compute_aggregate_metrics(results: List[EvalResult]) -> AggregateMetrics:
    """
    Compute aggregate metrics from evaluation results.

    Args:
        results: List of EvalResult objects

    Returns:
        AggregateMetrics object
    """
    valid_scores = [r.llm_score for r in results if r.llm_score >= 0 and r.error is None]

    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        median_score = sorted(valid_scores)[len(valid_scores) // 2]
    else:
        avg_score = 0.0
        median_score = 0.0

    avg_retrieval = sum(r.retrieved_docs_count for r in results) / len(results) if results else 0
    errors = sum(1 for r in results if r.error is not None)
    total_time = sum(r.eval_time_seconds for r in results)

    return AggregateMetrics(
        total_questions=len(results),
        avg_llm_score=avg_score,
        median_llm_score=median_score,
        avg_retrieval_count=avg_retrieval,
        questions_with_errors=errors,
        total_eval_time_seconds=total_time
    )


def print_results_summary(results: List[EvalResult], metrics: AggregateMetrics):
    """Print evaluation results summary."""
    print("\n" + "=" * 80)
    print("RAG EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTotal Questions Evaluated: {metrics.total_questions}")
    print(f"Average LLM Score: {metrics.avg_llm_score:.2f}/100")
    print(f"Median LLM Score: {metrics.median_llm_score:.2f}/100")
    print(f"Average Retrieved Docs: {metrics.avg_retrieval_count:.1f}")
    print(f"Questions with Errors: {metrics.questions_with_errors}")
    print(f"Total Evaluation Time: {metrics.total_eval_time_seconds:.2f}s")

    # Score distribution
    score_ranges = {
        "Excellent (90-100)": sum(1 for r in results if 90 <= r.llm_score <= 100),
        "Good (70-89)": sum(1 for r in results if 70 <= r.llm_score < 90),
        "Fair (50-69)": sum(1 for r in results if 50 <= r.llm_score < 70),
        "Poor (30-49)": sum(1 for r in results if 30 <= r.llm_score < 50),
        "Very Poor (0-29)": sum(1 for r in results if 0 <= r.llm_score < 30),
    }

    print("\nScore Distribution:")
    for range_name, count in score_ranges.items():
        print(f"  {range_name}: {count}")

    # Top and bottom performers
    valid_results = [r for r in results if r.llm_score >= 0]
    if valid_results:
        sorted_results = sorted(valid_results, key=lambda x: x.llm_score, reverse=True)

        print("\n" + "-" * 80)
        print("TOP 3 PERFORMING QUESTIONS:")
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"\n{i}. Q#{result.idx} (Score: {result.llm_score:.1f})")
            print(f"   Question: {result.question[:100]}...")
            print(f"   Reasoning: {result.llm_reasoning}")

        print("\n" + "-" * 80)
        print("BOTTOM 3 PERFORMING QUESTIONS:")
        for i, result in enumerate(sorted_results[-3:], 1):
            print(f"\n{i}. Q#{result.idx} (Score: {result.llm_score:.1f})")
            print(f"   Question: {result.question[:100]}...")
            print(f"   Reasoning: {result.llm_reasoning}")

    print("\n" + "=" * 80)


def save_results(results: List[EvalResult], metrics: AggregateMetrics, output_path: str):
    """Save evaluation results to JSON file."""
    output_data = {
        "aggregate_metrics": asdict(metrics),
        "detailed_results": [asdict(r) for r in results]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved evaluation results to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline using Q&A dataset"
    )
    parser.add_argument(
        "--qa-dataset",
        type=str,
        required=True,
        help="Path to Q&A dataset JSON file"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=RAG_PERSIST_DIR,
        help="Directory where vector database is persisted"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query"
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM-as-judge evaluation (only compute retrieval metrics)"
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI instead of Anthropic Claude for LLM judge"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of Q&A pairs to evaluate (for testing)"
    )

    args = parser.parse_args()

    # Load Q&A dataset
    qa_data = load_qa_dataset(args.qa_dataset)

    # Limit if specified
    if args.limit:
        qa_data = qa_data[:args.limit]
        logger.info(f"Limited to first {args.limit} Q&A pairs")

    # Load RAG retriever
    retriever = load_rag_retriever(persist_dir=args.persist_dir)

    # Evaluate each Q&A pair
    results = []
    use_anthropic = not args.use_openai

    for qa_item in qa_data:
        result = evaluate_single_qa(
            qa_item,
            retriever,
            top_k=args.top_k,
            use_llm_judge=not args.no_llm_judge,
            use_anthropic=use_anthropic
        )
        results.append(result)

    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(results)

    # Print summary
    print_results_summary(results, metrics)

    # Save results
    save_results(results, metrics, args.output)


if __name__ == "__main__":
    main()

# RAG Pipeline Evaluation Guide

## Overview

This guide explains how to evaluate the RAG (Retrieval-Augmented Generation) pipeline using the Q&A dataset with reference answers.

## Evaluation Approaches

### 1. LLM-as-a-Judge (Recommended) ⭐

**What it is:**
- Uses a large language model (Claude or GPT-4) to evaluate if the retrieved context contains sufficient information to answer the question when compared to a reference answer
- Provides nuanced assessment of semantic similarity, completeness, and accuracy

**Advantages:**
- ✅ Captures semantic similarity (not just word overlap)
- ✅ Evaluates completeness and coverage of information
- ✅ Provides interpretable reasoning for each score
- ✅ Handles paraphrasing and different phrasings well

**Disadvantages:**
- ❌ Costs money (API calls)
- ❌ Slower than traditional metrics
- ❌ May have some variance in scoring

**When to use:**
- When you need accurate semantic evaluation
- When reference answers vary in phrasing from retrieved content
- For final/production evaluation

### 2. Traditional Metrics (BLEU, ROUGE, BERTScore)

**What they are:**
- BLEU: Measures n-gram overlap (originally for machine translation)
- ROUGE: Measures recall of n-grams (originally for summarization)
- BERTScore: Computes semantic similarity using BERT embeddings

**Advantages:**
- ✅ Fast and cheap
- ✅ Deterministic and reproducible
- ✅ No API costs

**Disadvantages:**
- ❌ Mainly measure word overlap, not semantic meaning
- ❌ Don't capture completeness well
- ❌ Can give high scores to irrelevant but overlapping text

**When to use:**
- For quick iterative testing during development
- When you need deterministic metrics
- As complementary metrics alongside LLM judge

### 3. Retrieval-Only Metrics

**What they are:**
- Measures how well the RAG system retrieves relevant documents
- Metrics: Precision@K, Recall@K, MRR (Mean Reciprocal Rank)
- Requires knowing which documents should be retrieved (citation matching)

**Advantages:**
- ✅ Fast to compute
- ✅ Isolates retrieval performance
- ✅ Useful for debugging retrieval issues

**When to use:**
- When you have ground-truth document citations
- For debugging retrieval vs. generation issues
- As a component of overall evaluation

## Our Implementation

The `eval_rag_pipeline.py` script implements **LLM-as-a-Judge** evaluation:

### Evaluation Criteria

1. **Relevance**: Does the retrieved context contain information relevant to the question?
2. **Completeness**: Does the context contain all key information from the reference answer?
3. **Accuracy**: Is the information consistent with the reference answer?
4. **Coverage**: Are all important points covered?

### Scoring Scale (0-100)

- **90-100**: Excellent - Retrieved context fully covers all aspects
- **70-89**: Good - Covers most key points with minor gaps
- **50-69**: Fair - Covers some key points but has significant gaps
- **30-49**: Poor - Limited relevant information
- **0-29**: Very Poor - Mostly irrelevant or missing critical information

## Usage

### Basic Usage

```bash
# Evaluate using Anthropic Claude (default)
python eval_rag_pipeline.py \
    --qa-dataset /home/ec2-user/git/data_sanitizer/data/Army\ Account\ Manger\ FY\ 2026\ Defense\ Budget\ Q\&As/defense-budget-fy26-QAs.json \
    --output eval_results.json
```

### Using OpenAI Instead

```bash
# Evaluate using OpenAI GPT-4
python eval_rag_pipeline.py \
    --qa-dataset path/to/qa.json \
    --use-openai \
    --output eval_results.json
```

### Test on Small Subset

```bash
# Evaluate only first 3 questions (for testing)
python eval_rag_pipeline.py \
    --qa-dataset path/to/qa.json \
    --limit 3 \
    --output test_results.json
```

### Retrieval-Only Mode

```bash
# Skip LLM evaluation, only check retrieval
python eval_rag_pipeline.py \
    --qa-dataset path/to/qa.json \
    --no-llm-judge \
    --output retrieval_results.json
```

### Custom Parameters

```bash
# Customize retrieval (k=10) and persist directory
python eval_rag_pipeline.py \
    --qa-dataset path/to/qa.json \
    --persist-dir /custom/path/to/chroma \
    --top-k 10 \
    --output eval_results.json
```

## Output

### Terminal Output

The script prints a summary including:
- Average and median LLM scores
- Score distribution (Excellent, Good, Fair, Poor, Very Poor)
- Top 3 and Bottom 3 performing questions
- Total evaluation time

Example:
```
================================================================================
RAG EVALUATION RESULTS SUMMARY
================================================================================

Total Questions Evaluated: 11
Average LLM Score: 72.45/100
Median LLM Score: 75.00/100
Average Retrieved Docs: 5.0
Questions with Errors: 0
Total Evaluation Time: 156.78s

Score Distribution:
  Excellent (90-100): 2
  Good (70-89): 5
  Fair (50-69): 2
  Poor (30-49): 0
  Very Poor (0-29): 0
```

### JSON Output File

The output JSON file contains:
```json
{
  "aggregate_metrics": {
    "total_questions": 11,
    "avg_llm_score": 72.45,
    "median_llm_score": 75.0,
    "avg_retrieval_count": 5.0,
    "questions_with_errors": 0,
    "total_eval_time_seconds": 156.78
  },
  "detailed_results": [
    {
      "idx": 1,
      "question": "List all the PEs and Projects...",
      "retrieved_docs_count": 5,
      "llm_score": 85.0,
      "llm_reasoning": "The retrieved context contains most of the key information...",
      "reference_answer": "The following Program Elements...",
      "retrieved_context": "[Document 1]\n...",
      "eval_time_seconds": 14.23,
      "error": null
    }
  ]
}
```

## Requirements

Install required dependencies:

```bash
# For Anthropic Claude (recommended)
pip install anthropic

# For OpenAI
pip install openai

# Already in your environment (from RAG pipeline)
# - langchain-community
# - chromadb
```

## Environment Setup

### For Anthropic Claude

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### For OpenAI

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Cost Considerations

### Anthropic Claude (claude-3-5-sonnet)
- **Input**: ~$3 per million tokens
- **Output**: ~$15 per million tokens
- **Estimated cost per question**: $0.01-0.03
- **For 11 questions**: ~$0.15-0.35

### OpenAI GPT-4
- **Input**: ~$10 per million tokens
- **Output**: ~$30 per million tokens
- **Estimated cost per question**: $0.03-0.08
- **For 11 questions**: ~$0.35-0.90

## Advanced Evaluation Strategies

### 1. Hybrid Evaluation

Combine multiple metrics for comprehensive evaluation:

```python
# Pseudo-code
final_score = (
    0.5 * llm_judge_score +
    0.3 * retrieval_precision +
    0.2 * bertscore
)
```

### 2. Citation-Based Evaluation

Compare retrieved documents against ground-truth citations:

```python
# Check if correct documents were retrieved
def citation_recall(retrieved_docs, ground_truth_citations):
    retrieved_sources = {doc.metadata['source'] for doc in retrieved_docs}
    gt_sources = set(ground_truth_citations.values())
    return len(retrieved_sources & gt_sources) / len(gt_sources)
```

### 3. Error Analysis

Categorize failure modes:
- **Retrieval Failure**: Correct documents not retrieved
- **Incomplete Retrieval**: Some but not all relevant documents retrieved
- **Irrelevant Retrieval**: Wrong documents retrieved

## Troubleshooting

### Issue: "No module named 'anthropic'"
**Solution**: `pip install anthropic`

### Issue: API key not found
**Solution**: Set environment variable:
```bash
export ANTHROPIC_API_KEY="your-key"
```

### Issue: Evaluation too slow
**Solutions**:
1. Use `--limit` to test on smaller subset
2. Use `--no-llm-judge` for faster retrieval-only metrics
3. Use cheaper/faster model (though less accurate)

### Issue: Low scores across the board
**Possible causes**:
1. RAG retrieval not finding relevant documents (check `retrieved_docs_count`)
2. Chunking strategy needs adjustment
3. Embedding model not suitable for domain
4. Top-k parameter too low (try `--top-k 10`)

## Best Practices

1. **Start Small**: Test on 3-5 questions first with `--limit 3`
2. **Check Retrieval First**: Use `--no-llm-judge` to verify documents are being retrieved
3. **Review Edge Cases**: Examine top and bottom performers to understand patterns
4. **Iterate**: Adjust chunking, top-k, or embedding model based on results
5. **Track Over Time**: Save results with timestamps to track improvements

## Next Steps

After evaluation, you can:
1. **Improve Retrieval**: Adjust chunk size, overlap, or top-k parameters
2. **Enhance Embeddings**: Try different embedding models
3. **Add Reranking**: Use a reranker to improve top-k results
4. **Tune Prompts**: If using RAG for generation, optimize prompts
5. **Add Metadata Filtering**: Filter by document type or date range

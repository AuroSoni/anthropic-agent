"""Script to process questions from CSV using Anthropic Agent."""
import asyncio
import csv
import os
from pathlib import Path
from anthropic_agent.core import AnthropicAgent
from anthropic_agent.database import FilesystemBackend
from datetime import datetime

# Configuration
NUM_QUESTIONS = 20
MAX_PARALLELISM = 5
INPUT_CSV = "Potential Q&A - Ques Only.csv"
OUTPUT_CSV = "Potential Q&A - With Answers.csv"
AGENT_DATA_DIR = "./temp/nova_labs/agent_data"
today_date = datetime.now().strftime("%Y-%m-%d")

def collate_final_text(message) -> str:
    """
    Collate all text blocks after the last server tool use/result block.
    
    Args:
        message: The ParsedBetaMessage (e.g., result.final_message)
        
    Returns:
        Concatenated text from all text blocks after the last tool block
    """
    content = message.content
    
    # Find the index of the last tool-related block
    last_tool_idx = -1
    tool_types = (
        'server_tool_use', 'tool_use', 'tool_result',
        'web_search_tool_result', 'web_fetch_tool_result', 'code_execution_tool_result'
    )
    for i, block in enumerate(content):
        if block.type in tool_types:
            last_tool_idx = i
    
    # Get all text blocks after the last tool block
    text_blocks = []
    for block in content[last_tool_idx + 1:]:
        if hasattr(block, 'text') and block.text is not None:
            text_blocks.append(block.text)
    
    return "".join(text_blocks)


SYSTEM_PROMPT = f"""Today's date is {today_date}.
You are a helpful financial and business analyst assistant. 
You are tasked with answering questions about Paytm, an Indian fintech company. 
Provide clear, concise, and accurate answers based on the question asked.
If you don't have enough information to answer accurately, say so."""


async def process_question(agent: AnthropicAgent, question: str, index: int, total: int) -> str:
    """Process a single question with the agent.
    
    Args:
        agent: The AnthropicAgent instance
        question: The question text to process
        index: The question index (1-based) for logging
        total: Total number of questions
        
    Returns:
        The answer text or "ERROR" if processing failed
    """
    try:
        print(f"[{index}/{total}] Processing question...")
        print(f"Question: {question[:80]}..." if len(question) > 80 else f"Question: {question}")
        
        result = await agent.run(prompt=question)
        
        if result.final_message and result.final_message.content:
            answer = collate_final_text(result.final_message)
            print(f"[{index}/{total}] ✓ Completed (conversation length: {len(result.conversation_history)} messages)")
            return answer
        else:
            print(f"[{index}/{total}] ✗ Failed: No response from agent")
            return "ERROR: No response from agent"
            
    except Exception as e:
        print(f"[{index}/{total}] ✗ Failed with error: {str(e)}")
        return f"ERROR: {str(e)}"


def create_agent() -> AnthropicAgent:
    """Create a new AnthropicAgent instance.
    
    Returns:
        A fresh AnthropicAgent configured for Paytm Q&A
    """
    print("Initializing agent...")
    file_backend = FilesystemBackend(base_path=AGENT_DATA_DIR)
    agent = AnthropicAgent(
        system_prompt=SYSTEM_PROMPT,
        model="claude-sonnet-4-5",
        db_backend=file_backend,
        server_tools=[{
            "type": "code_execution_20250825",
            "name": "code_execution"
        },
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 50
        },
        {
            "type": "web_fetch_20250910",
            "name": "web_fetch",
            "max_uses": 50
        }],
        beta_headers=["code-execution-2025-08-25", "web-fetch-2025-09-10", "files-api-2025-04-14"]
    )
    print(f"Agent UUID: {agent.agent_uuid}")
    return agent


async def process_question_with_semaphore(
    semaphore: asyncio.Semaphore,
    question_num: str,
    question_text: str,
    index: int,
    total: int,
    all_results: list,
    completed_count: list,
    lock: asyncio.Lock,
    output_path: Path
) -> tuple[int, str, str, str]:
    """Process a question with semaphore control and thread-safe result storage.
    
    Args:
        semaphore: Semaphore to limit concurrent execution
        question_num: Original question number from CSV
        question_text: The question text
        index: Question index (0-based) for result ordering
        total: Total number of questions
        all_results: Shared list to store results
        completed_count: Shared counter for completed questions
        lock: Lock for thread-safe operations
        output_path: Path to save results
        
    Returns:
        Tuple of (index, question_num, question_text, answer)
    """
    async with semaphore:
        # Create a fresh agent instance for this question
        agent = create_agent()
        
        # Process the question
        answer = await process_question(agent, question_text, index + 1, total)
        
        # Thread-safe result storage and saving
        async with lock:
            all_results[index] = (question_num, question_text, answer)
            completed_count[0] += 1
            
            # Get all completed results in order
            sorted_results = [r for r in all_results if r is not None]
            
            # Save incrementally
            save_results(str(output_path), sorted_results)
            
            print(f"✓ Progress: {completed_count[0]}/{total} questions completed, saved to {OUTPUT_CSV}")
            print("-" * 80)
            print()
        
        return (index, question_num, question_text, answer)


def read_questions(csv_path: str, num_questions: int) -> list[tuple[str, str]]:
    """Read questions from CSV file.
    
    Args:
        csv_path: Path to the input CSV file
        num_questions: Number of questions to read
        
    Returns:
        List of (question_number, question_text) tuples
    """
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_questions:
                break
            questions.append((row['#'], row['Question']))
    return questions


def save_results(csv_path: str, results: list[tuple[str, str, str]]) -> None:
    """Save results to CSV file.
    
    Args:
        csv_path: Path to the output CSV file
        results: List of (question_number, question, answer) tuples
    """
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(['#', 'Question', 'Answer'])
        writer.writerows(results)


async def main():
    """Main function to process questions."""
    print("=" * 80)
    print("Anthropic Agent - CSV Question Processing")
    print("=" * 80)
    print()
    
    # Get script directory and set up paths
    script_dir = Path(__file__).parent
    input_path = script_dir / INPUT_CSV
    output_path = script_dir / OUTPUT_CSV
    
    # Verify input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    # Create agent data directory if it doesn't exist
    os.makedirs(AGENT_DATA_DIR, exist_ok=True)
    print("Agent instances will be created per question for isolation.")
    print()
    
    # Read questions
    print(f"Reading {NUM_QUESTIONS} questions from {INPUT_CSV}...")
    questions = read_questions(str(input_path), NUM_QUESTIONS)
    print(f"Loaded {len(questions)} questions")
    print(f"Processing with max parallelism: {MAX_PARALLELISM}")
    print()
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(MAX_PARALLELISM)
    lock = asyncio.Lock()
    
    # Initialize shared data structures
    all_results = [None] * len(questions)  # Pre-allocated list to maintain order
    completed_count = [0]  # Mutable counter
    
    # Create all tasks for parallel execution
    tasks = [
        process_question_with_semaphore(
            semaphore=semaphore,
            question_num=question_num,
            question_text=question_text,
            index=i,
            total=len(questions),
            all_results=all_results,
            completed_count=completed_count,
            lock=lock,
            output_path=output_path
        )
        for i, (question_num, question_text) in enumerate(questions)
    ]
    
    # Process all questions concurrently
    print("Starting parallel processing...")
    print("=" * 80)
    await asyncio.gather(*tasks)
    
    # Get final results (should all be populated now)
    results = [r for r in all_results if r is not None]
    
    # Final summary
    print()
    print("=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Processed {len(results)} questions")
    print(f"Results saved to: {output_path}")
    
    # Count errors
    error_count = sum(1 for _, _, answer in results if answer.startswith("ERROR"))
    if error_count > 0:
        print(f"⚠ Warning: {error_count} question(s) had errors")
    else:
        print("✓ All questions processed successfully")


if __name__ == "__main__":
    asyncio.run(main())


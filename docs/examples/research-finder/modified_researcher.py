#!/usr/bin/env python3
"""
Research Finder CLI - A tool to research topics or questions using Perplexity's Sonar API.
Supports organization-specific research and batch processing of CSV files.
"""

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Any, List
from dotenv import load_dotenv
load_dotenv()

import requests
from requests.exceptions import RequestException

class ResearchAssistant:
    """A class to interact with Perplexity Sonar API for research."""

    API_URL = "https://api.perplexity.ai/chat/completions"
    DEFAULT_MODEL = "sonar-pro" # Using sonar-pro for potentially better research capabilities
    PROMPT_FILE = "system_prompt.md"

    def __init__(self, api_key: Optional[str] = None, prompt_file: Optional[str] = None):
        """
        Initialize the ResearchAssistant with API key and system prompt.

        Args:
            api_key: Perplexity API key. If None, will try to read from file or environment.
            prompt_file: Path to file containing the system prompt. If None, uses default relative path.
        """
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            raise ValueError(
                "API key not found. Please provide via argument, environment variable (PPLX_API_KEY), or key file."
            )

        # Construct path relative to this script's directory if prompt_file is not absolute
        script_dir = Path(__file__).parent
        prompt_path = Path(prompt_file) if prompt_file else script_dir / self.PROMPT_FILE
        if not prompt_path.is_absolute() and prompt_file:
             # If a relative path was given via argument, resolve it relative to CWD
             prompt_path = Path.cwd() / prompt_file
        elif not prompt_path.is_absolute():
             # Default case: resolve relative to script dir
             prompt_path = script_dir / self.PROMPT_FILE

        self.system_prompt = self._load_system_prompt(prompt_path)

    def _get_api_key(self) -> str:
        """
        Try to get API key from environment or from a file in the script's directory or CWD.

        Returns:
            The API key if found, empty string otherwise.
        """
        api_key = os.environ.get("PPLX_API_KEY", "")
        if api_key:
            return api_key

        # Check in current working directory and script's directory
        search_dirs = [Path.cwd(), Path(__file__).parent]
        key_filenames = ["pplx_api_key", ".pplx_api_key", "PPLX_API_KEY", ".PPLX_API_KEY"]

        for directory in search_dirs:
            for key_file in key_filenames:
                key_path = directory / key_file
                if key_path.exists():
                    try:
                        return key_path.read_text().strip()
                    except Exception:
                        pass # Ignore errors reading key file

        return ""

    def _load_system_prompt(self, prompt_path: Path) -> str:
        """
        Load the system prompt from a file.

        Args:
            prompt_path: Absolute path to the file containing the system prompt

        Returns:
            The system prompt as a string
        """
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Prompt file not found at {prompt_path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not load system prompt from {prompt_path}: {e}", file=sys.stderr)

        # Fallback default prompt if file loading fails
        print("Using fallback default system prompt.", file=sys.stderr)
        return (
            "You are an AI equity research assistant. Your task is to research the user's query, "
            "provide a concise summary in 300 words or less and list the sources used."
        )

    def research_topic(self, query: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """
        Research a given topic or question using the Perplexity API.

        Args:
            query: The research question or topic.
            model: The Perplexity model to use.

        Returns:
            A dictionary containing the research results or an error message.
        """
        if not query or not query.strip():
            return {"error": "Input query is empty. Cannot perform research."}

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ]
            # Add other parameters like temperature, max_tokens if needed
            ,"temperature": 0,
            # "max_tokens": 512
        }

        try:
            # Increased timeout for potentially longer research tasks
            response = requests.post(self.API_URL, headers=headers, json=data, timeout=90)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            if "choices" in result and result["choices"] and "message" in result["choices"][0]:
                content = result["choices"][0]["message"]["content"]
                # Attempt to extract citations if available (structure might vary)
                citations = result.get("citations", []) # Check top level
                if not citations and "sources" in result: # Check other common names
                     citations = result.get("sources", [])

                # Basic parsing attempt (can be improved based on observed API output)
                # Assuming the model follows the prompt to separate summary and sources
                summary = content # Default to full content if parsing fails
                sources_list = citations # Use structured citations if available

                # Simple text parsing if no structured citations and "Sources:" marker exists
                if not sources_list and "Sources:" in content:
                    try:
                        parts = content.split("Sources:", 1)
                        summary = parts[0].strip()
                        sources_text = parts[1].strip()
                        # Split sources by newline or common delimiters like '- '
                        sources_list = [s.strip().lstrip('- ') for s in sources_text.split('\n') if s.strip()]
                    except Exception:
                        # If splitting fails, revert to using the full content as summary
                        summary = content
                        sources_list = [] # No reliable sources found via text parsing

                # If still no sources, check if the content itself looks like a list of URLs
                if not sources_list and '\n' in summary and all(s.strip().startswith('http') for s in summary.split('\n') if s.strip()):
                     sources_list = [s.strip() for s in summary.split('\n') if s.strip()]
                     summary = "Summary could not be automatically separated. Please check raw response."


                return {
                    "summary": summary,
                    "sources": sources_list,
                    "raw_response": content # Include raw response for debugging
                }
            else:
                # Handle cases where the API response structure is unexpected
                error_msg = "Unexpected API response format."
                if "error" in result:
                    error_msg += f" API Error: {result['error'].get('message', 'Unknown error')}"
                return {"error": error_msg, "raw_response": result}

        except RequestException as e:
            error_message = f"API request failed: {str(e)}"
            if e.response is not None:
                try:
                    error_details = e.response.json()
                    error_message += f" - {error_details.get('error', {}).get('message', e.response.text)}"
                except json.JSONDecodeError:
                    error_message += f" - Status Code: {e.response.status_code}"
            return {"error": error_message}
        except json.JSONDecodeError:
            # This might happen if the response isn't valid JSON
            return {"error": "Failed to parse API response as JSON", "raw_response": response.text if 'response' in locals() else 'No response object'}
        except Exception as e:
            # Catch-all for other unexpected errors
            return {"error": f"An unexpected error occurred: {str(e)}"}


def process_csv_file(csv_file_path: str, column_name: str, assistant: ResearchAssistant,
                    orgname: Optional[str] = None, model: str = ResearchAssistant.DEFAULT_MODEL,
                    max_concurrent_calls: int = 5) -> None:
    """
    Process a CSV file by researching each row's specified column and appending results.

    Args:
        csv_file_path: Path to the CSV file
        column_name: Name of the column containing queries
        assistant: ResearchAssistant instance
        orgname: Optional organization name to include in queries
        model: Perplexity model to use
    """
    try:
        # Read the CSV file
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        if not rows:
            print("Error: CSV file is empty or has no data rows.", file=sys.stderr)
            return

        # Check if the specified column exists
        if column_name not in rows[0]:
            print(f"Error: Column '{column_name}' not found in CSV file. Available columns: {list(rows[0].keys())}", file=sys.stderr)
            return

        # Set up output CSV file
        output_file = csv_file_path.rsplit('.', 1)[0] + '_with_results_2.csv'
        fieldnames = list(rows[0].keys()) + ['research_results']

        # Initialize CSV file with header
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        # Process rows in parallel with immediate writing
        total_rows = len(rows)
        completed_rows = {}  # Store completed rows by index
        next_write_idx = 0  # Next row index to write
        write_lock = threading.Lock()  # Thread-safe writing

        def process_single_row(row_idx: int, row: Dict[str, Any]) -> None:
            """Process a single CSV row and write result immediately if possible."""
            nonlocal next_write_idx

            query_text = row[column_name].strip()
            if not query_text:
                print(f"Warning: Row {row_idx + 1} has empty {column_name}, skipping.", file=sys.stderr)
                row_copy = row.copy()
                row_copy['research_results'] = format_results_text({"error": "Empty query"})
                completed_rows[row_idx] = row_copy
            else:
                # Construct the research query
                research_query = query_text
                if orgname:
                    research_query = f"For the organization: {orgname}, answer this query only using reputed sources such as the company's website, annual reports, press releases, and other credible sources. Query: {query_text}"

                print(f"Processing row {row_idx + 1}/{total_rows}: {research_query}", file=sys.stderr)

                # Perform research
                results = assistant.research_topic(research_query, model=model)

                # Add results as formatted text to the row
                row_copy = row.copy()
                row_copy['research_results'] = format_results_text(results)
                completed_rows[row_idx] = row_copy

            # Check if we can write consecutive rows starting from next_write_idx
            with write_lock:
                while next_write_idx in completed_rows:
                    row_to_write = completed_rows[next_write_idx]
                    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(row_to_write)
                    print(f"✓ Row {next_write_idx + 1}/{total_rows} result saved to CSV", file=sys.stderr)
                    del completed_rows[next_write_idx]
                    next_write_idx += 1

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_concurrent_calls) as executor:
            # Submit all tasks
            futures = [executor.submit(process_single_row, i, row) for i, row in enumerate(rows)]

            # Wait for all tasks to complete
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions that occurred

        print(f"\n✅ CSV processing complete! All results saved to: {output_file}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file_path}", file=sys.stderr)
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}", file=sys.stderr)


def format_results_text(results: Dict[str, Any]) -> str:
    """
    Format research results as clean, readable text (for CSV storage).

    Args:
        results: The research results dictionary.

    Returns:
        A formatted text string of the results.
    """
    lines = []

    if "error" in results:
        lines.append(f"❌ Error: {results['error']}")
        if "raw_response" in results:
            lines.append("\n📄 Raw Response Snippet:")
            raw_response_str = json.dumps(results["raw_response"]) if isinstance(results["raw_response"], dict) else str(results["raw_response"])
            lines.append(raw_response_str[:500] + ("..." if len(raw_response_str) > 500 else ""))
        return "\n".join(lines)

    lines.append(results.get("summary", "No summary provided."))

    sources = results.get("sources")
    if sources:
        lines.append("\n SOURCES:")
        if isinstance(sources, list):
            for i, source in enumerate(sources, 1):
                 if isinstance(source, dict):
                     title = source.get('title', 'No Title')
                     url = source.get('url', '')
                     lines.append(f"  {i}. {title}{(' (' + url + ')') if url else ''}")
                 elif isinstance(source, str):
                     lines.append(f"  {i}. {source}")
                 else:
                     lines.append(f"  {i}. {str(source)}")
        else:
            lines.append(f"  {sources}")
    else:
        lines.append("\n🔗 SOURCES: No sources were explicitly listed or extracted.")
        if "raw_response" in results:
             lines.append("(Check raw response below for potential sources within the text)")
             lines.append("\n📄 Raw Response:")
             lines.append(results["raw_response"])

    return "\n".join(lines)


def display_results(results: Dict[str, Any], output_json: bool = False):
    """
    Display the research results in a human-readable format or as JSON.

    Args:
        results: The research results dictionary.
        output_json: If True, print results as JSON.
    """
    if output_json:
        print(json.dumps(results, indent=2))
        return

    # Use the text formatting function to display results
    print(format_results_text(results))


def main():
    """Main entry point for the research finder CLI."""
    parser = argparse.ArgumentParser(
        description="Research Finder CLI - Research topics using Perplexity Sonar API. Supports organization-specific queries and CSV batch processing."
    )

    parser.add_argument(
        "query",
        type=str,
        nargs='?',
        help="The research question or topic to investigate. Not required when using --csv-file."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=ResearchAssistant.DEFAULT_MODEL,
        help=f"Perplexity model to use (default: {ResearchAssistant.DEFAULT_MODEL})"
    )
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        help="Perplexity API key (if not provided, checks PPLX_API_KEY env var or key file)"
    )
    parser.add_argument(
        "-p",
        "--prompt-file",
        type=str,
        help=f"Path to file containing the system prompt (default: {ResearchAssistant.PROMPT_FILE} in script dir)"
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable format."
    )
    parser.add_argument(
        "--orgname",
        type=str,
        help="Organization name to include in the research query."
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        help="Path to CSV file to process. If provided, will process each row using the specified column."
    )
    parser.add_argument(
        "--column",
        type=str,
        help="Column name in CSV file containing the queries to research. Required when using --csv-file."
    )
    parser.add_argument(
        "--ncall",
        type=int,
        default=5,
        help="Number of parallel API calls to make when processing CSV (default: 5)."
    )

    args = parser.parse_args()

    # Validate arguments
    if args.csv_file and not args.column:
        print("Error: --column is required when using --csv-file", file=sys.stderr)
        sys.exit(1)
    if args.column and not args.csv_file:
        print("Error: --csv-file is required when using --column", file=sys.stderr)
        sys.exit(1)
    if not args.csv_file and not args.query:
        print("Error: Either provide a query as a positional argument or use --csv-file with --column", file=sys.stderr)
        sys.exit(1)

    try:
        print("Initializing research assistant...", file=sys.stderr)
        assistant = ResearchAssistant(api_key=args.api_key, prompt_file=args.prompt_file)

        if args.csv_file:
            # CSV processing mode
            print(f"Processing CSV file: {args.csv_file} using column: {args.column} with {args.ncall} parallel calls", file=sys.stderr)
            process_csv_file(args.csv_file, args.column, assistant, args.orgname, args.model, args.ncall)
        else:
            # Single query mode
            research_query = args.query
            if args.orgname:
                research_query = f"For the organization: {args.orgname}, answer this query only using reputed sources such as the company's website, annual reports, press releases, and other credible sources. Query: {args.query}"

            print(f"Researching query: \"{research_query}\"", file=sys.stderr)
            print("Researching in progress...", file=sys.stderr)
            results = assistant.research_topic(research_query, model=args.model)
            display_results(results, output_json=args.json)

    except ValueError as e: # Catch API key error specifically
        print(f"Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

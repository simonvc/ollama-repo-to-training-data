#!/usr/bin/env python3
"""
Ollama Repo to Training Data (ORTD)
Converts code repositories into training data files using Ollama LLM
"""

import os
import sys
import ast
import re
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import deque
import requests
import json
import time
from rich.console import Console
from rich.text import Text
from rich.live import Live
from rich.table import Table

console = Console()


class OllamaClient:
    """Client for interacting with local Ollama instance"""
    
    def __init__(self, model="devstral", base_url=None):
        self.model = model
        if base_url is None:
            # Try to detect WSL2 and use Windows host IP
            try:
                import subprocess
                result = subprocess.run(['cat', '/proc/version'], capture_output=True, text=True)
                if 'microsoft' in result.stdout.lower() or 'wsl' in result.stdout.lower():
                    # We're in WSL2, get the Windows host IP
                    result = subprocess.run(['cat', '/etc/resolv.conf'], capture_output=True, text=True)
                    for line in result.stdout.split('\n'):
                        if line.startswith('nameserver'):
                            host_ip = line.split()[1]
                            base_url = f"http://{host_ip}:11434"
                            print(f"Detected WSL2, using Windows host IP: {host_ip}")
                            break
                if base_url is None:
                    base_url = "http://localhost:11434"
            except:
                base_url = "http://localhost:11434"
        self.base_url = base_url
    
    def test_connection(self) -> bool:
        """Test connection to Ollama and check if model exists"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [model.get("name", "").split(":")[0] for model in models]
            
            print(f"Ollama is running at {self.base_url}")
            print(f"Available models: {', '.join(model_names)}")
            
            if self.model in model_names:
                print(f"✓ Model '{self.model}' is available")
                return True
            else:
                print(f"✗ Model '{self.model}' not found")
                print(f"Please install it with: ollama pull {self.model}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to Ollama at {self.base_url}")
            print("Please make sure Ollama is running")
            return False
        except Exception as e:
            print(f"✗ Error testing Ollama connection: {e}")
            return False
    
    def test_generate(self) -> bool:
        """Test text generation with the model"""
        if not self.test_connection():
            return False
            
        try:
            print(f"\nTesting text generation with {self.model}...")
            response = self.generate("Say hello in one word.")
            if response and response.strip():
                print(f"✓ Generation test successful: '{response.strip()}'")
                return True
            else:
                print("✗ Generation test failed: empty response")
                return False
        except Exception as e:
            print(f"✗ Generation test failed: {e}")
            return False
    
    def generate(self, prompt: str) -> str:
        """Generate response from Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return f"Error generating response: {e}"


class CodeAnalyzer:
    """Analyzes code files and extracts functions"""
    
    @staticmethod
    def extract_python_functions(content: str) -> List[Tuple[str, str, int]]:
        """Extract function definitions from Python code"""
        functions = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_start = node.lineno
                    func_end = node.end_lineno if hasattr(node, 'end_lineno') else func_start + 10
                    func_lines = content.split('\n')[func_start-1:func_end]
                    func_code = '\n'.join(func_lines)
                    functions.append((node.name, func_code, func_start))
        except SyntaxError:
            pass
        return functions
    
    @staticmethod
    def extract_go_functions(content: str) -> List[Tuple[str, str, int]]:
        """Extract function definitions from Go code"""
        functions = []
        lines = content.split('\n')
        
        func_pattern = re.compile(r'^func\s+(\w+|\([^)]*\)\s*\w+)\s*\([^)]*\)')
        current_func = None
        func_lines = []
        brace_count = 0
        line_num = 0
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            if func_pattern.match(line.strip()):
                if current_func:
                    # Save previous function
                    func_code = '\n'.join(func_lines)
                    functions.append((current_func, func_code, line_num - len(func_lines)))
                
                # Start new function
                match = func_pattern.match(line.strip())
                current_func = match.group(1) if match else "unknown"
                func_lines = [line]
                brace_count = line.count('{') - line.count('}')
            elif current_func:
                func_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count <= 0 and '{' in ''.join(func_lines):
                    # Function ended
                    func_code = '\n'.join(func_lines)
                    functions.append((current_func, func_code, line_num - len(func_lines) + 1))
                    current_func = None
                    func_lines = []
                    brace_count = 0
        
        # Handle last function if file ends
        if current_func and func_lines:
            func_code = '\n'.join(func_lines)
            functions.append((current_func, func_code, line_num - len(func_lines) + 1))
        
        return functions


class ToolSystem:
    """Implements tools for the LLM to use when analyzing code"""
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
    
    def list_files(self, directory: str = ".") -> str:
        """List files in a directory"""
        try:
            path = self.repo_path / directory
            files = [f.name for f in path.iterdir() if f.is_file()]
            return "\n".join(files)
        except Exception as e:
            return f"Error listing files: {e}"
    
    def list_files_recursively(self, directory: str = ".") -> str:
        """List files recursively in a directory"""
        try:
            path = self.repo_path / directory
            files = []
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(self.repo_path)
                    files.append(str(rel_path))
            return "\n".join(files)
        except Exception as e:
            return f"Error listing files recursively: {e}"
    
    def find_in_files(self, pattern: str, target: str = ".") -> str:
        """Search for pattern in files"""
        try:
            path = self.repo_path / target
            results = []
            
            if path.is_file():
                # Search in single file
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for i, line in enumerate(content.split('\n'), 1):
                            if pattern in line:
                                results.append(f"{path}:{i}: {line.strip()}")
                except Exception:
                    pass
            else:
                # Search in directory
                for file_path in path.rglob("*"):
                    if file_path.is_file() and file_path.suffix in ['.py', '.go']:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                for i, line in enumerate(content.split('\n'), 1):
                                    if pattern in line:
                                        rel_path = file_path.relative_to(self.repo_path)
                                        results.append(f"{rel_path}:{i}: {line.strip()}")
                        except Exception:
                            continue
            
            return "\n".join(results) if results else f"Pattern '{pattern}' not found"
        except Exception as e:
            return f"Error searching: {e}"


class TrainingDataGenerator:
    """Generates training data files from code repositories"""
    
    def __init__(self, repo_path: str, preamble: str):
        self.repo_path = Path(repo_path)
        self.preamble = preamble
        self.ollama = OllamaClient()
        self.tools = ToolSystem(repo_path)
        self.question_queue = deque()
        self.analyzer = CodeAnalyzer()

        self.start_time = time.time()
        self.files_processed_count = 0
        self.training_files_created_count = 0
        
        # Ensure output directory exists
        os.makedirs("training_data", exist_ok=True)

    def _generate_status_table(self) -> Table:
        """Generates a Rich Table object for status display."""
        elapsed_seconds = time.time() - self.start_time
        
        # Format elapsed time
        hours, rem = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Calculate files per hour
        if elapsed_seconds < 1:
            files_per_hour_str = "Calculating..."
        else:
            files_per_hour = (self.files_processed_count / elapsed_seconds) * 3600
            files_per_hour_str = f"{files_per_hour:.2f} files/hr"
            
        table = Table(show_header=False, show_edge=False, pad_edge=False)
        table.add_column("Metric", style="dim", width=7)
        table.add_column("Value")

        table.add_row("⏱️ Time", elapsed_time_str)
        table.add_row("📂 Files", str(self.files_processed_count))
        table.add_row("🚀 Speed", files_per_hour_str)
        table.add_row("💾 Trained", str(self.training_files_created_count))
        
        return table
    
    def generate_random_filename(self) -> str:
        """Generate random training data filename"""
        return f"training_data/{random.randint(100000000000, 999999999999)}.td"
    
    def save_training_data(self, content: str) -> str:
        """Save training data to file"""
        filename = self.generate_random_filename()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        console.print(f"Created training data: [green]{filename}[/green]")
        self.training_files_created_count += 1
        return filename
    
    def generate_hypothetical_questions(self, file_path: str, content: str) -> List[str]:
        """Generate hypothetical questions about a code file"""
        prompt = f"""
Given this code file from {file_path}:

```
{content[:2000]}...
```

Generate 5 specific, technical questions about this code that would help someone understand:
- How it works
- What dependencies it uses
- How it relates to other parts of the system
- Business logic it implements
- Design patterns used

Format each question on a single line starting with "Q: "
"""
        
        response = self.ollama.generate(prompt)
        questions = []
        
        for line in response.split('\n'):
            if line.strip().startswith('Q: '):
                questions.append(line.strip()[3:])
        
        # Fallback questions if LLM doesn't respond properly
        if not questions:
            questions = [
                f"What is the main purpose of {file_path}?",
                f"What dependencies does {file_path} use?",
                f"How does {file_path} handle errors?",
                f"What design patterns are used in {file_path}?",
                f"How does {file_path} integrate with other system components?"
            ]
        
        return questions[:5]
    
    def answer_question_with_tools(self, question: str, file_path: str, content: str) -> str:
        """Answer a question about code using available tools"""
        tools_prompt = f"""
You are analyzing a codebase. You have access to these tools:
- list_files(directory): List files in directory
- list_files_recursively(directory): List all files recursively  
- find_in_files(pattern, target): Search for pattern in files

Current file: {file_path}
Question: {question}

Code content:
```
{content}
```

Provide a detailed answer with code snippets and explanations. Include business logic assumptions and implementation details.
"""
        
        return self.ollama.generate(tools_prompt)
    
    def process_file(self, file_path: Path):
        """Process a single source file"""
        rel_path = file_path.relative_to(self.repo_path)
        console.print(f"Processing: [bold blue]{rel_path}[/bold blue]")
        self.files_processed_count += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return
        
        # Generate training data for whole file
        whole_file_content = f"""** {self.preamble} **

Full source code of {rel_path}:

```
{content}
```

This file contains the complete implementation with all functions, imports, and logic.
"""
        
        filename = self.save_training_data(whole_file_content)
        
        # Extract functions
        if file_path.suffix == '.py':
            functions = self.analyzer.extract_python_functions(content)
        elif file_path.suffix == '.go':
            functions = self.analyzer.extract_go_functions(content)
        else:
            functions = []
        
        # Generate training data for each function
        for func_name, func_code, line_num in functions:
            func_content = f"""** {self.preamble} **

Function {func_name} from {rel_path} (line {line_num}):

```
{func_code}
```

This function is part of the larger codebase and implements specific business logic.
"""
            
            filename = self.save_training_data(func_content)
        
        # Generate hypothetical questions and queue them
        questions = self.generate_hypothetical_questions(str(rel_path), content)
        for question in questions:
            self.question_queue.append((question, str(rel_path), content))
        
        print(f"Queued {len(questions)} questions for {rel_path}")
    
    def process_questions(self, live: Live):
        """Process all queued questions (breadth-first)"""
        console.print(f"\n[bold]Processing {len(self.question_queue)} queued questions...[/bold]")
        
        processed_questions_count = 0
        while self.question_queue:
            question, file_path, content = self.question_queue.popleft()
            console.print(f"Answering Q{processed_questions_count+1}: {question[:60]}...")
            
            answer = self.answer_question_with_tools(question, file_path, content)
            
            qa_content = f"""** {self.preamble} **

Question about {file_path}: {question}

{answer}

This answer includes code analysis and business logic explanations based on the codebase.
"""
            
            filename = self.save_training_data(qa_content)
            processed_questions_count += 1
            live.update(self._generate_status_table())
    
    def run(self):
        """Main processing loop"""
        if not self.repo_path.exists():
            console.print(f"[bold red]Error: Repository path {self.repo_path} does not exist[/bold red]")
            return
        
        console.print(f"[bold]Processing repository:[/bold] {self.repo_path}")
        console.print(f"[bold]Preamble:[/bold] {self.preamble}")
        
        # Find all source files
        source_files = []
        for ext in ['.py', '.go']:
            source_files.extend(self.repo_path.rglob(f"*{ext}"))
        
        console.print(f"Found {len(source_files)} source files to process.")
        
        with Live(self._generate_status_table(), refresh_per_second=4, console=console) as live:
            # Process each file
            for file_path in source_files:
                self.process_file(file_path)
                live.update(self._generate_status_table())
            
            # Process all questions
            self.process_questions(live)
        
        # Explicitly print final statistics after Live context
        final_stats = self._generate_status_table()
        console.print("\n[bold]Final Statistics:[/bold]")
        console.print(final_stats)
        console.print(f"\n[bold green]Processing Complete![/bold green] Generated {self.training_files_created_count} training data files in training_data/ directory.")


def test_ollama():
    """Test Ollama connection and model availability"""
    client = OllamaClient()
    success = client.test_generate()
    return success


def main():
    parser = argparse.ArgumentParser(description='Generate training data from code repositories')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test Ollama connection and model')
    
    # Generate command  
    gen_parser = subparsers.add_parser('generate', help='Generate training data')
    gen_parser.add_argument('repo_path', help='Path to the git repository')
    gen_parser.add_argument('preamble', help='Preamble text for each training sample')
    
    # For backward compatibility, if no subcommand is used, assume generate
    args, unknown = parser.parse_known_args()
    
    if args.command == 'test':
        success = test_ollama()
        sys.exit(0 if success else 1)
    elif args.command == 'generate':
        generator = TrainingDataGenerator(args.repo_path, args.preamble)
        generator.run()
    else:
        # Backward compatibility: assume old format
        if len(sys.argv) >= 3:
            repo_path = sys.argv[1]
            preamble = sys.argv[2]
            generator = TrainingDataGenerator(repo_path, preamble)
            generator.run()
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
import os
import yaml
from pathlib import Path
import subprocess
import json 
from tqdm import tqdm
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

RETRY_PROMPT = """<|im_start|>user
We could not parse your response through json.loads() method. Please double check the format and content of your response. Did you include escape characters that jons.loads() can't parse? Or did you respond with more than just the JSON object? 

Remember, You should format your response into a JSON object with the following fields:
```json
{{
    "function_name": str,
    "function_signature": str,
    "function_docstring":str,
    "solution": str,
    "problem_statement": str
}}
```
Please respond with ONLY the JSON object and nothing else.
<|im_end|>
"""

def verify_folder_structure(folder_path):
    test_path = os.path.join(folder_path, "tests/test.cpp")
    solution_path = os.path.join(folder_path, "solution/solution.cpp")
    run_path = os.path.join(folder_path, "run.bash")
    if not all([os.path.exists(test_path), os.path.exists(solution_path), os.path.exists(run_path)]):
        return False
    try:
        if os.path.getsize(test_path) == 0 or \
            os.path.getsize(solution_path) == 0 or \
            os.path.getsize(run_path) == 0:
            return False
    except OSError:
        return False
    return True

def create_table_visualization(df, output_file='dataset_statistics.pdf'):
    """Create a PDF visualization of the skills statistics table."""
    plt.figure(figsize=(12, 8))
    
    # Create a mask for the lower triangle
    mask = np.tril(np.ones_like(df), k=-1)
    
    # Create heatmap
    sns.heatmap(df, 
                annot=True,  # Show numbers in cells
                fmt='g',     # Format as general number
                cmap='YlOrRd',  # Color scheme
                mask=mask,   # Mask lower triangle
                cbar_kws={'label': 'Count'},
                square=True)
    
    plt.title('Skills Statistics Heatmap')
    plt.tight_layout()
    
    # Save to PDF
    output_path = os.path.join(output_file)
    print(colored(f'Saving visualization to {output_path}', 'green'))
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_file

def update_cost(new_cost):
    try:
        with open("cost.txt", "r") as f:
            current_cost = float(f.read())
    except:
        current_cost = 0
    new_cost = current_cost + new_cost
    with open("cost.txt", "w") as f:
        f.write(str(new_cost))
    return

@dataclass
class TestCase:
    name: str
    status: str
    time: float
    classname: str
    failure_message: Optional[str] = None
    failure_type: Optional[str] = None

@dataclass
class TestSuite:
    name: str
    tests: int
    failures: int
    disabled: int
    errors: int
    time: float
    timestamp: datetime
    test_cases: List[TestCase]

def parse_gtest_xml(xml_path: str) -> List[TestSuite]:
    """
    Parse a GTest XML report and return structured test results.
    
    Args:
        xml_path (str): Path to the GTest XML report
        
    Returns:
        List[TestSuite]: List of test suites with their test cases and results
        
    Example:
        results = parse_gtest_xml("test_results.xml")
        for suite in results:
            print(f"Suite: {suite.name} - {suite.failures} failures")
            for case in suite.test_cases:
                print(f"  {case.name}: {case.status}")
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        test_suites = []
        
        # Parse each test suite
        for testsuite in root.findall('.//testsuite'):
            test_cases = []
            
            # Parse individual test cases
            for testcase in testsuite.findall('testcase'):
                # Check for failures
                failure = testcase.find('failure')
                failure_message = None
                failure_type = None
                status = 'PASSED'
                
                if failure is not None:
                    failure_message = failure.text
                    failure_type = failure.get('type')
                    status = 'FAILED'
                
                # Create TestCase object
                test_case = TestCase(
                    name=testcase.get('name'),
                    status=status,
                    time=float(testcase.get('time')),
                    classname=testcase.get('classname'),
                    failure_message=failure_message,
                    failure_type=failure_type
                )
                test_cases.append(test_case)
            
            # Create TestSuite object
            suite = TestSuite(
                name=testsuite.get('name'),
                tests=int(testsuite.get('tests')),
                failures=int(testsuite.get('failures')),
                disabled=int(testsuite.get('disabled')),
                errors=int(testsuite.get('errors')),
                time=float(testsuite.get('time')),
                timestamp=datetime.fromisoformat(testsuite.get('timestamp')),
                test_cases=test_cases
            )
            test_suites.append(suite)
            
        return test_suites
        
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML format: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing GTest XML: {e}")

def generate_summary_report(test_suites: List[TestSuite]) -> str:
    """
    Generate a human-readable summary report from parsed test results.
    
    Args:
        test_suites (List[TestSuite]): List of parsed test suites
        
    Returns:
        str: Formatted summary report
    """
    summary = []
    total_tests = 0
    total_failures = 0
    total_disabled = 0
    total_time = 0
    
    summary.append("GTest Results Summary")
    summary.append("===================")
    
    for suite in test_suites:
        total_tests += suite.tests
        total_failures += suite.failures
        total_disabled += suite.disabled
        total_time += suite.time
        
        summary.append(f"\nTest Suite: {suite.name}")
        summary.append(f"Tests: {suite.tests}, Failures: {suite.failures}, "
                      f"Disabled: {suite.disabled}, Time: {suite.time:.3f}s")
        
        # List failed tests
        if suite.failures > 0:
            summary.append("\nFailed Tests:")
            for case in suite.test_cases:
                if case.status == 'FAILED':
                    summary.append(f"  - {case.classname}.{case.name}")
                    if case.failure_message:
                        summary.append(f"    Error: {case.failure_message.strip()}")

    summary = {}
    summary["total_tests"] = total_tests
    summary["total_failures"] = total_failures
    summary["total_disabled"] = total_disabled
    summary["total_time"] = total_time

    
    return summary

class EvaluationParser():
    """
    Parser to postprocess and evaluate solutions to skill-mix dataset.
    """
    def __init__(self, config):
        if isinstance(config, str):
            # load yaml
            with open(config, 'r') as file:
                self.config = yaml.load(file, Loader=yaml.FullLoader)
        else:
            self.config = config 
        self.base_folder = os.path.join(
            self.config['absolute_path_base'], 
            self.config['output_folder']
        )
        # assert exist 
        self.questions_path = os.path.join(self.base_folder, "questions.json")
        assert os.path.exists(self.questions_path), f"Path does not exist: {self.questions_path}"


        if not os.path.exists(self.questions_path):
            print(colored(f"Questions file not found at {self.questions_path}. This may cause errors.", "red"))

        self.solution_path = "solution/solution.cpp"
        self.tests_path = "tests/test.cpp"
        self.run_path = "run.bash"
        self.xlm_path = "gtest_xla_report.xml"

        self.TIMEOUT = 4 # seconds
    
    def _convert_skill_to_list(self, skills:str):
        """
        Consider either k=1 or k=2 case.
        """
        return [s.strip() for s in skills.split(", ")]
    
    def _verify_folder_structure(self, folder_path):
        test_path = os.path.join(folder_path, self.tests_path)
        solution_path = os.path.join(folder_path, self.solution_path)
        run_path = os.path.join(folder_path, self.run_path)
        if not all([os.path.exists(test_path), os.path.exists(solution_path), os.path.exists(run_path)]):
            return False
        try:
            if os.path.getsize(test_path) == 0 or \
                os.path.getsize(solution_path) == 0 or \
                os.path.getsize(run_path) == 0:
                return False
        except OSError:
            return False
        return True
    
    def evaluate_solution(self, question_id, solution):
        """
        Evaluate a model solution to questions generated with the config given to constructor.
            question_id: str, the question id
            solution: str, the solution code
        """
        folder_path = os.path.join(self.base_folder, question_id)
        xml_path = os.path.join(folder_path, self.xlm_path)
        if os.path.exists(xml_path):
            os.remove(xml_path)
        os.environ["GTEST_OUTPUT"] = f"xml:{xml_path}"
        run_bash_path = os.path.join(folder_path, self.run_path)
        test_runner_path = os.path.join(folder_path, 'test_runner')
        if not os.path.exists(run_bash_path):
            raise FileNotFoundError(f"run.bash not found in {folder_path}")
        if os.path.exists(test_runner_path):
            os.remove(test_runner_path)
        solution_path = os.path.join(folder_path, self.solution_path)
        temp_path = os.path.join(folder_path, 'solution', 'temp.cpp')
        with open(solution_path, 'r') as file:
            old_solution = file.read()
        with open(temp_path, 'w') as file:
            file.write(old_solution)
        with open(solution_path, 'w') as file:
            file.write(solution)
        try:
            result = subprocess.run(
                [run_bash_path],
                cwd=folder_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.TIMEOUT,
            )
        except subprocess.TimeoutExpired as e:
            with open(solution_path, 'w') as file:
                file.write(old_solution)
            return {
                "status": 0,
                "message": "TIMEOUT"
            }
        except subprocess.CalledProcessError as e:
            with open(solution_path, 'w') as file:
                file.write(old_solution)
            print(e)
            return {
                "status": 0,
                "message": e
            }
        with open(solution_path, 'w') as file:
            file.write(old_solution)
        
        if not os.path.exists(xml_path):
            return {
                "status": 0,
                "message": "XML_ERROR"
            }
        test_results = parse_gtest_xml(xml_path)
        summary = generate_summary_report(test_results)
        return {
            "status": 1,
            "message": summary
        }
    
    def score_all_solutions(self, solution_path):
        """
        solution_path should be file generated from generate_solutions.py for parsing compatibility. 
        """
        with open(solution_path, 'r') as file:
            solutions = json.load(file)

        for qid in tqdm(solutions):
            scored_solutions = []
            responses = solutions[qid]["responses"]
            for solution in responses:
                result = self.evaluate_solution(qid, solution["code"])
                if not result["status"]:
                    scored_solutions.append({
                        "solution": solution,
                        "score": 0,
                        "message": result["message"]
                    })
                else:
                    score = (result["message"]["total_tests"] - result["message"]["total_failures"] ) / result["message"]["total_tests"]
                    
                    scored_solutions.append({
                        "solution": solution,
                        "score": score,
                        "message": result["message"]
                    })
            solutions["responses"] = scored_solutions

        with open(solution_path, 'w') as f:
            json.dump(solutions, f, indent=4)
        
        print(colored(f"Scoring complete. Results saved to {solution_path}", "green"))
        return 
    
    def postprocess(self, 
                    remove_unverified=False, 
                    compute_generation_statistics = False):
        print("Postprocessing the generated questions and unit tests. This includes \n- Removing unverified questions' folders to save space\n- Adding unit test code to `questions.json` file\n- Adding verified gold solutions to `questions.json`\n- (Optional) Computing generation statistics\n- (Optional) Setting up `gradio.json` file which contains only CORE information.")

        questions  = json.load(open(self.questions_path, "r"))
        count = 0
        mismatch = 0
        if compute_generation_statistics:
            skill_count = {}
            unique_skills = set()
        else:
            skill_count = None
        for qid in questions.copy():
            if not questions[qid]["verification_status"]:
                count += 1
                if remove_unverified:
                    folder_path = os.path.join(self.base_folder, qid)
                    if os.path.exists(folder_path):
                        os.rmdir(folder_path)
                    del questions[qid]

            else:
                sample_solution = questions[qid]["sample_solution"]
                # fetch gold solution from solution path
                folder_path = os.path.join(self.base_folder, qid)
                solution_path = os.path.join(folder_path, self.solution_path)
                with open(solution_path, 'r') as file:
                    gold_solution = file.read()
                questions[qid]["gold_solution"] = gold_solution
                if sample_solution != gold_solution:
                    mismatch += 1
                skills = questions[qid]["skills"]
                skill_list = self._convert_skill_to_list(skills)
                for s in skill_list:
                    if s not in unique_skills:
                        unique_skills.add(s)
                if skills in skill_count:
                    skill_count[skills] += 1
                else:
                    skill_count[skills] = 1


        print(colored(f"Removed {count} unverified questions.", "green"))
        
        for qid in questions:
            folder_path = os.path.join(self.base_folder, qid)
            if not self._verify_folder_structure(folder_path):
                print(colored(f"Question {qid} folder structure is incorrect. Skipping...", "red"))
                continue
            with open(os.path.join(folder_path, self.tests_path), 'r') as file:
                questions[qid]["unit_tests"] = file.read()
        
        with open(self.questions_path, 'w') as file:
            json.dump(questions, file, indent=4)
        print(colored(f"Added unit test and gold solution to `questions.json` file.", "green"))

        if compute_generation_statistics:
            failed_verification = count
            total_questions = len(questions)
            print(f"Total questions: {total_questions}")
            print(f"Failed verification: {failed_verification} / {total_questions}")
            print(f"Failed first attempt by Claude: {mismatch} / {total_questions} ({mismatch/total_questions*100})")
            unique_skills = list(unique_skills)
            n = len(unique_skills)
            table = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    skill_a = unique_skills[i]
                    skill_b = unique_skills[j]
                    if f"{skill_a}, {skill_b}" in skill_count:
                        table[i][j] = skill_count[f"{skill_a}, {skill_b}"]
                    elif f"{skill_b}, {skill_a}" in skill_count:
                        table[i][j] = skill_count[f"{skill_b}, {skill_a}"]
                    elif f"{skill_a}" in skill_count:
                        table[i][j] = skill_count[f"{skill_a}"]
            df = pd.DataFrame(table, index=unique_skills, columns=unique_skills)
            create_table_visualization(df, output_file = self.base_folder+'/dataset_statistics.pdf')

    def setup_gradio(self):
        questions = json.load(open(self.questions_path, "r"))
        gradio_file = {}
        new_id = 1
        for qid in questions:
            q = questions[qid]
            gradio_file[new_id] = {
                "skills": q["skills"],
                "problem_statement": q["problem_statement"],
                "function_name": q["function_name"],
                "function_signature": q["function_signature"],
                "function_docstring": q["function_docstring"],
                "gold_solution": q["gold_solution"],
                "unit_tests": q["unit_tests"]
            }
            new_id += 1
        with open('gradio.json', 'w') as f:
            json.dump(gradio_file, f, indent=4)
    
        print(colored(f"Gradio file setup complete. Saved to gradio.json", "green"))
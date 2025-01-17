import json
import gradio as gr

# 1. Load the data from gradio.json
with open("gradio.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert the keys to integers (if they are numeric as strings like "1", "2", etc.)
# so we can more easily handle them in a slider.
question_ids = sorted([int(k) for k in data.keys()])

def render_question(question_id: int):
    """Given a question ID, returns the strings needed to populate our Gradio UI."""
    # Convert question_id to string to retrieve from data.
    q = data[str(question_id)]

    # Extract the relevant fields.
    skills = q["skills"]
    problem_statement = q["problem_statement"]
    function_name = q["function_name"]
    function_signature = q["function_signature"]
    function_docstring = q["function_docstring"]
    gold_solution = q["gold_solution"]
    unit_tests = q["unit_tests"]

    skills = " ".join(skills.split("_"))

    # We'll format each piece using Markdown-friendly text.
    # Use .replace("\\n", "\n") to correctly interpret escaped \n in the JSON strings.
    skills_md = f"## Question\n- **Function Name**: {function_name}\n\n- **Skills**: {skills}"

    # Problem statement (and docstring) might contain literal "\n" characters in the JSON. 
    # Convert them to actual newlines so Markdown displays them properly.
    problem_template = """## Problem Statement
{problem_statement}

**Signature**:
```cpp
{function_signature}
```

**Docstring**:
```cpp
{function_docstring}
```"""
    problem_md = problem_template.format(
        problem_statement=problem_statement.replace("\\n", "\n"),
        function_name=function_name,
        function_signature=function_signature,
        function_docstring=function_docstring.replace("\\n", "\n")
    )
    gold_solution_md = f"## Gold Solution\n```cpp\n{gold_solution}\n```"
    unit_tests_md = f"## Unit Tests\n```cpp\n{unit_tests}\n```"

    return skills_md, problem_md, gold_solution_md, unit_tests_md

# 2. Set up the Gradio interface
def update_display(question_number):
    """This function is called whenever the slider changes; it updates the displayed content."""
    return render_question(question_number)

with gr.Blocks() as demo:
    gr.Markdown("## Code Skill-Mix Viewer")

    # Slider for selecting question index
    with gr.Row():
        question_slider = gr.Slider(
            minimum=min(question_ids),
            maximum=max(question_ids),
            step=1,
            value=min(question_ids),
            label="Select Question"
        )

    # Top panels (left: question, right: gold solution)
    with gr.Row():
        with gr.Column():
            skills_box = gr.Markdown()
            problem_box = gr.Markdown()
        with gr.Column():
            gold_solution_box = gr.Markdown()

    # Bottom panel (unit tests)
    with gr.Row():
        unit_test_box = gr.Markdown()

    # Whenever the slider changes, update all displays
    question_slider.change(
        fn=update_display,
        inputs=question_slider,
        outputs=[skills_box, problem_box, gold_solution_box, unit_test_box]
    )

    # Initialize the interface with the first question.
    # We'll manually set them for the default value so that something appears on load.
    default_skills, default_problem, default_solution, default_tests = render_question(question_ids[0])
    skills_box.value = default_skills
    problem_box.value = default_problem
    gold_solution_box.value = default_solution
    unit_test_box.value = default_tests

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
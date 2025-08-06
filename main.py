import gradio as gr
import openai
from dotenv import load_dotenv
import os
import json
import csv
import tempfile
from datetime import datetime
import pandas as pd
import altair as alt

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

model_options = ["GPT-3.5", "GPT-4", "Claude", "Gemini", "Mistral"]
judge_models = ["GPT-4", "Claude", "Gemini", "Mistral"]
model_costs = {"GPT-3.5": 0.0015, "GPT-4": 0.09}

evaluation_focus_choices = [
    "Overall Quality", "Conciseness", "Creativity",
    "Technical Accuracy", "Factual Correctness", "Tone & Clarity"
]

session_data = {
    "prompt": "", "selected_models": [], "model_outputs": {},
    "judge_model": "", "eval_focus": "", "judge_feedback": "",
    "cost_summary": ""
}

def get_gpt_response(prompt, model_name):
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        content = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        return content, total_tokens
    except Exception as e:
        return f"⚠️ Error: {str(e)}", 0
def compare_selected_models(prompt, selected_models):
    session_data["prompt"] = prompt
    session_data["selected_models"] = selected_models

    responses = []
    cost_summary = ""

    for model in model_options:
        if model not in selected_models:
            responses.append("")
            continue

        if model == "GPT-3.5":
            output, tokens = get_gpt_response(prompt, "gpt-3.5-turbo")
        elif model == "GPT-4":
            output, tokens = get_gpt_response(prompt, "gpt-4")
        else:
            output = f"🤖 {model} is not yet connected. Placeholder response."
            tokens = 0

        session_data["model_outputs"][model] = output

        if model in model_costs:
            cost = (tokens / 1000) * model_costs[model]
            cost_summary += f"💵 {model}: ${cost:.4f} ({tokens} tokens)\n"

        responses.append(output)

    session_data["cost_summary"] = cost_summary
    return responses + [cost_summary]

def judge_models_fn(prompt, judge_model, eval_focus, *model_outputs):
    comparison_text = f"The user asked:\n\"{prompt}\"\n\n"
    scores = {}
    hallucinations = []
    winner = ""

    for model, output in zip(model_options, model_outputs):
        if output.strip() and model in session_data["selected_models"]:
            comparison_text += f"### {model} Response:\n{output.strip()}\n\n"

    comparison_text += (
        f"Please evaluate the above responses focusing on: **{eval_focus}**.\n"
        "As an impartial judge, do the following:\n"
        "1. Give each model a score out of 10 with a short reason.\n"
        "2. Mention any hallucinations (false/made-up info).\n"
        "3. Declare the model that gave the best response.\n\n"
        "Return your answer in Markdown format:\n"
        "- **Model Name:** score/10 - short reason\n"
        "- ⚠️ Hallucinations\n"
        "- 🏆 Best Response: <Model Name>"
    )

    session_data["judge_model"] = judge_model
    session_data["eval_focus"] = eval_focus

    if judge_model == "GPT-4":
        result, _ = get_gpt_response(comparison_text, "gpt-4")
    elif judge_model == "Claude":
        result = "🧠 Placeholder: Claude would evaluate the models here."
    elif judge_model == "Gemini":
        result = "🔮 Placeholder: Gemini would provide the evaluation here."
    elif judge_model == "Mistral":
        result = "⚙️ Placeholder: Mistral evaluation coming soon."
    else:
        result = "❓ Invalid judge model."

    session_data["judge_feedback"] = result

    try:
        for line in result.splitlines():
            if line.strip().startswith("- **"):
                parts = line.split("**")
                model_name = parts[1].strip()
                score = int(parts[2].split("/")[0].replace(":", "").strip())
                scores[model_name] = score
            elif "Hallucinations" in line:
                for m in model_options:
                    if m in line:
                        hallucinations.append(m)
            elif "🏆" in line and "Best Response" in line:
                for m in model_options:
                    if m in line:
                        winner = m
                        break
    except:
        pass

    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "models_tested": session_data["selected_models"],
        "judge_model": judge_model,
        "evaluation_focus": eval_focus,
        "scores": scores,
        "hallucinations": hallucinations,
        "winner": winner,
        "costs": {
            m: round(model_costs[m] * 1000, 4)
            for m in session_data["selected_models"]
            if m in model_costs
        }
    }

    os.makedirs("logs", exist_ok=True)
    log_file = "logs/evaluations.json"
    try:
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(log)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print("Logging failed:", str(e))

    return result
def export_csv():
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Response"])
            for model, output in session_data["model_outputs"].items():
                writer.writerow([model, output])
            writer.writerow([])
            writer.writerow(["Judge Feedback"])
            writer.writerow([session_data["judge_feedback"]])
            return f.name
    except Exception as e:
        print("CSV export failed:", e)
        return None

def reset_all():
    for k in session_data:
        if isinstance(session_data[k], str):
            session_data[k] = ""
        elif isinstance(session_data[k], list) or isinstance(session_data[k], dict):
            session_data[k] = {} if isinstance(session_data[k], dict) else []
    return [""] * (len(model_options) + 3)

def load_analytics():
    if not os.path.exists("logs/evaluations.json"):
        return "No data yet.", None, None, None

    with open("logs/evaluations.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    win_counts = df["winner"].value_counts()
    most_wins = win_counts.idxmax()
    most_wins_count = win_counts.max()

    avg_scores = {}
    for row in df["scores"]:
        for k, v in row.items():
            avg_scores[k] = avg_scores.get(k, []) + [v]
    avg_scores = {k: sum(v) / len(v) for k, v in avg_scores.items()}
    highest_avg_model = max(avg_scores, key=avg_scores.get)

    hallucinated = {}
    for hs in df["hallucinations"]:
        for h in hs:
            hallucinated[h] = hallucinated.get(h, 0) + 1
    hallucination_prone = max(hallucinated, key=hallucinated.get) if hallucinated else "None"

    summary = (
        f"📈 **{len(df)} evaluations analyzed**\n\n"
        f"🏆 **Top Winner:** {most_wins} ({most_wins_count} wins)\n"
        f"💯 **Highest Avg Score:** {highest_avg_model} ({avg_scores[highest_avg_model]:.2f})\n"
        f"⚠️ **Most Hallucinations:** {hallucination_prone}\n"
    )

    win_chart = alt.Chart(win_counts.reset_index()).mark_bar().encode(
        x="index:N", y="winner:Q", tooltip=["index", "winner"]
    ).properties(title="🥇 Win Rate by Model")

    avg_df = pd.DataFrame([(k, v) for k, v in avg_scores.items()], columns=["model", "avg_score"])
    score_chart = alt.Chart(avg_df).mark_bar().encode(
        x="model:N", y="avg_score:Q", tooltip=["model", "avg_score"]
    ).properties(title="📉 Average Score by Model")

    hallu_chart = None
    if hallucinated:
        hallu_df = pd.DataFrame(list(hallucinated.items()), columns=["model", "hallucinations"])
        hallu_chart = alt.Chart(hallu_df).mark_bar().encode(
            x="model:N", y="hallucinations:Q", tooltip=["model", "hallucinations"]
        ).properties(title="⚠️ Hallucination Frequency")

    return summary, win_chart, score_chart, hallu_chart

# 🌐 Gradio UI
with gr.Blocks(title="LLM Evaluator: AI Jury Mode") as demo:
    with gr.Tab("🏁 Evaluate Models"):
        prompt_input = gr.Textbox(label="Enter your prompt", lines=3, placeholder="Ask anything...")
        model_selector = gr.CheckboxGroup(model_options, label="Select models to compare", value=["GPT-3.5", "GPT-4"])
        compare_button = gr.Button("⚖️ Compare Models")

        output_boxes = [gr.Textbox(label=model, lines=5) for model in model_options]
        cost_box = gr.Markdown()

        compare_button.click(
            fn=compare_selected_models,
            inputs=[prompt_input, model_selector],
            outputs=output_boxes + [cost_box]
        )

        with gr.Row():
            eval_focus = gr.Dropdown(choices=evaluation_focus_choices, label="Evaluation Focus", value="Factual Correctness")
            judge_model_dropdown = gr.Dropdown(choices=judge_models, label="Choose Judge Model", value="GPT-4")

        evaluate_button = gr.Button("🧑‍⚖️ Evaluate Models")
        judge_output = gr.Markdown()

        evaluate_button.click(
            fn=judge_models_fn,
            inputs=[prompt_input, judge_model_dropdown, eval_focus] + output_boxes,
            outputs=judge_output
        )

        with gr.Row():
            export_btn = gr.Button("📤 Export to CSV")
            reset_btn = gr.Button("🔄 Reset")

        export_btn.click(fn=export_csv, inputs=[], outputs=gr.File())
        reset_btn.click(fn=reset_all, inputs=[], outputs=output_boxes + [cost_box, judge_output])

    with gr.Tab("📈 Analytics"):
        analytics_summary = gr.Markdown()
        win_chart = gr.Plot()
        score_chart = gr.Plot()
        hallu_chart = gr.Plot()

        gr.Button("🔄 Refresh Analytics").click(
            fn=load_analytics,
            inputs=[],
            outputs=[analytics_summary, win_chart, score_chart, hallu_chart]
        )

demo.launch(share=True)


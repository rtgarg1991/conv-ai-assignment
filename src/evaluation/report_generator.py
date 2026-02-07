"""
Report Generator Module for Hybrid RAG System.

Generates comprehensive HTML evaluation reports with:
- Summary statistics and metrics
- Metric justifications
- Visualizations (charts and tables)
- Ablation study results
- Error analysis breakdown
"""

import json
import csv
import ast
from collections import defaultdict
import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config


# Metric Justifications (as required by assignment)
METRIC_JUSTIFICATIONS = {
    "mrr": {
        "name": "Mean Reciprocal Rank (MRR)",
        "justification": """
            MRR is the mandatory metric for this assignment, measuring how quickly the system 
            identifies the correct source document. It's calculated at the URL level (not chunk level)
            to evaluate document-level retrieval accuracy. A high MRR indicates the system 
            consistently ranks relevant documents near the top of results.
        """,
        "formula": "MRR = (1/|Q|) × Σ(1/rank_i) where rank_i is position of first relevant document",
        "interpretation": """
            - MRR = 1.0: Perfect - correct document always ranked #1
            - MRR > 0.8: Excellent retrieval performance
            - MRR 0.5-0.8: Good, but room for improvement
            - MRR < 0.5: Needs significant improvement
        """,
    },
    "rouge_l": {
        "name": "ROUGE-L (Longest Common Subsequence)",
        "justification": """
            ROUGE-L measures the quality of generated answers by comparing them to reference answers.
            It captures the longest matching sequence of words, rewarding fluency and coherent structure.
            This is particularly valuable for RAG systems where answers should accurately reflect 
            the source content while maintaining readability.
        """,
        "formula": "ROUGE-L F1 = (2 × Precision × Recall) / (Precision + Recall)",
        "interpretation": """
            - ROUGE-L > 0.5: Strong overlap with reference answers
            - ROUGE-L 0.3-0.5: Moderate overlap, captures key information
            - ROUGE-L < 0.3: Low overlap, may indicate answer quality issues
        """,
    },
    "bert_score": {
        "name": "BERTScore (Semantic Similarity)",
        "justification": """
            BERTScore evaluates semantic similarity between generated and reference answers using
            contextual embeddings. Unlike ROUGE, it captures meaning rather than exact word matches,
            making it robust to paraphrasing. This is crucial for RAG systems that may generate
            semantically correct answers using different phrasing than the source.
        """,
        "formula": "BERTScore uses cosine similarity of BERT embeddings, aggregated across tokens",
        "interpretation": """
            - BERTScore > 0.9: Excellent semantic alignment
            - BERTScore 0.7-0.9: Good semantic similarity
            - BERTScore < 0.7: May indicate semantic drift or hallucination
        """,
    },
}


class ReportGenerator:
    """Generates comprehensive HTML evaluation reports."""

    def __init__(self):
        self.data_dir = Config.DATA_DIR
        self.report_path = self.data_dir / "evaluation_report.html"

    def generate_report(self) -> str:
        """
        Generate complete HTML evaluation report.

        Returns:
            Path to generated report file.
        """
        print("Generating evaluation report...")

        # Load all available data
        summary = self._load_json("evaluation_summary.json")
        ablation = self._load_json("ablation_results.json")
        error_analysis = self._load_json("error_analysis.json")
        qa_dataset = self._load_json("qa_dataset.json")
        eval_results = self._load_csv("evaluation_results.csv")

        # Generate HTML
        html = self._build_html(
            summary, ablation, error_analysis, qa_dataset, eval_results
        )

        # Save
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Report saved to {self.report_path}")
        return str(self.report_path)

    def _load_json(self, filename: str) -> Optional[Dict]:
        """Load JSON file if exists."""
        path = self.data_dir / filename
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def _load_csv(self, filename: str) -> Optional[List[Dict]]:
        """Load CSV file if exists."""
        path = self.data_dir / filename
        if not path.exists():
            return None
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _build_html(
        self,
        summary: Optional[Dict],
        ablation: Optional[Dict],
        error_analysis: Optional[Dict],
        qa_dataset: Optional[List],
        eval_results: Optional[List[Dict]],
    ) -> str:
        """Build complete HTML report."""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hybrid RAG Evaluation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #4361ee;
            --secondary: #3f37c9;
            --success: #2ec4b6;
            --warning: #ff9f1c;
            --danger: #e63946;
            --bg: #f8f9fa;
            --card-bg: #ffffff;
            --text: #212529;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: var(--primary); margin-bottom: 0.5rem; }}
        h2 {{ color: var(--secondary); margin: 2rem 0 1rem; border-bottom: 2px solid var(--primary); padding-bottom: 0.5rem; }}
        h3 {{ color: var(--text); margin: 1.5rem 0 0.75rem; }}
        .subtitle {{ color: #6c757d; margin-bottom: 2rem; }}
        .card {{
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }}
        .metric-card {{
            text-align: center;
            padding: 1.5rem;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
        }}
        .metric-value {{ font-size: 2.5rem; font-weight: bold; }}
        .metric-label {{ font-size: 0.9rem; opacity: 0.9; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{ background: var(--primary); color: white; }}
        tr:hover {{ background: #f1f3f5; }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 1rem 0;
        }}
        .justification {{
            background: #e7f5ff;
            border-left: 4px solid var(--primary);
            padding: 1rem;
            margin: 0.5rem 0;
        }}
        .formula {{
            font-family: 'Courier New', monospace;
            background: #f1f3f5;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            display: inline-block;
        }}
        .success {{ color: var(--success); }}
        .warning {{ color: var(--warning); }}
        .danger {{ color: var(--danger); }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .badge-success {{ background: var(--success); color: white; }}
        .badge-warning {{ background: var(--warning); color: white; }}
        .badge-danger {{ background: var(--danger); color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Hybrid RAG System - Evaluation Report</h1>
        <p class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        {self._section_summary(summary)}
        {self._section_metrics_justification()}
        {self._section_ablation(ablation)}
        {self._section_error_analysis(error_analysis)}
        {self._section_retrieval_heatmap(eval_results)}
        {self._section_question_analysis(qa_dataset)}
        {self._section_config(summary)}
    </div>
    
    {self._generate_charts(summary, ablation, error_analysis)}
</body>
</html>"""

    def _section_summary(self, summary: Optional[Dict]) -> str:
        """Generate summary metrics section."""
        if not summary:
            return "<p>No evaluation summary available. Run evaluation first.</p>"

        metrics = summary.get("metrics", {})
        latency = summary.get("latency", {})

        return f"""
        <h2>📊 Performance Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get("mrr", 0):.2%}</div>
                <div class="metric-label">MRR (Retrieval)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get("rouge_l", 0):.2%}</div>
                <div class="metric-label">ROUGE-L (Generation)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get("bert_score", 0):.2%}</div>
                <div class="metric-label">BERTScore (Semantic)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{latency.get("avg_seconds", 0):.2f}s</div>
                <div class="metric-label">Avg Latency</div>
            </div>
        </div>
        <div class="card">
            <p><strong>Questions Evaluated:</strong> {summary.get("num_questions", 0)}</p>
            <p><strong>Failed:</strong> {summary.get("num_failed", 0)}</p>
            <p><strong>Total Processing Time:</strong> {latency.get("total_seconds", 0):.1f}s</p>
        </div>
        """

    def _section_metrics_justification(self) -> str:
        """Generate metrics justification section."""
        sections = []

        for metric_key, info in METRIC_JUSTIFICATIONS.items():
            sections.append(f"""
            <div class="card">
                <h3>{info["name"]}</h3>
                <div class="justification">
                    <strong>Why this metric?</strong>
                    <p>{info["justification"].strip()}</p>
                </div>
                <p><strong>Formula:</strong></p>
                <p class="formula">{info["formula"]}</p>
                <p><strong>Interpretation:</strong></p>
                <pre>{info["interpretation"].strip()}</pre>
            </div>
            """)

        return f"""
        <h2>📐 Metric Justifications</h2>
        <p>The following metrics were selected to evaluate different aspects of the RAG system:</p>
        {"".join(sections)}
        """

    def _section_ablation(self, ablation: Optional[Dict]) -> str:
        """Generate ablation study section."""
        if not ablation:
            return """
            <h2>🔬 Ablation Studies</h2>
            <p class="warning">No ablation results available. Run: <code>python src/evaluation/ablation.py</code></p>
            """

        methods = ablation.get("methods", {})
        analysis = ablation.get("analysis", {})

        rows = ""
        for method, data in sorted(
            methods.items(), key=lambda x: x[1]["mrr"], reverse=True
        ):
            mrr = data["mrr"]
            badge = (
                "badge-success"
                if mrr > 0.7
                else "badge-warning"
                if mrr > 0.5
                else "badge-danger"
            )
            rows += f"""
            <tr>
                <td>{method}</td>
                <td><span class="badge {badge}">{mrr:.4f}</span></td>
            </tr>
            """

        return f"""
        <h2>🔬 Ablation Studies</h2>
        <p>Comparison of different retrieval methods:</p>
        <div class="card">
            <table>
                <thead>
                    <tr><th>Method</th><th>MRR</th></tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <div class="chart-container">
                <canvas id="ablationChart"></canvas>
            </div>
        </div>
        <div class="card">
            <h3>Analysis</h3>
            <p><strong>Best Method:</strong> {analysis.get("best_method", "N/A")} (MRR: {analysis.get("best_mrr", 0):.4f})</p>
            <p><strong>Hybrid vs Single Method Improvement:</strong> {analysis.get("hybrid_improvement_pct", 0):.1f}%</p>
            <p><strong>Recommendation:</strong> {analysis.get("recommendation", "N/A")}</p>
        </div>
        """

    def _section_error_analysis(self, error_analysis: Optional[Dict]) -> str:
        """Generate error analysis section."""
        if not error_analysis:
            return """
            <h2>🔍 Error Analysis</h2>
            <p class="warning">No error analysis available. Run: <code>python src/evaluation/error_analysis.py</code></p>
            """

        summary = error_analysis.get("summary", {})
        categories = summary.get("categories", {})
        recommendations = error_analysis.get("recommendations", [])

        rows = ""
        for cat, data in categories.items():
            color = (
                "success"
                if cat == "SUCCESS"
                else "warning"
                if data["percentage"] < 20
                else "danger"
            )
            rows += f"""
            <tr>
                <td>{cat}</td>
                <td>{data["count"]}</td>
                <td><span class="{color}">{data["percentage"]}%</span></td>
            </tr>
            """

        rec_html = "".join(f"<li>{rec}</li>" for rec in recommendations)

        return f"""
        <h2>🔍 Error Analysis</h2>
        <div class="card">
            <p><strong>Success Rate:</strong> <span class="success">{summary.get("success_rate", 0)}%</span></p>
            <p><strong>Retrieval Accuracy:</strong> {summary.get("retrieval_accuracy", 0)}%</p>
            <table>
                <thead>
                    <tr><th>Category</th><th>Count</th><th>Percentage</th></tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <div class="chart-container">
                <canvas id="errorChart"></canvas>
            </div>
        </div>
        <div class="card">
            <h3>Recommendations</h3>
            <ul>{rec_html}</ul>
        </div>
        """

    def _section_retrieval_heatmap(
        self, eval_results: Optional[List[Dict]]
    ) -> str:
        """Generate a simple retrieval heatmap (rank distribution by question type)."""
        if not eval_results:
            return ""

        # Determine max rank (use configured top-N if present)
        max_rank = (
            int(Config.TOP_N_RETRIEVAL)
            if hasattr(Config, "TOP_N_RETRIEVAL")
            else 10
        )
        if max_rank <= 0:
            max_rank = 10

        # Count ranks by question type
        counts = defaultdict(lambda: defaultdict(int))
        for row in eval_results:
            qtype = row.get("question_type", "unknown") or "unknown"
            gt_url = row.get("ground_truth_url")
            retrieved_raw = row.get("retrieved_urls", "")

            try:
                retrieved = (
                    ast.literal_eval(retrieved_raw)
                    if isinstance(retrieved_raw, str)
                    else retrieved_raw
                )
            except Exception:
                retrieved = []

            # URL-level rank among unique URLs
            seen = set()
            rank = 0
            url_rank = 0
            for url in retrieved or []:
                if url in seen:
                    continue
                seen.add(url)
                url_rank += 1
                if url == gt_url:
                    rank = url_rank
                    break

            if rank <= 0 or rank > max_rank:
                counts[qtype]["miss"] += 1
            else:
                counts[qtype][rank] += 1

        # Find max count for color scaling
        max_count = 1
        for by_rank in counts.values():
            for value in by_rank.values():
                if value > max_count:
                    max_count = value

        # Build table rows
        header_cells = "".join(
            [f"<th>{i}</th>" for i in range(1, max_rank + 1)]
        ) + "<th>Miss</th>"

        rows = ""
        for qtype in sorted(counts.keys()):
            cells = ""
            for i in range(1, max_rank + 1):
                val = counts[qtype].get(i, 0)
                alpha = 0.1 + 0.9 * (val / max_count) if val > 0 else 0.05
                cells += (
                    f"<td style='background-color: rgba(67,97,238,{alpha:.2f})'>{val}</td>"
                )
            miss_val = counts[qtype].get("miss", 0)
            alpha = 0.1 + 0.9 * (miss_val / max_count) if miss_val > 0 else 0.05
            cells += (
                f"<td style='background-color: rgba(230,57,70,{alpha:.2f})'>{miss_val}</td>"
            )
            rows += f"<tr><td>{qtype}</td>{cells}</tr>"

        return f"""
        <h2>🔥 Retrieval Heatmap (Rank Distribution by Question Type)</h2>
        <div class="card">
            <p>Heatmap shows how often the correct URL appears at each rank (Top-{max_rank}).</p>
            <table>
                <thead>
                    <tr><th>Question Type</th>{header_cells}</tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </div>
        """

    def _section_question_analysis(self, qa_dataset: Optional[List]) -> str:
        """Generate question type analysis section."""
        if not qa_dataset:
            return ""

        # Count question types
        type_counts = {}
        for item in qa_dataset:
            qtype = item.get("question_type", "unknown")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

        rows = ""
        for qtype, count in sorted(
            type_counts.items(), key=lambda x: x[1], reverse=True
        ):
            rows += f"<tr><td>{qtype}</td><td>{count}</td></tr>"

        return f"""
        <h2>❓ Question Type Distribution</h2>
        <div class="card">
            <table>
                <thead>
                    <tr><th>Question Type</th><th>Count</th></tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
            <div class="chart-container">
                <canvas id="questionChart"></canvas>
            </div>
        </div>
        """

    def _section_config(self, summary: Optional[Dict]) -> str:
        """Generate configuration section."""
        if not summary:
            return ""

        config = summary.get("config", {})

        return f"""
        <h2>⚙️ System Configuration</h2>
        <div class="card">
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Embedding Model</td><td>{config.get("embedding_model", "N/A")}</td></tr>
                <tr><td>Generation Model</td><td>{config.get("generation_model", "N/A")}</td></tr>
                <tr><td>Top-N Retrieval</td><td>{config.get("top_n_retrieval", "N/A")}</td></tr>
                <tr><td>RRF K Value</td><td>{config.get("rrf_k", "N/A")}</td></tr>
            </table>
        </div>
        """

    def _generate_charts(
        self,
        summary: Optional[Dict],
        ablation: Optional[Dict],
        error_analysis: Optional[Dict],
    ) -> str:
        """Generate Chart.js scripts for visualizations."""
        scripts = ["<script>"]

        # Ablation chart
        if ablation:
            methods = ablation.get("methods", {})
            labels = list(methods.keys())
            values = [methods[m]["mrr"] for m in labels]
            scripts.append(f"""
            new Chart(document.getElementById('ablationChart'), {{
                type: 'bar',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [{{
                        label: 'MRR Score',
                        data: {json.dumps(values)},
                        backgroundColor: ['#4361ee', '#3f37c9', '#7209b7', '#f72585', '#4cc9f0']
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{ y: {{ beginAtZero: true, max: 1 }} }}
                }}
            }});
            """)

        # Error analysis chart
        if error_analysis:
            categories = error_analysis.get("summary", {}).get(
                "categories", {}
            )
            labels = list(categories.keys())
            values = [categories[c]["count"] for c in labels]
            scripts.append(f"""
            new Chart(document.getElementById('errorChart'), {{
                type: 'doughnut',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [{{
                        data: {json.dumps(values)},
                        backgroundColor: ['#2ec4b6', '#ff9f1c', '#e63946', '#4361ee']
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false
                }}
            }});
            """)

        scripts.append("</script>")
        return "\n".join(scripts)


if __name__ == "__main__":
    generator = ReportGenerator()
    generator.generate_report()

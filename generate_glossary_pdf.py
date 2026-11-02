"""
generate_glossary_pdf.py
Generates a publication-grade Technical Reference Glossary PDF:
Abbreviations, Acronyms, and Mathematical Notations
Course: IM60210 Distributed Analytics for Industrial Automation
IIT Kharagpur
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = os.path.abspath("Technical_Glossary_Abbreviations_Symbols.pdf")

# Custom Canvas for Dynamic Page Numbering
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#57606A"))
        
        # Header rule & text on pages > 1
        if self._pageNumber > 1:
            self.setStrokeColor(colors.HexColor("#D0D7DE"))
            self.setLineWidth(0.5)
            self.line(36, letter[1] - 30, letter[0] - 36, letter[1] - 30)
            self.drawString(36, letter[1] - 25, "IM60210: Distributed Analytics | Mine Safety MARL Glossary")
            self.drawRightString(letter[0] - 36, letter[1] - 25, "Mun Sarthak Satish (22IM30021)")

        # Footer rule & text
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.5)
        self.line(36, 40, letter[0] - 36, 40)
        
        footer_text = "Dept. of Industrial & Systems Engineering, IIT Kharagpur | Instructor: Prof. Subhajit Sidhanta"
        self.drawString(36, 28, footer_text)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 36, 28, page_str)
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    KGP_BLUE = colors.HexColor("#002147")
    KGP_MAROON = colors.HexColor("#800000")
    KGP_GOLD = colors.HexColor("#B8860B")
    TEXT_DARK = colors.HexColor("#1F2328")
    TEXT_MUTED = colors.HexColor("#57606A")
    BG_LIGHT = colors.HexColor("#F6F8FA")
    BORDER_LIGHT = colors.HexColor("#D0D7DE")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=KGP_BLUE,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=KGP_MAROON,
        spaceAfter=2
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=TEXT_MUTED,
        spaceAfter=8
    )

    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=KGP_BLUE,
        spaceBefore=10,
        spaceAfter=4
    )

    tbl_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white
    )

    term_style = ParagraphStyle(
        'TermStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=KGP_BLUE
    )

    fullform_style = ParagraphStyle(
        'FullFormStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10.5,
        textColor=KGP_MAROON
    )

    desc_style = ParagraphStyle(
        'DescStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=TEXT_DARK
    )

    math_term_style = ParagraphStyle(
        'MathTermStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=KGP_MAROON
    )

    story = []

    # Title & Header Banner
    story.append(Paragraph("Distributed Safety and Risk Analytics for Underground Mines using Computer Vision & MARL", title_style))
    story.append(Paragraph("TECHNICAL REFERENCE GLOSSARY: ABBREVIATIONS, ACRONYMS & MATHEMATICAL FORMULATIONS", subtitle_style))
    story.append(Paragraph(
        "<b>Course:</b> IM60210 — Distributed Analytics for Industrial Automation &nbsp;|&nbsp; "
        "<b>Author:</b> Mun Sarthak Satish (Roll No: 22IM30021)<br/>"
        "<b>Department:</b> Department of Industrial and Systems Engineering, IIT Kharagpur &nbsp;|&nbsp; "
        "<b>Instructor:</b> Prof. Subhajit Sidhanta",
        meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=KGP_BLUE, spaceAfter=8))

    # Helper function to generate clean tables
    def make_table(data_rows, col_widths=[90, 150, 300]):
        table_data = [[
            Paragraph("<b>Term / Symbol</b>", tbl_header_style),
            Paragraph("<b>Full Form / Notation</b>", tbl_header_style),
            Paragraph("<b>Technical Meaning & Project Relevance</b>", tbl_header_style)
        ]]
        for row in data_rows:
            table_data.append([
                Paragraph(row[0], term_style),
                Paragraph(row[1], fullform_style),
                Paragraph(row[2], desc_style)
            ])
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), KGP_BLUE),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER_LIGHT),
        ]))
        return t

    # -------------------------------------------------------------------------
    # SECTION 1: MARL & GAME THEORY
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Multi-Agent Reinforcement Learning (MARL) & Game Theory", section_style))
    marl_data = [
        ("MARL", "Multi-Agent Reinforcement Learning",
         "A framework where multiple autonomous agents (Scout Drone, Rescue Rover, Smart Ventilation Controller) learn policies simultaneously in a shared stochastic environment."),
        ("CTDE", "Centralized Training with Decentralized Execution",
         "Core paradigm where a Centralized Critic evaluates the global state during training, while decentralized actors execute strictly on local partial observations during live deployment."),
        ("MAPPO", "Multi-Agent Proximal Policy Optimization",
         "An extension of PPO for cooperative multi-agent teams using clipped surrogate objectives and Generalized Advantage Estimation (GAE) to guarantee policy gradient stability."),
        ("PPO", "Proximal Policy Optimization",
         "Policy gradient algorithm that clips probability ratio updates within 1 ± ε (ε=0.2) to prevent catastrophic destabilizing policy shifts."),
        ("IQL", "Independent Q-Learning",
         "Decentralized baseline where each agent optimizes an independent Q-network treating all other agents as environment noise (induces non-stationarity)."),
        ("POMDP", "Partially Observable Markov Decision Process",
         "A Markov decision process where agents only perceive noisy, incomplete partial observations of the underlying global state."),
        ("Dec-POMDP", "Decentralized Partially Observable Markov Decision Process",
         "Formal mathematical tuple ⟨S, {Aᵢ}, P, {Rᵢ}, {Ωᵢ}, O, γ⟩ modeling cooperative teams under communication and sensory constraints."),
        ("GAE", "Generalized Advantage Estimation",
         "An exponentially weighted advantage estimator balancing bias and variance in policy gradient updates through hyperparameter λ."),
        ("Nash Equilibrium", "Pareto Nash Equilibrium",
         "A multi-agent state where no agent can unilaterally adjust its policy to improve safety without disproportionately worsening energy or productivity penalties.")
    ]
    story.append(make_table(marl_data))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 2: COMPUTER VISION & EDGE PERCEPTION
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Computer Vision & Edge Perception Analytics", section_style))
    cv_data = [
        ("CV", "Computer Vision",
         "Algorithms extracting high-level understanding and geometric hazard coordinates directly from digital video frames."),
        ("CCTV", "Closed-Circuit Television",
         "Subterranean optical cameras monitoring underground coal extraction faces, haulage rail corridors, and ventilation shafts."),
        ("HSV", "Hue, Saturation, Value",
         "Cylindrical color model isolating chromatic components from intensity, enabling robust segmentation of reflective neon PPE and amber vehicles under poor mine lighting."),
        ("NMS", "Non-Maximum Suppression",
         "Clustering filter that eliminates redundant, overlapping bounding boxes to retain the single highest-confidence detection per worker."),
        ("FPS", "Frames Per Second",
         "Perception throughput metric; the edge vision engine operates at 20 FPS zero-frame-drop real-time processing."),
        ("HUD", "Heads-Up Display",
         "Real-time visual overlay superimposed over CCTV video displaying active hazards, risk index gauge, worker labels, and dynamic proximity alert vectors."),
        ("PPE", "Personal Protective Equipment",
         "Mandatory underground miner gear (high-visibility neon vests and hardhat helmets); violations are detected via upper-torso HSV cropping."),
        ("px", "Pixels",
         "Discrete coordinate unit in 640×480 CCTV frames; hazard alert threshold is triggered when worker-machinery distance is < 75 px.")
    ]
    story.append(make_table(cv_data))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 3: INDUSTRIAL AUTOMATION, MINING & DISTRIBUTED SYSTEMS
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Industrial Automation, Mining SCADA & Distributed Systems", section_style))
    scada_data = [
        ("SCADA", "Supervisory Control and Data Acquisition",
         "Industrial telemetry and process control infrastructure connecting field PLCs, ventilation fans, and surface operator displays."),
        ("DGMS", "Directorate General of Mines Safety",
         "Statutory Indian regulatory authority enforcing underground mine safety limits, including the 5% worker incident safety ceiling."),
        ("OSHA", "Occupational Safety and Health Administration",
         "Global occupational health standard mandating zero-harm proximity controls around heavy continuous mining machinery."),
        ("IIoT / IoT", "Industrial Internet of Things",
         "Distributed network of intelligent edge cameras, gas sensors, autonomous rovers, and ventilation actuators."),
        ("JSONL", "JSON Lines (Newline-Delimited JSON)",
         "Lightweight streaming message bus format (hazard_stream.jsonl) transmitting structured risk events from Edge CV to the MARL twin."),
        ("SQLite", "Structured Query Language (Lite)",
         "Embedded, zero-configuration ACID relational document store persisting 2,031 real-time hazard events with 0.42 ms query latency."),
        ("ACID", "Atomicity, Consistency, Isolation, Durability",
         "Database transaction reliability guarantees ensuring zero telemetry event loss during underground network disconnects."),
        ("WAL", "Write-Ahead Logging",
         "High-concurrency SQLite journaling mode allowing simultaneous edge event appending and SCADA dashboard querying."),
        ("PLC", "Programmable Logic Controller",
         "Hardened industrial microcomputers executing electrical relay commands for auxiliary ventilation fans and haulage conveyors."),
        ("Modbus TCP", "Modbus Transmission Control Protocol",
         "Standard industrial fieldbus protocol transmitting discrete register values between the MARL controller and field actuators."),
        ("OPC-UA", "Open Platform Communications Unified Architecture",
         "Cross-platform, secure industrial interoperability framework connecting the digital twin to enterprise SCADA platforms."),
        ("VFD", "Variable Frequency Drive",
         "Power electronics controller modulating the rotational speed and airflow rate of main and auxiliary mine ventilation fans.")
    ]
    story.append(make_table(scada_data))
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 4: MATHEMATICAL FORMULATIONS & SYMBOLS
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Mathematical Formulations & Dec-POMDP Formal Symbols", section_style))
    math_data = [
        ("M", "⟨S, {Aᵢ}, P, {Rᵢ}, {Ωᵢ}, O, γ⟩",
         "Decentralized Partially Observable Markov Decision Process formal tuple defining the underground multi-agent system."),
        ("S", "Global State Space ∈ ℝ²⁰⁸",
         "Complete environment state containing 10×10 Risk Grid (100), 10×10 Uncertainty Grid (100), and Global Fleet Telemetry (8)."),
        ("o₀", "Scout Drone Observation ∈ ℝ²⁸",
         "Local 5×5 risk patch (25) + Normalized coordinates (x/9, y/9) + Normalized battery level (b)."),
        ("o₁", "Barrier Rover Observation ∈ ℝ²⁸",
         "Local 5×5 risk patch (25) + Normalized coordinates (x/9, y/9) + Remaining barrier inventory (k/4)."),
        ("o₂", "Vent Controller Observation ∈ ℝ⁷",
         "Mean sector risk scores (A, B, C, D) + Ventilation Level (v/3) + Haulage Status (h/3) + Un-evacuated worker ratio."),
        ("A₀", "Drone Action Space (9 discrete)",
         "0: Hover/Stay, 1–4: Cardinal 1-step moves (Up, Down, Left, Right), 5–8: Cardinal high-speed 2-step burst moves."),
        ("A₁", "Rover Action Space (6 discrete)",
         "0: Stay, 1–4: Cardinal 1-step navigation, 5: Deploy physical containment barrier and evacuate adjacent worker."),
        ("A₂", "Vent Controller Action Space (4 states)",
         "0: Low Vent / Normal Haul, 1: Med Vent / Normal Haul, 2: High Vent / Throttled Haul, 3: Max Emergency Vent / Halt Haul."),
        ("P", "P(s' | s, a)",
         "State transition probability distribution mapping joint actions a=(a₀, a₁, a₂) and current state s to next state s'."),
        ("R_total", "Shared Cooperative Team Reward",
         "R_total = r_scout + r_rover + r_vent + 3 · (-0.5 Σ R(x,y) - 25.0 Σ I(worker injured))."),
        ("V_φ(S)", "Centralized Critic Value Function",
         "Neural network parameterized by φ predicting expected cumulative discounted return from global state S; eliminates non-stationarity."),
        ("π_θᵢ(aᵢ | oᵢ)", "Decentralized Actor Policy",
         "Independent neural network for agent i parameterized by θᵢ mapping local partial observation oᵢ directly to action probabilities."),
        ("ρ_t", "Importance Sampling Ratio",
         "Ratio ρ_t = π_θ(a_t | o_t) / π_θ_old(a_t | o_t) measuring policy divergence between PPO optimization iterations."),
        ("Â_t", "Generalized Advantage Estimate",
         "Advantage metric measuring how much better action a_t performed relative to baseline state expectation V(s_t)."),
        ("ε", "PPO Clipping Epsilon (0.2)",
         "Hyperparameter constraining the policy update ratio within [1 - ε, 1 + ε] to guarantee monotonic policy improvement."),
        ("H(π)", "Policy Entropy",
         "Information-theoretic entropy regularization term promoting exploratory behavior across early training episodes."),
        ("γ", "Discount Factor (0.98)",
         "Temporal weighting factor balancing immediate hazard penalties against long-term operational costs."),
        ("R(x, y)", "Cell Risk Index ∈ [0.0, 1.0]",
         "Continuous spatial risk severity at grid cell (x, y) dynamically injected from real-time Edge CV detection events."),
        ("U(x, y)", "Uncertainty Index ∈ [0.0, 1.0]",
         "Information uncertainty at cell (x, y); accumulates over time and is cleared to 0.1 by Scout Drone reconnaissance.")
    ]
    story.append(make_table(math_data, col_widths=[75, 165, 300]))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✓ PDF generated successfully: {PDF_OUTPUT_PATH} ({os.path.getsize(PDF_OUTPUT_PATH)/1024:.1f} KB)")


if __name__ == "__main__":
    build_pdf()

Leadership Score v4 – 7-Dimensional Model

Leadership Score v4 is a resume analysis system that extracts leadership-related signals from PDF resumes and evaluates candidates using a structured 7-dimensional leadership model. This version replaces earlier keyword-only scoring approaches with a dimension-based evaluation framework.

Overview

   The system:
   Parses a PDF resume
   Extracts relevant leadership signals
   Scores the resume across seven leadership dimensions
   Produces a final composite leadership score
   The scoring model is deterministic and rule-based (not ML-based), designed for transparency and interpretability.


7 Leadership Dimensions
The model evaluates leadership across the following dimensions:
   
      1.Strategic Thinking: Long-term planning, vision, decision-making, roadmap ownership.   
      2.Execution & Results: Delivery metrics, KPIs, measurable outcomes, performance impact.   
      3.Initiative & Ownership: Founding initiatives, driving projects independently, accountability.   
      4.Influence & Communication: Stakeholder management, presentations, cross-functional alignment.   
      5.Team Leadership: Managing teams, mentoring, recruiting, organizational coordination.   
      6.Innovation & Problem Solving: Process improvement, systems building, optimization, technical innovation.   
      7.Impact & Scale: Growth metrics, revenue impact, scaling systems, operational expansion.

Each dimension is scored independently and combined into an overall leadership score.




Project Structure

      resume_extractor/
      │
      ├── parser.py
      ├── leadership_scorer_v4.py
      ├── requirements.txt
      └── README.md
      
      parser.py – Entry point. Handles PDF parsing and orchestration.
      leadership_scorer_v4.py – Implements the 7-dimensional scoring logic.
      requirements.txt – Python dependencies.


Requirements
      Python 3.8+
      pip

Install dependencies: 

      pip install -r requirements.txt
      python -m spacy download en_core_web_sm 


Usage

Run the parser with a resume in PDF of docx format:   

      python parser.py your_resume.pdf
   
   OR
      
      python parser.py your_resume.docx



Design Rationale

   The v4 architecture focuses on:
      Dimensional separation instead of flat keyword scoring
      Interpretability (clear mapping between signals and score)
      Extensibility (new dimensions or weights can be added easily)
      Deterministic scoring for reproducibility




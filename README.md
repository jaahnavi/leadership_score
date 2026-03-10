Leadership Score v4 – 7-Dimensional Model

Leadership Score v4 is a resume analysis system that extracts leadership-related signals from PDF and docx resumes and evaluates candidates using a structured 7-dimensional leadership model and stores the result ina SQLite database. Designed for transparency, auditability and ML integration.

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
      ├── parser.py --Entry point: batch/single scoring, SQLite storage, terminal summary
      ├── leadership_scorer.py --NLP feature extraction and 7-dimension scoring logic
      ├── weights.json --Dimension weights — edit this to update weights of scoring logic
      ├── calibrate_and_train.py --Two-level regression: per-dimension R² + weight optimisation
      ├── verbs.json --Influence, initiative, and mentorship verb sets
      ├── update_verbs.py --Add API key and run to update verb.json 
      ├── leadership_scores.db --SQLite DB, auto-created on first run
      ├── requirements.txt --Python dependencies
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
Batch scoring

      python parser.py resume/              # score all, save to DB
      python parser.py resume/ --force      # rescore even unchanged files
      python parser.py resume/ --out results  # also write per-resume JSON

Single file

      python parser.py candidate.pdf
      python parser.py candidate.docx --out results

Calibration and weight optimisation

      python calibrate_and_train.py --csv scores_norm.csv
      python calibrate_and_train.py --csv scores_norm.csv --export-weights weights_ml.json

Updating Weights
When the ML model produces new weights, update only the weights object in weights.json — no code changes needed:

      {
        "weights": {
          "influence": 20.0, "impact": 25.0, "initiative": 15.0,
          "mentorship": 15.0, "scope_scale": 15.0, "ownership": 10.0, "seniority": 10.0
        },
        "version": "ml-v1"
      }

The weights_version is saved in the DB with every score for auditability.





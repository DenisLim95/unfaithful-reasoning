I'm working on a mechanistic interpretability research project. I've attached a project overview document that describes the goals, methodology, and high-level plan.

Please read the project description titled "project_1_cot_unfaithfulnes.md" and create a detailed technical specification that translates the high-level overview into concrete, actionable implementation steps.

## What I Need in the Spec

Generate a comprehensive technical specification markdown that includes:

### 1. Project Structure
- Repository organization (directories, files)
- Data storage approach
- Results/outputs organization

### 2. Technical Implementation Details
- Exact models to use (with HuggingFace paths)
- Libraries and dependencies (versions)
- Hardware requirements and compute estimates
- Environment setup steps

### 3. Data Generation Pipeline
- How to generate question pairs (with categories and quantities)
- Prompt templates for the models
- Response collection and storage format
- Quality control measures

### 4. Faithfulness Evaluation Pipeline
- Detailed algorithm for consistency scoring
- Automated answer extraction approach
- Edge case handling
- Manual validation process

### 5. Mechanistic Analysis Methods
Specify concrete implementations for:
- Activation caching and extraction
- Linear probe training procedure
- Attention pattern analysis
- Statistical tests to use

### 6. Experiment Design
- Specific experiments to run (with parameters)
- Baselines and ablations
- Expected outputs for each experiment
- Success criteria

### 7. Analysis & Visualization
- Key metrics to track
- Graphs/plots to generate
- Statistical comparisons needed

### 8. Timeline Breakdown
- Detailed hour-by-hour schedule
- Milestones and checkpoints
- Contingency plans

### 9. Code Architecture
- Main scripts and their purposes
- Function signatures for key components
- Data flow between components

### 10. Deliverables
- Final report structure
- What figures/tables to include
- How to present findings

## Output Requirements

Generate a markdown document with:
- Clear section headers
- Code snippets for key components
- Specific parameter values (not "tune as needed")
- Concrete file names and paths
- Actionable steps (not vague guidance)

Make it detailed enough that I could hand it to another researcher and they could execute the project without further clarification.
# Multi-Agent Misconception Analysis System

A multi-agent system designed to enhance learning environments by predicting student misconceptions based on their answers to questions. This project uses **dspy** and **OpenAI** to process questions, generate potential misconceptions, and evaluate semantic similarities between these misconceptions, facilitating a deeper understanding of student reasoning.

## Features

- **Multi-Agent Communication**: Implements the Exchange-of-Thought (EoT) framework inspired by large language model communication paradigms like Memory, Report, Relay, and Debate.
- **Misconception Prediction**: Identifies the reasoning behind incorrect answers provided by students.
- **Semantic Evaluation**: Analyzes and ranks misconceptions based on their similarity to student reasoning.

## Getting Started

### Prerequisites

- Python 3.8+
- `dspy` library
- OpenAI API key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/CS194Group/multi-agent-misconceptions.git
   cd multi-agent-misconceptions
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Training and Evaluation

Run the main script to process questions and evaluate misconceptions:
`bash
    python main.py
    `

#### Data

Place your training, testing, and validation datasets in the `data/` directory.

### Project Structure

- `agents/`: Contains multi-agent logic for generating and evaluating misconceptions.
- `data/`: Includes training, testing, and validation datasets.
- `dataloader.py`: Preprocessing and data loading utilities.
- `evaluation.py`: Performance evaluation and reporting tools.
- `main.py`: Entry point for training and inference.
- `predictmode.py`: Prediction logic for generating misconceptions.
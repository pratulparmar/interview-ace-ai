"""
Evaluator Agent - InterviewAce AI

Purpose:
    Evaluates candidate answers and provides detailed feedback.
    Scores answers on multiple criteria and generates comprehensive reports.

What This Agent Does:
    1. Scores individual answers (technical accuracy, completeness, etc.)
    2. Identifies strengths and weaknesses
    3. Provides constructive feedback
    4. Generates overall interview assessment
    
Learning Focus:
    - Multi-criteria evaluation
    - Rubric-based scoring
    - Structured output parsing
    - Report generation
"""


# ============================ PATH SETUP ===================================

import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent.parent))


from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from backend.config.settings import get_settings, setup_langsmith
from backend.agents.question_generator import InterviewQuestion


#============================ DEMO CODE =====================================

class AnswerScore(BaseModel):
    """
    Scores for a single answer across multiple criteria.
    
    Why separate model: Keeps scoring logic organized and reusable.
    
    Each score is 0-10:
    - 0-2: Poor
    - 3-4: Below average
    - 5-6: Average
    - 7-8: Good
    - 9-10: Excellent
    """
    technical_accuracy: float = Field(
        description = "How technically correct is the answer? (0-10)",
        ge = 0.0,
        le = 10.0
    )

    completeness: float = Field(
        description = "Does it cover all expected topics?(0-10)",
        ge = 0.0,
        le = 10.0
    )

    communication: float = Field(
        description = "How clear and well structured is the answer?(0-10)",
        ge = 0.0,
        le = 10.0
    )

    depth: float = Field(
        description = "How deep is the understanding shown?(0-10)",
        ge = 0.0,
        le =10.0
    )

    relevance: float = Field(
        description = "How relevant is the answer to the question?(0-10)",
        ge = 0.0,
        le = 10.0
    )

class AnswerEvaluation(BaseModel):
    """
    Complete evaluation of a single answer.
    
    Why: Combines scores with qualitative feedback.
    Use: One evaluation per question-answer pair.
    """

    question: str = Field(description = "The interview question")
    answer: str = Field(description = "Candidate's answer")

    scores: AnswerScore = Field(description = "Detailed scores across criteria")

    overall_score: float = Field(
        description = "Weighted overall score (0-10)",
        ge = 0.0,
        le = 10.0
    )

    # Qualitative feedback
    strengths: List[str] = Field(
        description = "List of answer strengths (2-4 items)"
    )

    weaknesses: List[str] = Field(
        description = "List of areas for improvement (1-3 items)"
    )

    detailed_feedback: str = Field(
        description = "Constructive paragraph of feedback(3-5 sentences)"
    )

    # Whether asnswer passes this question

    pass_fail: str = Field(
        description = "PASS or FAIL based on overall score",
        pattern = "^(PASS|FAIL)$"
    )


class InterviewReport(BaseModel):
    """
    Complete interview assessment report.
    
    Why: Summarizes entire interview, not just individual answers.
    Use: One report per candidate interview.
    """
    candidate_name: str = Field(description = "candidate's name")
    role: str = Field(description = "Position interviewd for")

    # Overall statistics
    total_questions: int = Field(description = "Number of questions asked")
    questions_passed: int = Field(description = "Number of questions passed")
    questions_failed: int = Field(description = "Number of questions failed")

    # Aggregate scores
    average_score: float = Field(
        description = "Average score across all questions(0-10)",
        ge = 0.0,
        le = 10.0
    )

    # Breakdown by category
    category_scores: Dict[str, float] = Field(
        description = "Average score per category (behavioral, technical, etc.)"
    )

    # Summary Feedback

    overall_strengths: List[str] = Field(
        description = "Key strength shown across interview (3-5 items)"
    )

    overall_weaknesses: List[str] = Field(
        description = "Key areas for improvement (2-4 items)"
    )


    # Final Recommendation
    recommendation: str =Field(
        description = "Hiring recommendation: STRONG_YES, YES, MAYBE, NO, STRONG_NO"
    )

    summary: str =Field(
        description = "Executive summary paragraph(4-6 sentences)"
    )


# ========================= EVALUATOR AGENT CLASS =====================

class EvaluatorAgent:
    """
    Agent that evaluates interview answers and generates reports.
    
    Architecture:
    1. Scores individual answers using rubric
    2. Calculates weighted scores
    3. Generates constructive feedback
    4. Compiles overall interview report
    """

    def __init__(self):
        """
        Initialize the evaluator agent.
        
        What happens here:
        1. Load settings (API keys, model config)
        2. Create LLM connection
        3. Build prompt templates
        4. Setup output parsers
        """

        # Load Configuration
        self.settings = get_settings()

        # Create LLM for evaluation
        self.llm = ChatOpenAI(
            model = self.settings.openai_model,
            temperature = 0.3,
            openai_api_key = self.settings.openai_api_key,
            max_tokens = 1500
        )

        # Scoring weights for overall calculation
        self.score_weights = {
            "technical_accuracy": 0.30,
            "completeness": 0.25,
            "communication": 0.20,
            "depth": 0.15,
            "relevance": 0.10,
        }

        print("Evaluator Agent initialized")

    def evaluate_answer(
            self,
            question: InterviewQuestion,
            answer: str
    ) -> AnswerEvaluation:
        """
        Evaluate a single answer against a question.
        
        Why this method:
        - Core functionality of evaluator
        - Can evaluate one answer at a time
        - Used by interview process
        
        Args:
            question: The interview question (InterviewQuestion object)
            answer: Candidate's text answer (string)
        
        Returns:
            AnswerEvaluation: Complete evaluation with scores and feedback
        
        Process:
        1. Create evaluation prompt
        2. Send to GPT for scoring
        3. Parse GPT's response into AnswerEvaluation
        4. Calculate weighted overall score
        5. Return structured evaluation
        """

        # Output Parser
        parser = PydanticOutputParser(pydantic_object = AnswerEvaluation)

        # Prompt for Evaluation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical interviewer and evaluator.
Your job is to fairly and objectively evaluate interview answers.

SCORING RUBRIC (0-10 for each criterion):

1. TECHNICAL ACCURACY (30% weight)
   9-10: Perfect technical understanding, no errors
   7-8: Good understanding, minor gaps
   5-6: Basic understanding, some errors
   3-4: Limited understanding, major errors
   0-2: Incorrect or irrelevant

2. COMPLETENESS (25% weight)
   9-10: Covers all expected topics thoroughly
   7-8: Covers most topics adequately
   5-6: Covers some topics, missing key points
   3-4: Missing many expected topics
   0-2: Minimal coverage

3. COMMUNICATION (20% weight)
   9-10: Clear, concise, well-structured
   7-8: Generally clear, minor issues
   5-6: Somewhat unclear
   3-4: Difficult to follow
   0-2: Incoherent

4. DEPTH (15% weight)
   9-10: Deep insights, expert-level
   7-8: Good depth, shows understanding
   5-6: Adequate, somewhat surface level
   3-4: Shallow, lacks detail
   0-2: No depth

5. RELEVANCE (10% weight)
   9-10: Perfectly on-topic
   7-8: Mostly relevant
   5-6: Somewhat relevant
   3-4: Partly off-topic
   0-2: Completely off-topic

IMPORTANT:
- Be fair but honest
- Provide constructive feedback
- Identify 2-4 specific strengths
- Identify 1-3 areas for improvement
- Overall score above 7.0 = PASS, below = FAIL
{format_instructions}"""),
            ("user", """Evaluate this interview answer:

QUESTION:
{question}

QUESTION CATEGORY: {category}
QUESTION DIFFICULTY: {difficulty}

EXPECTED TOPICS TO COVER:
{expected_topics}

CANDIDATE'S ANSWER:
{answer}

Provide detailed evaluation now.""")
        ])

        # Create the chain
        chain = prompt |self.llm | parser

        # Invoke the chain
        try:
            result = chain.invoke({
                "question": question.question,
                "category": question.category,
                "difficulty":question.difficulty,
                "expected_topics":"\n".join([f"-{topic}" for topic in question.expected_topics]),
                "answer": answer,
                "format_instructions": parser.get_format_instructions()

            })


            # Calculate weighted overall score
            overall = (
                result.scores.technical_accuracy*self.score_weights["technical_accuracy"] + 
                result.scores.completeness*self.score_weights["completeness"] +
                result.scores.communication*self.score_weights["communication"] +
                result.scores.depth*self.score_weights["depth"] +
                result.scores.relevance*self.score_weights["relevance"]
            )

            # Update the overall score in result
            result.overall_score = round(overall, 1)  #Round to one decimal

            # Ensure Pass/Fail matches score
            result.pass_fail = "PASS" if result.overall_score >=7.0 else "FAIL"
            
            return result
        
        except Exception as e:
            # If evaluation fails, print error
            print(f"Error evaluating answer: {e}")
            raise
        

    def generate_report(
            self,
            candidate_name: str,
            role: str,
            evaluations: List[AnswerEvaluation]
    ) -> InterviewReport:
        """
        Generate comprehensive interview report from all evaluations.
        
        Why this method:
        - Summarizes entire interview
        - Provides hiring recommendation
        - Gives executive-level overview
        
        Args:
            candidate_name: Name of candidate
            role: Position interviewed for
            evaluations: List of all answer evaluations
        
        Returns:
            InterviewReport: Complete interview assessment
        
        Process:
        1. Calculate aggregate statistics
        2. Group scores by category
        3. Identify overall patterns
        4. Generate recommendation
        5. Create summary
        """

        # Calculate basic statistics
        total_questions = len(evaluations)
        questions_passed = sum(1 for e in evaluations if e.pass_fail == "PASS")
        questions_failed = total_questions - questions_passed

        # Calculate average score
        average_score = sum(e.overall_score for e in evaluations)/total_questions if total_questions > 0 else 0.0

        # Calculate scores by category
        category_scores: Dict[str,List[float]]={}
        for eval in evaluations:
            # Extract category from question
            pass

        category_scores = {
            "overall": average_score
        }

        # Collect all strengths and weaknesses

        all_strengths = []
        all_weaknesses = []

        for eval in evaluations:
            all_strengths.extend(eval.strengths)
            all_weaknesses.extend(eval.weaknesses)

        # Deduplicate and take top items
        # Why : Don't repeat the same feedback
        overall_strengths = list(set(all_strengths))[:5]
        overall_weaknesses = list(set(all_weaknesses))[:4]

        # Determine recommendations 

        pass_rate = questions_passed / total_questions if total_questions > 0 else 0

        if average_score >= 9.0 and pass_rate >= 0.9:
            recommendation = "STRONG_YES"
        elif average_score >= 7.5 and pass_rate >= 0.8:
            recommendation = "YES"
        elif average_score >= 6.5 and pass_rate >= 0.6:
            recommendation = "MAYBE"
        elif average_score >= 5.0:
            recommendation = "NO"
        else:
            recommendation = "STRONG_NO"

        # Generate summary using LLM

        summary = self._generate_summary(
            candidate_name = candidate_name,
            role = role,
            average_score = average_score,
            pass_rate = pass_rate,
            strengths = overall_strengths,
            weaknesses = overall_weaknesses,
            recommendation = recommendation
        )

        # Create and return report

        return InterviewReport(
            candidate_name = candidate_name,
            role = role,
            total_questions = total_questions,
            questions_passed = questions_passed,
            questions_failed = questions_failed,
            average_score = round(average_score,1),
            category_scores = category_scores,
            overall_strengths = overall_strengths,
            overall_weaknesses = overall_weaknesses,
            recommendation = recommendation,
            summary = summary
        )
    
    def _generate_summary(
            self,
            candidate_name: str,
            role: str,
            average_score: float,
            pass_rate:float,
            strengths: List[str],
            weaknesses: List[str],
            recommendation: str
    ) -> str:
        """ Generate executive summary using LLM"""
        prompt = ChatPromptTemplate([
            ("system",""" You are a hiring manager writing an executive summary.
Create a professional 4-6 sentences summary of the interview.
Include:
1. Overall impression
2. Key strengths
3. Areas of concern (if any)
4. Recommendation context

Tone: Professional, balanced, constructive"""),
            ("user","""Write interview summary:

Candidate: {name}
Role: {role}
Average Score: {score}/10
Pass Rate: {pass_rate}%
Recommendation: {recommendation}

Key Strengths:
{strengths}

Areas for Improvement:
{weaknesses}""")
        ])

        chain = prompt | self.llm

        result = chain.invoke({
            "name": candidate_name,
            "role": role,
            "score": round(average_score, 1),
            "pass_rate": round(pass_rate * 100, 0),
            "recommendation": recommendation.replace("_", " "),
            "strengths": "\n".join([f"- {s}" for s in strengths[:3]]),
            "weaknesses": "\n".join([f"- {w}" for w in weaknesses[:2]])
        })

        return result.content
    

# ============================ UTILITY FUNCTIONS ========================

def display_evaluation(evaluation: AnswerEvaluation):
    """ Display evaluation in readable format"""

    print("\n" + "="*80)
    print("ANSWER EVALUATION")
    print("="*80)

    print(f"\nQuestion: {evaluation.question}")
    print(f"\nAnswer: {evaluation.answer[:200]}...")

    print(f"\n--- SCORES ---")
    print(f"Technical Accuracy: {evaluation.scores.technical_accuracy}/10")
    print(f"Completeness: {evaluation.scores.completeness}/10")
    print(f"Communication: {evaluation.scores.communication}/10")
    print(f"Depth: {evaluation.scores.depth}/10")
    print(f"Relevance: {evaluation.scores.relevance}/10")

    print(f"\nOVERALL SCORE: {evaluation.overall_score}/10")
    print(f"RESULT: {evaluation.pass_fail}")

    print(f"\n--- STRENGTHS ---")
    for s in evaluation.strengths:
        print(f"+ {s}")

    print(f"\n--- AREAS FOR IMPROVEMENT ---")
    for w in evaluation.weaknesses:
        print(f"- {w}")
    
    print(f"\n--- FEEDBACK ---")
    print(evaluation.detailed_feedback)
    print("\n" + "="*80)


def display_report(report: InterviewReport):
    """Display interview report in readable format"""
    print("\n" + "="*80)
    print("INTERVIEW ASSESSMENT REPORT")
    print("="*80)
    
    print(f"\nCandidate: {report.candidate_name}")
    print(f"Role: {report.role}")
    
    print(f"\n--- STATISTICS ---")
    print(f"Total Questions: {report.total_questions}")
    print(f"Questions Passed: {report.questions_passed}")
    print(f"Questions Failed: {report.questions_failed}")
    print(f"Average Score: {report.average_score}/10")
    
    print(f"\n--- KEY STRENGTHS ---")
    for s in report.overall_strengths:
        print(f"+ {s}")
    
    print(f"\n--- AREAS FOR IMPROVEMENT ---")
    for w in report.overall_weaknesses:
        print(f"- {w}")
    
    print(f"\n--- RECOMMENDATION ---")
    print(f"{report.recommendation}")
    
    print(f"\n--- EXECUTIVE SUMMARY ---")
    print(report.summary)
    
    print("\n" + "="*80)



# =========================== DEMO CODE ===================================

if __name__ == "__main__":
    """
    Demo the Evaluator Agent
    
    """
    setup_langsmith()
    
    print("\n" + "="*80)
    print("EVALUATOR AGENT - DEMO")
    print("="*80)
    
    # Initialize evaluator
    evaluator = EvaluatorAgent()
    
    # Sample question
    question = InterviewQuestion(
        question="Explain your experience with Python and give examples of projects you've worked on.",
        category="technical",
        difficulty="medium",
        expected_topics=["Python experience", "Project examples", "Technical skills", "Frameworks"],
        follow_up="What Python frameworks have you used?"
    )
    
    # Sample answer (good answer)
    good_answer = """I have about 5 years of Python experience, primarily in data engineering 
and AI projects. I started using it for automating design calculations when I was a mechanical 
engineer, but transitioned into full-time software development.

One of my main projects was building a data pipeline using Pandas and Airflow to process cricket 
match data for predictive analytics. I also integrated LangChain and OpenAI APIs to generate 
contextual insights. I'm comfortable with FastAPI for building APIs, PyTorch for ML models, and 
I've worked extensively with data processing libraries like NumPy and scikit-learn."""
    
    # Evaluate the answer
    print("\nEvaluating answer...")
    evaluation = evaluator.evaluate_answer(question, good_answer)
    display_evaluation(evaluation)
    
    # Generate report for multiple evaluations
    print("\n\nGenerating interview report...")
    
    evaluations = [evaluation]  # In real scenario, would have multiple
    
    report = evaluator.generate_report(
        candidate_name="Alex",
        role="Senior Python Developer",
        evaluations=evaluations
    )
    
    display_report(report)
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Integrate with Interviewer Agent")
    print("2. Evaluate multiple answers")
    print("3. Generate comprehensive reports")
    print("4. Build complete interview pipeline")
"""
Question Generator Agent - InterviewAce Ai

WHAT THIS DOES:
- Generates personalized interview questions based on job description
- Creates questions at different difficulty levels
- Categorizes questions (technical, behavioral, system design)
- Returns structured output for the interview system
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from backend.config.settings import get_settings, setup_langsmith

# ========== STEP 1: DEFINE DATA STRUCTURES ==========

"""
Pydantic models define the STRUCTURE of our output
This ensures GPT returns data in a format we can use programmatically

WHY: instead of parsing free text , we get structures JSON
"""

class InterviewQuestion(BaseModel):
    """
    Single Interview Question with metadata

    FIELDS:
    - Question : The actual question text
    - Category : Type of question (technical, behavioral, system_design)
    - Difficulty : Easy/ Medium/ Hard
    - Expected_topics : What the asnwer should cover
    - Follow_up : Optional follow-up question
    """

    question: str =Field(description = "The interview question text")
    category: str = Field(description = "Question category: technical, behavioral or system_design")
    difficulty: str = Field(description = "Difficulty level: easy, medium , or hard")
    expected_topics: List[str] = Field(description = "Key topics the answer should cover" )
    follow_up: Optional[str] = Field(default = None, description = "Optional follow-up question")

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {"question": "Explain the difference between REST and GraphQL APIs",
                        "category": "technical",
                        "difficulty": "medium",
                        "expected_topics": ["REST principles", "GraphQL query language", "Use cases", "Trade-offs"],
                        "follow_up": "When would you choose GraphQL over REST?"
                        
                    }
                }
    )
        

class QuestionSet(BaseModel):
    """
    Complete set of interview questions
    
    FIELDS:
    - role: The job role these questions are for
    - questions: List of InterviewQuestion objects
    - total_count: Number of questions generated
    """

    role: str = Field(description = "The Job role/position")
    questions: List[InterviewQuestion] = Field(description = "List of interview questions")
    total_count: int = Field(description= "total number of questions")

    
# ========== STEP 2: CREATE THE AGENT CLASS ==========

"""
Object-oriented design makes the agent reusable and testable.
"""

class QuestionGeneratorAgent:
    """
    Agent that generates tailored interview questions
    
    ARCHITECTURE:
    1. Initialize with LLM and settings
    2. Build complex prompt template
    3. Parse structured output
    4. Generate questions on demand
    
    """

    def __init__(self):
        """Initialize the agent with settings and LLM"""

        self.settings = get_settings()

        # Create LLM specialize for this agent
        self.llm = ChatOpenAI(
            model = self.settings.openai_model,
            temperature = 0.8,
            openai_api_key = self.settings.openai_api_key,
            max_tokens = 2000
        )

        # Create Output parser

        self.output_parser = PydanticOutputParser(pydantic_object = QuestionSet)

        # Build the Prompt template

        self.prompt = self._create_prompt_template()

        # Create the chain

        self.chain = self.prompt | self.llm | self.output_parser

        print("Question Generator Agent initalized")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create a sophisticated prompt template for question generation
        
        Techniques Used:
        1. Role Assignment (You are an expert interviewer...)
        2. Clear Instructions
        3. Few-Shot Examples
        4. Structured output format
        5. Constraints (Number of questions, difficulty, distribution)
        """

        # Get format instructions from parser

        format_instructions = self.output_parser.get_format_instructions()

        # Complex prompt with multiple sections

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical interviewer and hiring manager with 15+ years of experience. 
Your specialty is creating thoughtful, relevant interview questions that assess both technical skills and cultural fit.

YOUR TASK:
Generate a comprehensive set of interview questions for a specific role based on the job description provided.

REQUIREMENTS:
1. Create {num_questions} questions total
2. Mix of categories:
   - Technical: 60% (coding, system design, technical concepts)
   - Behavioral: 30% (past experience, problem-solving, teamwork)
   - System Design: 10% (architecture, scalability, trade-offs)
3. Difficulty distribution:
   - Easy: 30% (screening, fundamentals)
   - Medium: 50% (core competencies)
   - Hard: 20% (senior-level, advanced topics)
4. Each question should:
   - Be clear and unambiguous
   - Test specific skills from the job description
   - Include expected topics in the answer
   - Have optional follow-up questions for deeper assessment

{format_instructions}

QUALITY GUIDELINES:
- Avoid generic questions that could apply to any role
- Focus on practical, real-world scenarios
- Include both "explain" and "implement" style questions
- Consider the seniority level mentioned in the job description"""),
            
            ("user", """Generate interview questions for this role:

JOB DESCRIPTION:
{job_description}

ADDITIONAL CONTEXT:
{context}

NUMBER OF QUESTIONS: {num_questions}

Generate the questions now, ensuring they are highly relevant to this specific role.""")
        ])

        return prompt_template
    

    def generate_questions(
            self,
            job_description: str,
            num_questions:int = 10,
            context: str = ""
    ) -> QuestionSet:
        
        """
        Generate interview questions for a given job description
        
        PARAMETERS:
        - job_description: The job posting or role description
        - num_questions: how many questions to generate (default : 10)
        - context: Additional context (resume, company info, etc)
        
        
        RETURNS:
        - QuestionSet: Structured set of interview questions
        
        EXAMPLE:
        >>> agent = QuestionGeneratorAgent()
        >>> questions = agent.generate_questions(
                job_description="Senior Python Developer with AI/ML experience"
                num_questions=5
            )
        >>> for q in questions.questions:
                print(f"{q.difficulty.upper()}: {q.question}")
                
        """

        try:
            print(f"\n🤔 Generating {num_questions} interview questions...")

            # Invoke the chain

            result = self.chain.invoke({
                "job_description": job_description,
                "num_questions": num_questions,
                "context": context or "No additional context provided.",
                "format_instructions": self.output_parser.get_format_instructions()
            })

            print(f"Generated {result.total_count} questions successfully")

            return result

        except Exception as e:
            print(f"Error generating questions: {e}")
            raise

    def generating_questions_by_resume(
            self,
            job_description: str,
            resume_text: str,
            num_questions:int = 10
    ) -> QuestionSet:
        """
        Generate persomalized questions based on job description AND resume
        
        This creates questions that:
        - Test skills mentioned in the resume
        - Probe gaps between resume and job requirements
        - Assess claims made in the resume
        
        PARAMETERS:
        - job_description: The job posting
        - resume_text: Candidate's resume content
        - num_questions: How many questions to generate
        
        RETURNS:
        - QuestionSet: Personalized interview questions
        """

        context = f"""
CANDIDATE'S RESUME: 
{resume_text}

PERSONALIZATION INSTRUCTIONS:
- Reference specific projects/experience from the resume
- Test claims made in the resume (e.g., "You mentioned experience with X...")
- Identify skill gaps between job requirements and resume
- Create questions that assess the depth of stated experience
"""
        return self.generate_questions(
            job_description = job_description,
            num_questions = num_questions,
            context = context
        )
    

# ========== STEP 3: HELPER FUNCTIONS ==========


def display_questions(question_set = QuestionSet):
    """
    Pretty print the generated questions
    
    USAGE:
    >>> questions = agent.generate_questions(job_desc)
    >>> display_questions(questions)
    """

    print("\n" + "="*80)
    print(f"Interview questions for : {question_set.role}")
    print(f"Total questions: {question_set.total_count}")
    print("="*80)

    # Group by category

    by_category = {}
    for q in question_set.questions:
        if q.category not in by_category:
            by_category[q.category] = []
        by_category[q.category].append(q)


    # Display each category
    for category, questions in by_category.items():
        print(f"\n{'='*80}")
        print(f"{category.upper().replace('_', ' ')}")
        print('='*80)

        for i, q in enumerate(questions, 1):
            difficulty_label = q.difficulty.upper()
            print(f"\nQuestion {i} - Difficulty: {difficulty_label}")
            print(f"{q.question}")

            if q.expected_topics:
                print(f"\n Expected Topics:")
                for topic in q.expected_topics:
                    print(f"    {topic}")

            if q.follow_up:
                print(f"\n Follow-up: {q.follow_up}")

            print("\n"+"-"*80)




# ========== STEP 4: TEST/DEMO CODE ==========
if __name__ == "__main__" :
    """
    Test the Question Generator Agent

    RUN:
    python -m backend.agents.question_generator.py
    """

    # Setup
    setup_langsmith()

    print("\n"+"="*80)
    print("QUESTION GENERATOR AGENT - DEMO")
    print("="*80)

    # Initialize Agent
    agent = QuestionGeneratorAgent()

    # Example Job Description
    job_description = """
Senior GenAI Engineer

We're looking for an experienced engineer to join our AI team. You'll build production 
AI systems using LLMs, LangChain, and vector databases.

Required Skills:
- 3+ years Python development
- Experience with LLM APIs (OpenAI, Anthropic)
- LangChain, LangGraph for agentic workflows
- Vector databases (Pinecone, ChromaDB, FAISS)
- RAG (Retrieval Augmented Generation)
- Prompt engineering
- Production ML deployment

Nice to have:
- FastAPI, async Python
- Fine-tuning experience
- Cloud platforms (AWS/GCP)

You'll be responsible for:
- Designing and implementing AI agents
- Building RAG pipelines
- Optimizing LLM performance and cost
- Mentoring junior engineers
"""

    try:
        # Generate Questions
        questions = agent.generate_questions(
            job_description= job_description,
            num_questions= 8 
        )

        # Display Results
        display_questions(questions)
        print("\n" + "="*80)
        print("DEMO COMPLETE!")
        print("="*80)
        print("\n Next steps:")
        print("   1. Try with your own job descriptions")
        print("   2. Adjust num_questions parameter")
        print("   3. Add resume context for personalization")
        print("   4. Integrate with interview system")

    except Exception as e:
        print(f"\n Demo failed: {e}")
        print("\nCheck:")
        print("  1. OpenAI API key is valid")
        print("  2. Internet connection")
        print("  3. LangSmith is configured (optional)")










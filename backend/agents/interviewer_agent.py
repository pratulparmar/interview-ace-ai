"""
Interviewer Agent - InterviewAce AI

Purpose:
     Conduct Interactive interviews with candidates.
     Asks questions, Listen to answers and provides follow-ups
     
Key Features:
    - Natural conversation flow
    - Context aware follow-ups
    - Maintains interview state
    - Provide feedback during interview
    
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from backend.config.settings import get_settings, setup_langsmith
from backend.agents.question_generator import InterviewQuestion


# ==================================== DATA MODELS ==================================

class InterviewResponse(BaseModel):
    """ Model for candidate's response to a question"""
    question : str = Field(description = "The Question that was asked")
    answer : str =Field(description = "Candidate's answer")
    follow_up_needed : bool = Field(description = "Whether a follow-up is needed")

class InterviewSession(BaseModel):
    """ Tracks the state of an ongoing interview """
    candidate_name : str
    role : str
    questions : List[InterviewQuestion]
    current_question_index : int = 0
    responses : List[InterviewResponse] = []
    is_complete : bool = False




#=================================== INTERVIEWER AGENT ==============================

class InterviewerAgent:
    """
    Conducts interactive interviews with candidates.
    
    This agent:
    - Asks questions in a natural, conversational way
    - Listens to candidate responses
    - Asks intelligent follow-up questions
    - Maintains conversation context
    - Provides a human-like interview experience
    
    Architecture:
    - Uses ChatOpenAI with conversation memory
    - Maintains chat history for context
    - Dynamically generates follow-ups based on answers
    """

    def __init__(self):
        """ Initialize the interviewer agent """
        self.settings = get_settings()

        # LLM configuration for conversational interview

        self.llm = ChatOpenAI(
            model = self.settings.openai_model,
            temperature = 0.7 , # Balanced for natural conversation
            openai_api_key = self.settings.openai_api_key
        )

        # Coverstational history
        self.chat_history : List[Dict] = []

        print("Interviewer Agent Initialized")


    def start_interview(
            self,
            candidate_name : str,
            role : str,
            questions : List[InterviewQuestion]
    ) -> InterviewSession:
        """
        Start a new interview session.
        
        Args:
            candidate_name: Name of the candidate
            role: Job role being interviewed for
            questions: List of questions to ask (from Question Generator)
        
        Returns:
            InterviewSession: New interview session object
            
        """

        session = InterviewSession(
            candidate_name = candidate_name,
            role = role,
            questions = questions
        )


        # Clear chat history for new interview
        self.chat_history = []

        # Welcome message

        welcome = self._generate_welcome_message(candidate_name, role)
        print(f"\n{welcome}\n")

        return session
    

    def _generate_welcome_message(
            self,
            candidate_name : str,
            role : str
    ) -> str:
        """ Generate a personalized welcome message"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """ You are a friendly , professional interviewer.
generate a warm welcome message for the candidate.
Keep it brief (2-3 sentences), professional but friendly."""),
            ("user", "Candidate name: {name}, Role: {role}")
        ])

        chain = prompt | self.llm
        result = chain.invoke({"name":candidate_name, "role": role})
        return result.content


    def ask_question(self, session: InterviewSession) -> str :
        """
        Ask the next question in the interview.
        
        Args:
            session: Current interview session

        Returns: 
                str: The question text with natural phrasing
        """

        if session.current_question_index >= len(session.questions):
            session.is_complete = True
            return "That completes all the questions. Thank you!"
        
        # Get current questions

        current_q = session.questions[session.current_question_index]

        # Generate natural question phrasing

        question_text = self._phrase_question_naturally(
            current_q,
            session.current_question_index + 1,
            len(session.questions)
        )

        # Add to chat history
        self.chat_history.append({
            "role" : "assistant",
            "content" : question_text
        })

        return question_text
    

    def _phrase_question_naturally(
            self,
            question : InterviewQuestion,
            question_num: int,
            total_questions: int
    ) -> str:
        """
        Rephrase question in natural, conversational way.
        
        Instead of just reading the question verbatim,
        add natural interview phrasing.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are conducting an interview. 
Rephrase the following question in a natural, conversational way.
Add a brief intro if appropriate (e.g., "Great, let's move on to...").
Keep it professional but friendly.
Don't change the core question content."""),
            ("user","""Question {num} of {total}
Category: {category}
Difficulty: {difficulty}
Question: {question}

Rephrase this naturally.""")
        ])

        chain = prompt | self.llm
        result = chain.invoke({
            "num": question_num,
            "total": total_questions,
            "category": question.category,
            "difficulty": question.difficulty,
            "question": question.question
        })

        return result.content
    
    def receive_answer(
            self,
            session: InterviewSession,
            answer: str
    ) -> Optional[str]:
        """
        Process candidate's answer and decide on follow-up.
        
        Args:
            session: Current interview session
            answer: Candidate's answer text
        
        Returns:
            Optional[str]: Follow-up question if needed, None otherwise
        """

        current_q = session.questions[session.current_question_index]

        # Add answer to chat history
        self.chat_history.append({
            "role": "user",
            "content": answer
        })

        # Analyze answer and decide on follow-up
        follow_up = self._generate_follow_up(current_q, answer)

        # Save response
        response = InterviewResponse(
            question = current_q.question,
            answer = answer,
            follow_up_needed = (follow_up is not None)
        )

        session.responses.append(response)

        if follow_up:
            # Add follow-up to chat
            self.chat_history.append({
                "role" : "assistant",
                "content" : follow_up
            })
            return follow_up
        else:
            # Move to the next question
            session.current_question_index += 1
            return None
        
    def _generate_follow_up(
            self,
            question: InterviewQuestion,
            answer: str
    ) -> Optional[str]:
        """
        Decide if follow-up is needed and generate it.
        
        Analyze the answer quality and depth.
        If answer is incomplete or unclear, generate follow-up.
        """
        # Check if predifined follow_up exists
        if question.follow_up:
            # Decide if we should ask it based on answer quality
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an interviewer evaluating an answer.
Decide if a follow-up question is needed.

Answer "YES" if:
- Answer is too brief or vague
- Answer doesn't cover key topics
- You want to probe deeper

Answer "NO" if:
- Answer is comprehensive and clear
- All key points were covered

Respond with just "YES" or "NO"."""),
                ("user", """Question: {question}
Expected topics: {topics}
Candidate's answer: {answer}

Should I ask the follow-up question?""")
            ])

            chain = prompt | self.llm
            result = chain.invoke({
                "question": question.question,
                "topics": ",".join(question.expected_topics),
                "answer": answer
            })

            decision = result.content.strip().upper()

            if decision =="YES":
                return f"That's interesting.{question.follow_up}"
            
        return None
    
    def provide_feedback(self, session: InterviewSession) -> str:
        """
        Provide brief feedback after interview completion.
        
        Args:
            session: Completed interview session
        
        Returns:
            str: Feedback message
        """
        prompt = ChatPromptTemplate([("system", """You are a professional interviewer providing brief feedback.
Based on the interview conversation, provide:
1. A positive observation
2. One area that stood out
3. A warm closing

Keep it brief (3-4 sentences) and encouraging."""),
            ("user", "Review this interview conversation:\n\n{history}")
        ])

        # Format chat history
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in self.chat_history
        ])

        chain = prompt | self.llm

        result = chain.invoke({
            "history": history_text
        })

        return result.content
    


# =========================== INTERACTIVE INTERVIEW FUNCTION ========================

def conduct_interview_interactive(
    candidate_name: str,
    role: str,
    questions: List[InterviewQuestion]
):
    """
    Conduct an interactive interview in the terminal
    
    This is a simple CLI interface for testing the interviewer agent.
    
    Args:
        candidate_name: Candidate's name
        role: Job Role
        questions: List of questions to ask
    """

    interviewer = InterviewerAgent()
    session = interviewer.start_interview(candidate_name, role, questions)

    print("="*80)
    print("INTERVIEW SESSION STARTED")
    print("="*80)
    print("(Type your answers and press Enter. Type 'quit' to end early.)\n")

    while not session.is_complete:
        # Ask question
        question_text = interviewer.ask_question(session)
        print(f"\nInterviewer: {question_text}\n")

        if session.is_complete:
            break

        # Get candidate's answer
        print("Your answer:", end="")
        answer = input().strip()

        if answer.lower() == 'quit':
            print("\nInterview ended early.")
            break

        if not answer:
            print("Please provide an answer.\n")
            continue

        # Process answer and check for follow-up
        follow_up = interviewer.receive_answer(session, answer)

        if follow_up:
            print(f"\nInterviewer: {follow_up}\n" )
            print("Your answer: ", end="")
            follow_up_answer = input().strip()

            if follow_up_answer:
                interviewer.chat_history.append({
                    "role": "user",
                    "content": follow_up_answer
                })

                # Move to next question after follow-up
                session.current_question_index += 1

    # Interview complete
    print("\n" + "="*80)
    print("INTERVIEW COMPLETE")
    print("="*80)
    
    # Provide feedback
    feedback = interviewer.provide_feedback(session)
    print(f"\nInterviewer: {feedback}\n")
    
    # Summary
    print("\n" + "="*80)
    print("INTERVIEW SUMMARY")
    print("="*80)
    print(f"Candidate: {session.candidate_name}")
    print(f"Role: {session.role}")
    print(f"Questions asked: {len(session.responses)}")
    print(f"Follow-ups asked: {sum(1 for r in session.responses if r.follow_up_needed)}")


# =================================== DEMO CODE =======================================

if __name__ == "__main__":
    """
    Demo the Interviewer Agent
    
    Run: python -m backend.agents.interviewer_agent
    """
    setup_langsmith()
    
    print("\n" + "="*80)
    print("INTERVIEWER AGENT - DEMO")
    print("="*80)
    
    # Sample questions (normally from Question Generator)
    sample_questions = [
        InterviewQuestion(
            question="Can you explain your experience with Python?",
            category="technical",
            difficulty="easy",
            expected_topics=["Python basics", "Projects", "Experience level"],
            follow_up="What Python frameworks have you used?"
        ),
        InterviewQuestion(
            question="Describe a challenging project you worked on.",
            category="behavioral",
            difficulty="medium",
            expected_topics=["Problem description", "Solution", "Outcome"],
            follow_up="What would you do differently if you could redo it?"
        ),
        InterviewQuestion(
            question="How would you design a scalable API?",
            category="system_design",
            difficulty="hard",
            expected_topics=["Architecture", "Scaling", "Trade-offs"],
            follow_up="How would you handle authentication?"
        )
    ]
    
    # Conduct interactive interview
    conduct_interview_interactive(
        candidate_name="Alex",
        role="Senior Python Developer",
        questions=sample_questions[:2]  # Just 2 questions for demo
    )
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)




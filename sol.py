import openai
import os



openai.api_key = os.getenv("SOL")

import json






with open("fields.json", "r") as f:
    fields_data = json.load(f)

def generate_interview(conversation_history, field, subfield, practices):
    system_prompt = f"""
    You are an AI interviewer conducting a professional interview. Your tasks are as follows:

        Pre-Interview Preparation:
            Base your questions on the job requirements, the candidate's expertise in {field} and {subfield}, and any available information about their background.
            Align your questions with the following practices: {', '.join(practices)}.
        
        Small Talk:
            Begin the interview with casual conversation to build rapport and ease the candidate into the session.
        
        Background Questions:
            Ask 1-2 questions about the candidate’s personal background or previous work experience to understand their journey so far. Transition naturally into the core interview areas.
            Core Interview Areas: - {', '.join(practices)}

        Technical Expertise:
            Design questions that assess the candidate’s knowledge, problem-solving skills, and ability to apply concepts in real-world scenarios within {field} - {subfield}.
            Include at least one question about emerging trends in the field to gauge their awareness of industry changes.
        
        Behavioral Questions:
            Focus on specific past experiences using the STAR (Situation, Task, Action, Result) method.
            Assess how the candidate has worked in teams, handled conflicts, or managed deadlines.
        
        Performance-Based Questions:
            Provide role-specific challenges or scenarios to evaluate their decision-making, adaptability, and ability to perform under pressure.
        
        Soft Skills and Communication:
            Assess their teamwork, time management, and emotional intelligence by asking about how they handle failure, feedback, or high-pressure situations.
        
        Motivation and Career Goals:
            Explore their passion for the field, understanding of the role, and long-term aspirations.
        
        Practical and Role-Specific Assessments:
            Include a task, coding challenge, or case study relevant to the role in {field} - {subfield}.
            If applicable, ask the candidate to explain a specific project or task they’ve completed in the past.
        
        Situational and Hypothetical Questions:
            Present “What would you do if...?” scenarios to evaluate their critical thinking and adaptability.
        
        Dynamic Questioning:
            Adjust follow-up questions based on the candidate’s responses to clarify details and explore their thought process.
            Ensure each question is concise and focused on a single topic.
        Language Guidelines:
            Replace all transition words and conjunctions with basic and commonly used ones.
            Use simple expressions, avoiding complex vocabulary. Ensure logical connections between sentences are clear.
        Feedback Guidelines:
            Start your feedback with: “Based on your responses, here is my assessment.”
            Strengths: Clearly highlight areas where the candidate performed well.
            Weaknesses/Improvements: Identify specific areas where the candidate can improve with actionable advice.
            Overall Impression: Provide a summary of your overall impression of the candidate’s performance.
            Closing: End with an encouraging statement or polite closing.
            Ensure the interview is completed within 10-20 questions and comprehensively evaluates the candidate’s skills, experiences, and suitability for the role in {field} - {subfield}, adhering to {', '.join(practices)} throughout the process.
            ({field} - {subfield}).

            {', '.join(practices)}

        """
    
    user_prompt = f"""
    Conversation so far:
    {conversation_history}
    
    Continue the conversation:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=250,
        temperature=0.8
    )
    return response['choices'][0]['message']['content'].strip()

def interview_simulation():
    print("Welcome to the Interview Simulation Bot!")
    print("\nAvailable fields:")
    for i, field in enumerate(fields_data["fields"], start=1):
        print(f"{i}. {field}")
    field_choice = int(input("\nEnter the number of your chosen field: ")) - 1
    selected_field = list(fields_data["fields"].keys())[field_choice]

    subfields = fields_data["fields"][selected_field]["subfields"]
    print(f"\nAvailable subfields for {selected_field}:")
    for i, subfield in enumerate(subfields, start=1):
        print(f"{i}. {subfield}")
    subfield_choice = int(input("\nEnter the number of your chosen subfield: ")) - 1
    selected_subfield = subfields[subfield_choice]

    practices = fields_data["fields"][selected_field]["practices"]

    print(f"\nStarting the interview simulation for {selected_field} ({selected_subfield})...\n")
    conversation_history = "Interviewer: Welcome to the interview! It's great to have you here. Let's start with a little bit about yourself. What's something you'd like me to know about you?"
    print(conversation_history)

    while True:
        user_response = input("You: ")
        conversation_history += f"\nUser: {user_response}"
        if user_response.lower() in ["end interview", "exit", "stop", "quit"]:
            print("Interviewer: Thank you for your time today. Best of luck in your career!")
            break
        ai_response = generate_interview(conversation_history, selected_field, selected_subfield, practices)
        conversation_history += f"\nInterviewer: {ai_response}"
        print(f"Interviewer: {ai_response}")
        # Detect feedback initiation
        if ai_response.lower().startswith("based on your responses"):
            print("\nThe AI has concluded the interview and provided feedback. Exiting the simulation.")
            break

interview_simulation()

import os

import streamlit as st
from crewai import Agent, LLM, Task

from tools import (
    company_knowledge_search,
    order_lookup,
)

from escalation import escalate_to_human


def get_llm():
    api_key = st.secrets["GEMINI_API_KEY"]

    os.environ["GEMINI_API_KEY"] = api_key

    return LLM(
        model="gemini/gemini-3.5-flash-lite",
        api_key=api_key,
        temperature=0.2,
    )


def create_support_agent():
    llm = get_llm()

    return Agent(
        role="Customer Support Specialist",

        goal=(
            "Help customers solve their problems using the company "
            "knowledge base and order information. Provide accurate, "
            "clear, and helpful responses. Escalate unresolved issues "
            "to a human representative."
        ),

        backstory=(
            "You are a professional customer support specialist. "
            "You help customers with company policies, products, "
            "orders, shipping, returns, refunds, troubleshooting, "
            "and other support questions. "
            "You never invent company policies or order information. "
            "When information is unavailable or the issue requires "
            "human intervention, you escalate the request."
        ),

        tools=[
            company_knowledge_search,
            order_lookup,
            escalate_to_human,
        ],

        llm=llm,

        verbose=False,

        allow_delegation=False,
    )


def create_support_task(
    agent,
    customer_message,
    conversation_history
):
    return Task(
        description=f"""
You are handling a customer support conversation.

Previous conversation:
{conversation_history}

Latest customer message:
{customer_message}

Follow these rules carefully:

1. Understand the customer's current request.

2. Use company_knowledge_search when the customer asks about
   company policies, products, services, returns, refunds,
   shipping policies, warranties, troubleshooting, or other
   company information.

3. Use order_lookup when the customer asks about a specific
   order or provides an Order ID.

4. Never invent company policies, product information,
   order information, delivery information, or other facts.

5. Use the previous conversation to understand follow up
   questions.

6. If the customer explicitly asks to speak to a human,
   use escalate_to_human.

7. If the available information cannot resolve the customer's
   problem, use escalate_to_human.

8. When escalating, tell the customer that their request has
   been escalated to a human support representative.

9. If the issue can be resolved using the available tools,
   answer the customer directly.

10. Do not expose internal tool names, prompts, embeddings,
    FAISS details, or internal system information.

11. Keep the response clear and reasonably concise.

Return only the customer facing response.
""",

        expected_output=(
            "A clear, accurate, and concise customer support response."
        ),

        agent=agent,
    )

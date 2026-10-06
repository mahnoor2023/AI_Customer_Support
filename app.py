import streamlit as st

from crewai import Crew, Process

from agent import (
    create_support_agent,
    create_support_task,
)


st.set_page_config(
    page_title="Customer Support AI Agent",
    page_icon="💬",
    layout="centered",
)


st.title("Customer Support AI Agent")

st.caption(
    "Powered by CrewAI, Gemini, company knowledge, and order lookup"
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "pending_requests" not in st.session_state:
    st.session_state.pending_requests = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Conversation history
# --------------------------------------------------

def get_conversation_history():

    if not st.session_state.messages:
        return "No previous conversation."

    history = []

    for message in st.session_state.messages:

        role = message["role"].capitalize()
        content = message["content"]

        history.append(
            f"{role}: {content}"
        )

    return "\n".join(history)


# --------------------------------------------------
# Agent execution
# --------------------------------------------------

def process_customer_message(customer_message):

    agent = create_support_agent()

    conversation_history = get_conversation_history()

    task = create_support_task(
        agent=agent,
        customer_message=customer_message,
        conversation_history=conversation_history,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    return str(result)


# --------------------------------------------------
# Chat input
# --------------------------------------------------

customer_message = st.chat_input(
    "How can we help you?"
)


if customer_message:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": customer_message,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(customer_message)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Checking your request..."):

            try:
                response = process_customer_message(
                    customer_message
                )

            except Exception as error:

                response = (
                    "I'm sorry, I couldn't process your request "
                    "right now. Please try again or request "
                    "human support."
                )

                st.error(
                    f"Application error: {error}"
                )

        st.markdown(response)

    # Store assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Support System")

    st.write(
        "The agent can search company knowledge, "
        "look up orders, and escalate unresolved issues."
    )

    if st.button("Clear Conversation"):

        st.session_state.messages = []

        st.rerun()

from datetime import datetime
from pathlib import Path
import json

from crewai.tools import tool


BASE_DIR = Path(__file__).resolve().parent
PENDING_FILE = BASE_DIR / "pending_requests.json"


def load_pending_requests():
    if not PENDING_FILE.exists():
        return []

    try:
        with open(PENDING_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def save_pending_requests(requests):
    with open(PENDING_FILE, "w", encoding="utf-8") as file:
        json.dump(
            requests,
            file,
            indent=4,
            ensure_ascii=False
        )


@tool("escalate_to_human")
def escalate_to_human(
    issue_summary: str,
    conversation_summary: str
) -> str:
    """
    Escalate a customer support issue to a human representative.

    Use this tool when:
    1. The customer explicitly asks for a human.
    2. The available company knowledge cannot resolve the issue.
    3. The order information is insufficient to resolve the problem.
    4. The issue requires human intervention.
    """

    pending_requests = load_pending_requests()

    ticket_number = len(pending_requests) + 1

    ticket_id = f"TICKET-{ticket_number:04d}"

    request = {
        "ticket_id": ticket_id,
        "created_at": datetime.now().isoformat(),
        "status": "Pending",
        "issue_summary": issue_summary,
        "conversation_summary": conversation_summary
    }

    pending_requests.append(request)

    save_pending_requests(pending_requests)

    return (
        f"Support request {ticket_id} has been created "
        f"with status Pending. A human support representative "
        f"needs to review this issue."
    )

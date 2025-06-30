"""
Conversation Handler for OpenPerturbation Agents

Manages conversations and interactions between users and AI agents.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ConversationHandler:
    """Handle conversations with OpenPerturbation agents."""

    def __init__(self, conversation_dir: str = "conversations"):
        """Initialize conversation handler."""
        self.conversation_dir = Path(conversation_dir)
        self.conversation_dir.mkdir(exist_ok=True)
        self.active_conversations: Dict[str, Dict[str, Any]] = {}

    def start_conversation(self, user_id: str, agent_type: str = "general") -> str:
        """Start a new conversation."""
        import uuid

        conversation_id = str(uuid.uuid4())

        conversation = {
            "id": conversation_id,
            "user_id": user_id,
            "agent_type": agent_type,
            "started_at": datetime.now().isoformat(),
            "messages": [],
            "context": {},
            "status": "active",
        }

        self.active_conversations[conversation_id] = conversation
        self._save_conversation(conversation_id)

        logger.info(f"Started conversation {conversation_id} for user {user_id}")
        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a message to the conversation."""

        if conversation_id not in self.active_conversations:
            logger.error(f"Conversation {conversation_id} not found")
            return False

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self.active_conversations[conversation_id]["messages"].append(message)
        self._save_conversation(conversation_id)

        return True

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history."""

        if conversation_id not in self.active_conversations:
            # Try to load from disk
            if not self._load_conversation(conversation_id):
                return []

        return self.active_conversations[conversation_id]["messages"]

    def update_context(self, conversation_id: str, context_updates: Dict[str, Any]) -> bool:
        """Update conversation context."""

        if conversation_id not in self.active_conversations:
            return False

        self.active_conversations[conversation_id]["context"].update(context_updates)
        self._save_conversation(conversation_id)

        return True

    def end_conversation(self, conversation_id: str) -> bool:
        """End a conversation."""

        if conversation_id not in self.active_conversations:
            return False

        self.active_conversations[conversation_id]["status"] = "ended"
        self.active_conversations[conversation_id]["ended_at"] = datetime.now().isoformat()

        self._save_conversation(conversation_id)

        # Remove from active conversations
        del self.active_conversations[conversation_id]

        logger.info(f"Ended conversation {conversation_id}")
        return True

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get a summary of the conversation."""

        if conversation_id not in self.active_conversations:
            if not self._load_conversation(conversation_id):
                return {}

        conversation = self.active_conversations[conversation_id]

        summary = {
            "id": conversation_id,
            "user_id": conversation["user_id"],
            "agent_type": conversation["agent_type"],
            "started_at": conversation["started_at"],
            "message_count": len(conversation["messages"]),
            "status": conversation["status"],
            "last_activity": conversation["messages"][-1]["timestamp"]
            if conversation["messages"]
            else None,
        }

        return summary

    def list_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """List all conversations for a user."""

        user_conversations = []

        # Check active conversations
        for conv_id, conv in self.active_conversations.items():
            if conv["user_id"] == user_id:
                user_conversations.append(self.get_conversation_summary(conv_id))

        # Check saved conversations
        for conv_file in self.conversation_dir.glob("*.json"):
            conv_id = conv_file.stem
            if conv_id not in self.active_conversations:
                if self._load_conversation(conv_id):
                    conv = self.active_conversations[conv_id]
                    if conv["user_id"] == user_id:
                        user_conversations.append(self.get_conversation_summary(conv_id))

        return sorted(user_conversations, key=lambda x: x["started_at"], reverse=True)

    def _save_conversation(self, conversation_id: str):
        """Save conversation to disk."""

        if conversation_id not in self.active_conversations:
            return

        conversation_file = self.conversation_dir / f"{conversation_id}.json"

        try:
            with open(conversation_file, "w") as f:
                json.dump(self.active_conversations[conversation_id], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversation {conversation_id}: {e}")

    def _load_conversation(self, conversation_id: str) -> bool:
        """Load conversation from disk."""

        conversation_file = self.conversation_dir / f"{conversation_id}.json"

        if not conversation_file.exists():
            return False

        try:
            with open(conversation_file, "r") as f:
                conversation = json.load(f)

            self.active_conversations[conversation_id] = conversation
            return True

        except Exception as e:
            logger.error(f"Failed to load conversation {conversation_id}: {e}")
            return False

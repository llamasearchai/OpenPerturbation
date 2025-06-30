"""
OpenAI Agents SDK Integration Tests
Author: Nik Jois <nikjois@llamasearch.ai>

This module tests the integration between OpenPerturbation and OpenAI's Agents SDK,
ensuring seamless AI-powered analysis and recommendations for perturbation biology experiments.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any, Optional

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None

from src.agents.openai_agent import OpenPerturbationAgent
from src.agents.conversation_handler import ConversationHandler
from src.agents.agent_tools import AgentTools


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock(spec=OpenAI)
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    return mock_client


@pytest.fixture
def sample_perturbation_data():
    """Sample perturbation data for testing."""
    return {
        "experiment_id": "exp_001",
        "perturbation_type": "chemical",
        "compound_id": "CHEMBL12345",
        "concentration": "10uM",
        "treatment_time": "24h",
        "cell_line": "HeLa",
        "readouts": {
            "imaging": {
                "num_cells": 1250,
                "viability": 0.87,
                "morphology_score": 0.62
            },
            "genomics": {
                "num_genes_perturbed": 2341,
                "top_pathways": ["cell_cycle", "apoptosis", "DNA_repair"]
            }
        }
    }


@pytest.fixture
def sample_analysis_request():
    """Sample analysis request for AI agent."""
    return {
        "query": "Analyze the effects of this compound on cellular viability and suggest follow-up experiments",
        "context": "drug_discovery",
        "data_types": ["imaging", "genomics"],
        "analysis_depth": "comprehensive"
    }


class TestOpenPerturbationAgent:
    """Test suite for OpenPerturbation AI Agent functionality."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not available")
    def test_agent_initialization(self, mock_openai_client):
        """Test proper initialization of OpenPerturbation AI agent."""
        with patch('src.agents.openai_agent.OpenAI', return_value=mock_openai_client):
            agent = OpenPerturbationAgent(
                api_key="test-key",
                model="gpt-4",
                temperature=0.7
            )
            
            assert agent.client == mock_openai_client
            assert agent.model == "gpt-4"
            assert agent.temperature == 0.7
            assert agent.conversation_history == []

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not available")
    @pytest.mark.asyncio
    async def test_analyze_perturbation_data(self, mock_openai_client, sample_perturbation_data):
        """Test AI analysis of perturbation experiment data."""
        # Mock response from OpenAI
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps({
            "analysis": {
                "viability_assessment": "Moderate cytotoxicity observed",
                "pathway_insights": ["Cell cycle disruption", "Apoptosis activation"],
                "morphology_changes": "Significant cellular shape changes detected"
            },
            "recommendations": [
                "Test lower concentrations (1uM, 5uM)",
                "Extend time-course analysis to 48h",
                "Add cell cycle markers for detailed analysis"
            ],
            "confidence": 0.85
        })
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        with patch('src.agents.openai_agent.OpenAI', return_value=mock_openai_client):
            agent = OpenPerturbationAgent(api_key="test-key")
            
            result = await agent.analyze_perturbation_data(sample_perturbation_data)
            
            assert "analysis" in result
            assert "recommendations" in result
            assert "confidence" in result
            assert result["confidence"] >= 0.8

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not available")
    @pytest.mark.asyncio
    async def test_suggest_follow_up_experiments(self, mock_openai_client, sample_perturbation_data):
        """Test AI-powered follow-up experiment suggestions."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = json.dumps({
            "experiments": [
                {
                    "type": "dose_response",
                    "parameters": {
                        "concentrations": ["0.1uM", "1uM", "5uM", "10uM", "50uM"],
                        "timepoints": ["6h", "12h", "24h", "48h"]
                    },
                    "rationale": "Determine IC50 and temporal dynamics"
                },
                {
                    "type": "pathway_analysis",
                    "parameters": {
                        "assays": ["cell_cycle_flow", "apoptosis_annexin", "DNA_damage_comet"],
                        "timepoint": "24h"
                    },
                    "rationale": "Confirm pathway involvement"
                }
            ],
            "priority": "high"
        })
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        with patch('src.agents.openai_agent.OpenAI', return_value=mock_openai_client):
            agent = OpenPerturbationAgent(api_key="test-key")
            
            suggestions = await agent.suggest_follow_up_experiments(sample_perturbation_data)
            
            assert "experiments" in suggestions
            assert len(suggestions["experiments"]) >= 1
            assert all("type" in exp for exp in suggestions["experiments"])
            assert all("parameters" in exp for exp in suggestions["experiments"])

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not available")
    def test_generate_experiment_protocol(self, mock_openai_client):
        """Test AI generation of experimental protocols."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = """
        # Dose-Response Experiment Protocol
        
        ## Objective
        Determine IC50 of compound CHEMBL12345 in HeLa cells
        
        ## Materials
        - HeLa cells (passage 15-25)
        - Compound CHEMBL12345 (10mM stock in DMSO)
        - 96-well imaging plates
        
        ## Protocol
        1. Seed cells at 5000 cells/well, 24h before treatment
        2. Prepare serial dilutions (0.1-50uM)
        3. Treat cells for 24h
        4. Assess viability using ATP assay
        5. Image for morphological analysis
        
        ## Analysis
        - Calculate IC50 using 4-parameter logistic fit
        - Quantify morphological parameters
        """
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        with patch('src.agents.openai_agent.OpenAI', return_value=mock_openai_client):
            agent = OpenPerturbationAgent(api_key="test-key")
            
            protocol = agent.generate_experiment_protocol(
                experiment_type="dose_response",
                parameters={"compound": "CHEMBL12345", "cell_line": "HeLa"}
            )
            
            assert "Protocol" in protocol
            assert "Materials" in protocol
            assert "Analysis" in protocol


class TestConversationHandler:
    """Test suite for conversation handling with AI agents."""

    def test_conversation_initialization(self):
        """Test conversation handler initialization."""
        handler = ConversationHandler(max_history=10)
        
        assert handler.max_history == 10
        assert handler.conversation_history == []
        assert handler.context == {}

    def test_add_message(self):
        """Test adding messages to conversation history."""
        handler = ConversationHandler(max_history=3)
        
        handler.add_message("user", "Hello, can you analyze my data?")
        handler.add_message("assistant", "Of course! Please share your experimental data.")
        
        assert len(handler.conversation_history) == 2
        assert handler.conversation_history[0]["role"] == "user"
        assert handler.conversation_history[1]["role"] == "assistant"

    def test_conversation_history_limit(self):
        """Test conversation history size limiting."""
        handler = ConversationHandler(max_history=2)
        
        for i in range(5):
            handler.add_message("user", f"Message {i}")
        
        assert len(handler.conversation_history) == 2
        assert handler.conversation_history[0]["content"] == "Message 3"
        assert handler.conversation_history[1]["content"] == "Message 4"

    def test_set_context(self):
        """Test setting conversation context."""
        handler = ConversationHandler()
        
        context = {
            "experiment_type": "drug_screening",
            "cell_line": "HeLa",
            "data_available": ["imaging", "genomics"]
        }
        
        handler.set_context(context)
        assert handler.context == context

    def test_get_conversation_summary(self):
        """Test conversation summarization."""
        handler = ConversationHandler()
        
        handler.add_message("user", "I need help analyzing perturbation data")
        handler.add_message("assistant", "I can help with that. What type of perturbation?")
        handler.add_message("user", "Chemical compound screening in cancer cells")
        
        summary = handler.get_conversation_summary()
        
        assert "messages" in summary
        assert "context" in summary
        assert len(summary["messages"]) == 3


class TestAgentTools:
    """Test suite for AI agent tools and utilities."""

    def test_data_formatter(self):
        """Test data formatting for AI consumption."""
        raw_data = {
            "cell_count": 1250,
            "viability": 0.87,
            "measurements": [1.2, 1.5, 1.8, 2.1]
        }
        
        formatted = AgentTools.format_data_for_ai(raw_data)
        
        assert isinstance(formatted, str)
        assert "cell_count: 1250" in formatted
        assert "viability: 87.0%" in formatted

    def test_response_parser(self):
        """Test parsing AI responses."""
        ai_response = """
        {
            "analysis": "Strong cytotoxic effect observed",
            "confidence": 0.92,
            "recommendations": ["Test lower doses", "Check cell cycle"]
        }
        """
        
        parsed = AgentTools.parse_ai_response(ai_response)
        
        assert "analysis" in parsed
        assert "confidence" in parsed
        assert parsed["confidence"] == 0.92

    def test_prompt_builder(self):
        """Test building prompts for AI agents."""
        data = {"compound": "CHEMBL12345", "viability": 0.85}
        context = {"experiment_type": "drug_screening"}
        
        prompt = AgentTools.build_analysis_prompt(
            data=data,
            context=context,
            query="What does this data suggest about the compound's mechanism?"
        )
        
        assert "CHEMBL12345" in prompt
        assert "viability" in prompt
        assert "drug_screening" in prompt
        assert "mechanism" in prompt

    def test_experiment_validator(self):
        """Test experiment design validation."""
        experiment_design = {
            "type": "dose_response",
            "compound": "CHEMBL12345",
            "concentrations": [0.1, 1, 10, 100],
            "timepoints": ["24h"],
            "cell_line": "HeLa"
        }
        
        validation = AgentTools.validate_experiment_design(experiment_design)
        
        assert validation["valid"] is True
        assert "errors" in validation
        assert len(validation["errors"]) == 0

    def test_experiment_validator_errors(self):
        """Test experiment design validation with errors."""
        experiment_design = {
            "type": "unknown_type",
            "concentrations": [],  # Empty concentrations
            # Missing required fields
        }
        
        validation = AgentTools.validate_experiment_design(experiment_design)
        
        assert validation["valid"] is False
        assert len(validation["errors"]) > 0


class TestAgentIntegration:
    """Integration tests for AI agent system."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not available")
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, mock_openai_client, sample_perturbation_data, sample_analysis_request):
        """Test complete AI-powered analysis workflow."""
        # Mock AI responses for different steps
        mock_responses = [
            # Initial analysis
            Mock(choices=[Mock(message=Mock(content=json.dumps({
                "analysis": {"summary": "Compound shows moderate cytotoxicity"},
                "insights": ["Cell cycle disruption observed"],
                "confidence": 0.88
            })))]),
            # Follow-up suggestions
            Mock(choices=[Mock(message=Mock(content=json.dumps({
                "experiments": [{"type": "dose_response", "priority": "high"}],
                "timeline": "1-2 weeks"
            })))]),
            # Protocol generation
            Mock(choices=[Mock(message=Mock(content="Detailed experimental protocol..."))])
        ]
        
        mock_openai_client.chat.completions.create.side_effect = mock_responses
        
        with patch('src.agents.openai_agent.OpenAI', return_value=mock_openai_client):
            agent = OpenPerturbationAgent(api_key="test-key")
            
            # Step 1: Analyze data
            analysis = await agent.analyze_perturbation_data(sample_perturbation_data)
            assert "analysis" in analysis
            
            # Step 2: Get suggestions
            suggestions = await agent.suggest_follow_up_experiments(sample_perturbation_data)
            assert "experiments" in suggestions
            
            # Step 3: Generate protocol
            protocol = agent.generate_experiment_protocol(
                experiment_type="dose_response",
                parameters={"compound": "test"}
            )
            assert len(protocol) > 0

    def test_agent_error_handling(self, mock_openai_client):
        """Test error handling in AI agent operations."""
        # Mock API error
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        with patch('src.agents.openai_agent.OpenAI', return_value=mock_openai_client):
            agent = OpenPerturbationAgent(api_key="test-key")
            
            with pytest.raises(Exception):
                agent.generate_experiment_protocol("invalid_type", {})

    @pytest.mark.asyncio
    async def test_conversation_flow(self):
        """Test conversational AI interaction flow."""
        handler = ConversationHandler(max_history=5)
        
        # Simulate conversation
        handler.add_message("user", "I have perturbation data to analyze")
        handler.add_message("assistant", "Great! What type of perturbation experiment?")
        handler.add_message("user", "Chemical compound screening in cancer cells")
        handler.add_message("assistant", "I'll analyze the compound effects. Please provide the data.")
        
        # Check conversation state
        assert len(handler.conversation_history) == 4
        assert handler.conversation_history[-1]["role"] == "assistant"
        
        # Test context setting
        handler.set_context({
            "experiment_type": "drug_screening",
            "domain": "oncology"
        })
        
        summary = handler.get_conversation_summary()
        assert summary["context"]["experiment_type"] == "drug_screening"

    def test_tool_integration(self):
        """Test integration between different agent tools."""
        # Test data flow through tools
        raw_data = {
            "compound_id": "CHEMBL12345",
            "viability": 0.75,
            "cell_count": 980
        }
        
        # Format data
        formatted = AgentTools.format_data_for_ai(raw_data)
        assert "CHEMBL12345" in formatted
        
        # Build prompt
        prompt = AgentTools.build_analysis_prompt(
            data=raw_data,
            context={"experiment_type": "viability_assay"},
            query="Analyze compound toxicity"
        )
        assert "viability" in prompt
        assert "toxicity" in prompt
        
        # Validate experiment
        experiment = {
            "type": "dose_response",
            "compound": "CHEMBL12345",
            "concentrations": [1, 10, 100],
            "timepoints": ["24h"],
            "cell_line": "HeLa"
        }
        
        validation = AgentTools.validate_experiment_design(experiment)
        assert validation["valid"] is True


@pytest.mark.integration
class TestOpenAIAPIIntegration:
    """Integration tests with actual OpenAI API (requires API key)."""

    @pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI SDK not available")
    @pytest.mark.asyncio
    async def test_real_api_call(self):
        """Test actual API call to OpenAI (requires valid API key)."""
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OpenAI API key not available")
        
        agent = OpenPerturbationAgent(api_key=api_key, model="gpt-3.5-turbo")
        
        # Simple test query
        test_data = {
            "compound": "aspirin",
            "viability": 0.92,
            "notes": "Well-known anti-inflammatory drug"
        }
        
        try:
            result = await agent.analyze_perturbation_data(test_data)
            assert isinstance(result, dict)
            # Should contain some analysis of aspirin
            assert any(keyword in str(result).lower() 
                      for keyword in ["aspirin", "anti-inflammatory", "drug"])
        except Exception as e:
            pytest.skip(f"API call failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
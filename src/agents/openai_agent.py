"""
OpenAI Agent for Perturbation Biology Analysis
Author: Nik Jois <nikjois@llamasearch.ai>

This module implements an AI-powered agent using OpenAI's GPT models to analyze
perturbation biology experiments, suggest follow-up studies, and generate protocols.
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

try:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletionMessageParam
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None
    ChatCompletionMessageParam = Dict[str, Any]

from .conversation_handler import ConversationHandler
from .agent_tools import AgentTools

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of AI analysis."""
    analysis: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence: float
    timestamp: datetime


@dataclass
class ExperimentSuggestion:
    """AI-generated experiment suggestion."""
    experiment_type: str
    parameters: Dict[str, Any]
    rationale: str
    priority: str
    estimated_duration: str
    required_resources: List[str]


class OpenPerturbationAgent:
    """
    AI agent for perturbation biology analysis using OpenAI's language models.
    
    This agent can:
    - Analyze experimental data and provide insights
    - Suggest follow-up experiments
    - Generate detailed protocols
    - Provide scientific explanations and interpretations
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: int = 30
    ):
        """
        Initialize the OpenPerturbation AI agent.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (gpt-4, gpt-3.5-turbo, etc.)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Initialize OpenAI clients
        if OPENAI_AVAILABLE and OpenAI is not None and AsyncOpenAI is not None:
            self.client = OpenAI(api_key=api_key, timeout=timeout)
            self.async_client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        else:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        # Initialize conversation handler - start a conversation
        self.conversation_handler = ConversationHandler()
        self.conversation_id = self.conversation_handler.start_conversation(
            user_id="agent_user", 
            agent_type="perturbation_analysis"
        )
        
        # System prompt for perturbation biology context
        self.system_prompt = """
        You are an expert AI assistant specializing in perturbation biology, cellular assays, 
        and high-content screening. Your role is to analyze experimental data, provide scientific 
        insights, and suggest follow-up experiments.
        
        Key capabilities:
        - Analyze high-content imaging data
        - Interpret genomics/transcriptomics results
        - Assess compound toxicity and efficacy
        - Design follow-up experiments
        - Generate detailed protocols
        - Explain biological mechanisms
        
        Always provide scientifically accurate, evidence-based responses with appropriate 
        confidence levels. When suggesting experiments, consider cost, feasibility, and 
        scientific value.
        """
        
        logger.info(f"Initialized OpenPerturbation Agent with model {model}")

    async def analyze_perturbation_data(
        self,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze perturbation experiment data using AI.
        
        Args:
            data: Experimental data to analyze
            context: Additional context about the experiment
            query: Specific question about the data
            
        Returns:
            AnalysisResult with AI insights and recommendations
        """
        try:
            # Format data for AI consumption
            formatted_data = AgentTools.format_data_for_ai(data)
            
            # Build analysis prompt
            prompt = self._build_analysis_prompt(formatted_data, context, query)
            
            # Get AI response
            response = await self._get_ai_response(prompt)
            
            # Parse response
            parsed_response = self._parse_analysis_response(response)
            
            # Create result object
            result = AnalysisResult(
                analysis=parsed_response.get("analysis", {}),
                insights=parsed_response.get("insights", []),
                recommendations=parsed_response.get("recommendations", []),
                confidence=parsed_response.get("confidence", 0.8),
                timestamp=datetime.now()
            )
            
            # Store in conversation history
            self.conversation_handler.add_message(
                self.conversation_id, 
                "user", 
                f"Analyze data: {formatted_data}"
            )
            self.conversation_handler.add_message(
                self.conversation_id,
                "assistant", 
                json.dumps(parsed_response)
            )
            
            logger.info(f"Completed data analysis with confidence {result.confidence}")
            return result
            
        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            raise

    async def suggest_follow_up_experiments(
        self,
        data: Dict[str, Any],
        current_results: Optional[AnalysisResult] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[ExperimentSuggestion]:
        """
        Generate AI-powered follow-up experiment suggestions.
        
        Args:
            data: Current experimental data
            current_results: Previous analysis results
            constraints: Budget, time, or resource constraints
            
        Returns:
            List of experiment suggestions
        """
        try:
            # Build suggestion prompt
            prompt = self._build_suggestion_prompt(data, current_results, constraints)
            
            # Get AI response
            response = await self._get_ai_response(prompt)
            
            # Parse suggestions
            suggestions_data = self._parse_suggestions_response(response)
            
            # Convert to ExperimentSuggestion objects
            suggestions = []
            for suggestion in suggestions_data.get("experiments", []):
                exp_suggestion = ExperimentSuggestion(
                    experiment_type=suggestion.get("type", "unknown"),
                    parameters=suggestion.get("parameters", {}),
                    rationale=suggestion.get("rationale", ""),
                    priority=suggestion.get("priority", "medium"),
                    estimated_duration=suggestion.get("duration", "1-2 weeks"),
                    required_resources=suggestion.get("resources", [])
                )
                suggestions.append(exp_suggestion)
            
            logger.info(f"Generated {len(suggestions)} experiment suggestions")
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            raise

    def generate_experiment_protocol(
        self,
        experiment_type: str,
        parameters: Dict[str, Any],
        detail_level: str = "detailed"
    ) -> str:
        """
        Generate detailed experimental protocol using AI.
        
        Args:
            experiment_type: Type of experiment (dose_response, time_course, etc.)
            parameters: Experiment parameters
            detail_level: Level of detail (basic, detailed, comprehensive)
            
        Returns:
            Formatted protocol text
        """
        try:
            prompt = self._build_protocol_prompt(experiment_type, parameters, detail_level)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent protocols
                max_tokens=self.max_tokens
            )
            
            protocol = response.choices[0].message.content or ""
            
            # Store in conversation history
            self.conversation_handler.add_message(
                self.conversation_id,
                "user", 
                f"Generate protocol: {experiment_type}"
            )
            self.conversation_handler.add_message(
                self.conversation_id,
                "assistant", 
                protocol
            )
            
            logger.info(f"Generated {experiment_type} protocol")
            return protocol
            
        except Exception as e:
            logger.error(f"Error generating protocol: {e}")
            raise

    async def explain_biological_mechanism(
        self,
        perturbation: str,
        observed_effects: List[str],
        cell_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Explain the biological mechanism behind observed perturbation effects.
        
        Args:
            perturbation: Type of perturbation (compound, gene knockdown, etc.)
            observed_effects: List of observed phenotypic effects
            cell_type: Cell type context
            
        Returns:
            Mechanism explanation with pathways and targets
        """
        try:
            prompt = f"""
            Explain the biological mechanism for the following perturbation:
            
            Perturbation: {perturbation}
            Cell Type: {cell_type}
            Observed Effects: {', '.join(observed_effects)}
            
            Please provide:
            1. Most likely molecular targets
            2. Affected biological pathways
            3. Mechanism of action
            4. Supporting evidence from literature
            5. Confidence in the explanation
            
            Format as JSON with appropriate sections.
            """
            
            response = await self._get_ai_response(prompt)
            mechanism = self._parse_json_response(response)
            
            logger.info(f"Generated mechanism explanation for {perturbation}")
            return mechanism
            
        except Exception as e:
            logger.error(f"Error explaining mechanism: {e}")
            raise

    async def chat_with_agent(self, message: str, context: Optional[Dict] = None) -> str:
        """
        Have a conversational interaction with the AI agent.
        
        Args:
            message: User message
            context: Optional context information
            
        Returns:
            Agent response
        """
        try:
            # Update context if provided
            if context:
                self.conversation_handler.update_context(self.conversation_id, context)
            
            # Add user message to history
            self.conversation_handler.add_message(self.conversation_id, "user", message)
            
            # Build conversation prompt using history
            conversation_history = self.conversation_handler.get_conversation_history(self.conversation_id)
            
            # Get AI response
            response = await self._get_ai_response(
                message,
                conversation_history=conversation_history
            )
            
            # Add response to history
            self.conversation_handler.add_message(self.conversation_id, "assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat interaction: {e}")
            raise

    async def _get_ai_response(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Get response from OpenAI API."""
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant", "system"]:
                    messages.append({"role": role, "content": content})
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content or ""

    def _build_analysis_prompt(
        self,
        data: str,
        context: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None
    ) -> str:
        """Build prompt for data analysis."""
        prompt = f"""
        Analyze the following perturbation biology data:
        
        Data:
        {data}
        """
        
        if context:
            prompt += f"\nContext: {json.dumps(context, indent=2)}"
        
        if query:
            prompt += f"\nSpecific Question: {query}"
        
        prompt += """
        
        Please provide a comprehensive analysis including:
        1. Summary of key findings
        2. Biological insights and interpretations
        3. Statistical significance assessment
        4. Potential confounding factors
        5. Recommendations for follow-up
        6. Confidence level (0.0-1.0)
        
        Format the response as JSON with the following structure:
        {
            "analysis": {
                "summary": "...",
                "key_findings": ["...", "..."],
                "statistical_assessment": "..."
            },
            "insights": ["...", "..."],
            "recommendations": ["...", "..."],
            "confidence": 0.85
        }
        """
        
        return prompt

    def _build_suggestion_prompt(
        self,
        data: Dict[str, Any],
        results: Optional[AnalysisResult] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for experiment suggestions."""
        prompt = f"""
        Based on the experimental data and analysis, suggest follow-up experiments:
        
        Current Data: {json.dumps(data, indent=2)}
        """
        
        if results:
            prompt += f"\nPrevious Analysis: {json.dumps(results.analysis, indent=2)}"
        
        if constraints:
            prompt += f"\nConstraints: {json.dumps(constraints, indent=2)}"
        
        prompt += """
        
        Suggest 2-4 follow-up experiments that would:
        1. Validate current findings
        2. Extend understanding
        3. Address limitations
        4. Provide mechanistic insights
        
        Format as JSON:
        {
            "experiments": [
                {
                    "type": "dose_response",
                    "parameters": {...},
                    "rationale": "...",
                    "priority": "high|medium|low",
                    "duration": "1-2 weeks",
                    "resources": ["equipment", "reagents"]
                }
            ]
        }
        """
        
        return prompt

    def _build_protocol_prompt(
        self,
        experiment_type: str,
        parameters: Dict[str, Any],
        detail_level: str
    ) -> str:
        """Build prompt for protocol generation."""
        return f"""
        Generate a {detail_level} experimental protocol for:
        
        Experiment Type: {experiment_type}
        Parameters: {json.dumps(parameters, indent=2)}
        
        Include:
        1. Objective and hypothesis
        2. Materials and reagents
        3. Equipment required
        4. Step-by-step procedure
        5. Quality control measures
        6. Data analysis plan
        7. Expected timeline
        8. Troubleshooting tips
        
        Format as clear, executable protocol.
        """

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse AI analysis response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            return {
                "analysis": {"summary": response[:500]},
                "insights": [],
                "recommendations": [],
                "confidence": 0.7
            }

    def _parse_suggestions_response(self, response: str) -> Dict[str, Any]:
        """Parse AI suggestions response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"experiments": []}

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse general JSON response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"response": response, "parsed": False}

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation with agent."""
        return self.conversation_handler.get_conversation_summary(self.conversation_id)

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_handler.end_conversation(self.conversation_id)
        self.conversation_id = self.conversation_handler.start_conversation(
            user_id="agent_user", 
            agent_type="perturbation_analysis"
        )
        logger.info("Reset conversation history")

    def set_model_parameters(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ):
        """Update model parameters."""
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if model is not None:
            self.model = model
        
        logger.info(f"Updated model parameters: {self.model}, temp={self.temperature}")


# Convenience functions for quick access
async def analyze_data_with_ai(data: Dict[str, Any], api_key: str) -> AnalysisResult:
    """Quick function to analyze data with AI."""
    agent = OpenPerturbationAgent(api_key=api_key)
    return await agent.analyze_perturbation_data(data)


async def get_experiment_suggestions(
    data: Dict[str, Any],
    api_key: str,
    constraints: Optional[Dict] = None
) -> List[ExperimentSuggestion]:
    """Quick function to get experiment suggestions."""
    agent = OpenPerturbationAgent(api_key=api_key)
    return await agent.suggest_follow_up_experiments(data, constraints=constraints)


def generate_protocol(
    experiment_type: str,
    parameters: Dict[str, Any],
    api_key: str
) -> str:
    """Quick function to generate protocol."""
    agent = OpenPerturbationAgent(api_key=api_key)
    return agent.generate_experiment_protocol(experiment_type, parameters)


class AnalysisAgent(OpenPerturbationAgent):
    """Backward-compatibility alias for OpenPerturbationAgent (analysis-focused)."""
    pass


class ExperimentDesignAgent(OpenPerturbationAgent):
    """Backward-compatibility alias for OpenPerturbationAgent (experiment-design helper)."""
    pass

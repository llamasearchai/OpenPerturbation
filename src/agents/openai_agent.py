#!/usr/bin/env python3
"""
OpenAI Agents SDK Integration for OpenPerturbation

Comprehensive AI agent integration using OpenAI's latest APIs and agents SDK
for intelligent conversation, analysis assistance, and automated workflows.

Author: Nik Jois
Email: nikjois@llamasearch.ai
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
import json
import asyncio
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available. Agent functionality will be limited.")

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logging.warning("Pydantic not available for data validation.")

from .agent_tools import (
    get_available_tools,
    execute_tool,
    DataAnalysisTool,
    CausalDiscoveryTool,
    ExplainabilityTool,
    InterventionDesignTool
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenPerturbationAgent:
    """
    Main OpenAI agent for OpenPerturbation platform.
    Provides intelligent assistance for data analysis, experiment design,
    and interpretation of results.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system_prompt: Optional[str] = None
    ):
        """Initialize OpenPerturbation AI agent."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("OpenAI client not initialized. Using mock responses.")
        
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.conversation_history: List[Dict[str, str]] = []
        self.tools = get_available_tools()
        self.tool_handlers = self._initialize_tool_handlers()
        
        logger.info(f"OpenPerturbation agent initialized with model: {model}")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for the agent."""
        return """
You are an expert AI assistant for the OpenPerturbation platform, specializing in:

1. **Perturbation Biology**: Understanding cellular responses to chemical and genetic perturbations
2. **Causal Discovery**: Identifying causal relationships in biological data
3. **Multi-modal Data Analysis**: Analyzing genomics, imaging, and molecular data
4. **Explainable AI**: Providing interpretable insights from complex models
5. **Intervention Design**: Designing optimal experimental interventions

Your capabilities include:
- Analyzing single-cell and bulk genomics data
- Running causal discovery algorithms
- Generating explainability reports
- Designing intervention strategies
- Providing scientific insights and recommendations

Always provide:
- Clear, scientifically accurate explanations
- Step-by-step guidance for complex analyses
- Actionable recommendations
- Code examples when appropriate
- References to relevant biological pathways and mechanisms

You have access to specialized tools for data analysis, causal discovery, explainability analysis, and intervention design. Use these tools when users request specific analyses.
"""
    
    def _initialize_tool_handlers(self) -> Dict[str, Callable]:
        """Initialize tool handlers for agent functions."""
        return {
            "data_analysis": DataAnalysisTool(),
            "causal_discovery": CausalDiscoveryTool(),
            "explainability": ExplainabilityTool(),
            "intervention_design": InterventionDesignTool()
        }
    
    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        use_tools: bool = True
    ) -> str:
        """Process a user message and generate a response."""
        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Prepare messages for OpenAI API
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history (limit to last 10 messages)
            for msg in self.conversation_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Check if user is requesting tool usage
            if use_tools and self._should_use_tools(message):
                response = await self._process_with_tools(message, context)
            else:
                response = await self._generate_response(messages)
            
            # Add assistant response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    def _should_use_tools(self, message: str) -> bool:
        """Determine if the message requires tool usage."""
        tool_keywords = [
            "analyze", "run analysis", "causal discovery", "explainability",
            "intervention", "design experiment", "upload data", "process dataset",
            "generate report", "statistical analysis", "pathway analysis"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in tool_keywords)
    
    async def _process_with_tools(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process message with tool assistance."""
        try:
            # Determine which tool to use based on message content
            tool_name = self._identify_required_tool(message)
            
            if tool_name and tool_name in self.tool_handlers:
                tool_handler = self.tool_handlers[tool_name]
                
                # Extract parameters from message and context
                parameters = self._extract_tool_parameters(message, context, tool_name)
                
                # Execute tool
                tool_result = await execute_tool(tool_name, parameters)
                
                # Generate response based on tool result
                response = await self._generate_tool_response(message, tool_result, tool_name)
                
                return response
            else:
                # Fall back to regular response generation
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message}
                ]
                return await self._generate_response(messages)
                
        except Exception as e:
            logger.error(f"Error in tool processing: {e}")
            return f"I encountered an error while using the analysis tools: {str(e)}"
    
    def _identify_required_tool(self, message: str) -> Optional[str]:
        """Identify which tool is needed based on the message."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["causal", "discovery", "graph", "relationships"]):
            return "causal_discovery"
        elif any(word in message_lower for word in ["explain", "explainability", "interpret", "pathway"]):
            return "explainability"
        elif any(word in message_lower for word in ["intervention", "design", "experiment", "optimal"]):
            return "intervention_design"
        elif any(word in message_lower for word in ["analyze", "analysis", "data", "statistics"]):
            return "data_analysis"
        else:
            return None
    
    def _extract_tool_parameters(
        self,
        message: str,
        context: Optional[Dict[str, Any]],
        tool_name: str
    ) -> Dict[str, Any]:
        """Extract parameters for tool execution from message and context."""
        parameters = {}
        
        if context:
            parameters.update(context)
        
        # Tool-specific parameter extraction
        if tool_name == "causal_discovery":
            parameters.update({
                "method": "pc",  # Default method
                "alpha": 0.05,
                "max_vars": 100
            })
        elif tool_name == "explainability":
            parameters.update({
                "analysis_types": ["attention", "concept", "pathway"],
                "num_samples": 100
            })
        elif tool_name == "intervention_design":
            parameters.update({
                "num_interventions": 10,
                "budget_constraints": {"max_cost": 10000}
            })
        elif tool_name == "data_analysis":
            parameters.update({
                "analysis_type": "descriptive",
                "include_plots": True
            })
        
        return parameters
    
    async def _generate_tool_response(
        self,
        original_message: str,
        tool_result: Dict[str, Any],
        tool_name: str
    ) -> str:
        """Generate a response based on tool execution results."""
        # Create a prompt that includes the tool results
        tool_summary = self._summarize_tool_result(tool_result, tool_name)
        
        prompt = f"""
The user asked: "{original_message}"

I executed the {tool_name} tool and got the following results:
{tool_summary}

Please provide a comprehensive, scientifically accurate response that:
1. Explains what was done
2. Interprets the key findings
3. Provides actionable insights
4. Suggests next steps if appropriate

Format your response in a clear, structured way that's accessible to both experts and non-experts.
"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        return await self._generate_response(messages)
    
    def _summarize_tool_result(self, result: Dict[str, Any], tool_name: str) -> str:
        """Create a summary of tool execution results."""
        if tool_name == "causal_discovery":
            return f"""
Causal Discovery Results:
- Number of variables analyzed: {result.get('n_variables', 'N/A')}
- Causal relationships found: {result.get('n_edges', 'N/A')}
- Discovery method: {result.get('method', 'N/A')}
- Statistical significance level: {result.get('alpha', 'N/A')}
- Execution time: {result.get('execution_time', 'N/A')} seconds
"""
        elif tool_name == "explainability":
            return f"""
Explainability Analysis Results:
- Analysis types completed: {result.get('analysis_types', [])}
- Samples analyzed: {result.get('num_samples', 'N/A')}
- Key findings: {result.get('key_findings', 'N/A')}
- Execution time: {result.get('execution_time', 'N/A')} seconds
"""
        elif tool_name == "intervention_design":
            return f"""
Intervention Design Results:
- Number of interventions designed: {result.get('num_interventions', 'N/A')}
- Total estimated cost: ${result.get('total_cost', 'N/A')}
- Expected effectiveness: {result.get('avg_effectiveness', 'N/A')}
- Execution time: {result.get('execution_time', 'N/A')} seconds
"""
        elif tool_name == "data_analysis":
            return f"""
Data Analysis Results:
- Dataset size: {result.get('dataset_size', 'N/A')}
- Features analyzed: {result.get('num_features', 'N/A')}
- Analysis type: {result.get('analysis_type', 'N/A')}
- Key statistics: {result.get('summary_stats', 'N/A')}
- Execution time: {result.get('execution_time', 'N/A')} seconds
"""
        else:
            return json.dumps(result, indent=2)
    
    async def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI API."""
        if not self.client:
            return self._generate_mock_response(messages[-1]["content"])
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._generate_mock_response(messages[-1]["content"])
    
    def _generate_mock_response(self, message: str) -> str:
        """Generate a mock response when OpenAI is not available."""
        message_lower = message.lower()
        
        if "causal" in message_lower:
            return """
I understand you're interested in causal discovery analysis. OpenPerturbation provides advanced causal discovery capabilities using methods like PC algorithm and others.

Key features include:
- Automated causal graph construction
- Statistical significance testing
- Bootstrap validation
- Visualization of causal relationships

To run causal discovery, you would typically:
1. Upload your dataset
2. Configure the discovery method and parameters
3. Execute the analysis
4. Interpret the resulting causal graph

Would you like me to guide you through the process?
"""
        elif "explain" in message_lower:
            return """
Explainability analysis in OpenPerturbation helps you understand how models make decisions and what biological mechanisms they capture.

Available analysis types:
- **Attention Analysis**: Shows which input features the model focuses on
- **Concept Activation**: Identifies biological concepts learned by the model
- **Pathway Analysis**: Maps findings to known biological pathways

This helps translate AI predictions into actionable biological insights.
"""
        elif "intervention" in message_lower:
            return """
Intervention design is a key strength of OpenPerturbation. The platform can:

1. **Analyze causal relationships** to identify intervention targets
2. **Optimize intervention strategies** based on desired outcomes
3. **Consider budget constraints** and experimental limitations
4. **Predict intervention effects** using causal models

This enables researchers to design more effective experiments and therapeutic strategies.
"""
        else:
            return """
I'm here to help you with OpenPerturbation analysis! I can assist with:

- **Data Analysis**: Understanding your datasets and experimental design
- **Causal Discovery**: Finding causal relationships in biological data
- **Explainability**: Interpreting model predictions and biological mechanisms
- **Intervention Design**: Optimizing experimental strategies

What would you like to explore today?
"""
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len([msg for msg in self.conversation_history if msg["role"] == "user"]),
            "assistant_messages": len([msg for msg in self.conversation_history if msg["role"] == "assistant"]),
            "conversation_start": self.conversation_history[0]["timestamp"] if self.conversation_history else None,
            "last_activity": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
    
    def export_conversation(self, format: str = "json") -> str:
        """Export conversation history in specified format."""
        if format == "json":
            return json.dumps(self.conversation_history, indent=2)
        elif format == "markdown":
            lines = ["# OpenPerturbation Conversation\n"]
            for msg in self.conversation_history:
                role = "**User**" if msg["role"] == "user" else "**Assistant**"
                timestamp = msg["timestamp"]
                content = msg["content"]
                lines.append(f"## {role} ({timestamp})\n{content}\n")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


class SpecializedAgent:
    """Base class for specialized agents with specific expertise."""
    
    def __init__(self, name: str, expertise: str, system_prompt: str):
        self.name = name
        self.expertise = expertise
        self.base_agent = OpenPerturbationAgent(system_prompt=system_prompt)
    
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process message with specialized expertise."""
        return await self.base_agent.process_message(message, context)


class CausalDiscoveryAgent(SpecializedAgent):
    """Specialized agent for causal discovery tasks."""
    
    def __init__(self):
        system_prompt = """
You are a specialized AI assistant expert in causal discovery and causal inference for biological systems.

Your expertise includes:
- Causal discovery algorithms (PC, GES, FCI, etc.)
- Directed acyclic graphs (DAGs) and causal graphs
- Confounding variables and causal identification
- Experimental design for causal inference
- Biological pathway causality
- Intervention effects and do-calculus

Provide detailed, technical guidance on causal analysis while making it accessible to biologists.
"""
        super().__init__("CausalDiscoveryAgent", "Causal Discovery & Inference", system_prompt)


class ExplainabilityAgent(SpecializedAgent):
    """Specialized agent for explainable AI tasks."""
    
    def __init__(self):
        system_prompt = """
You are a specialized AI assistant expert in explainable AI and interpretable machine learning for biological applications.

Your expertise includes:
- Attention mechanisms and attention visualization
- Concept activation vectors and TCAV
- Feature importance and SHAP values
- Biological pathway analysis and enrichment
- Model interpretability techniques
- Translating AI insights to biological understanding

Help users understand what their models learned and how to interpret AI predictions in biological context.
"""
        super().__init__("ExplainabilityAgent", "Explainable AI & Interpretability", system_prompt)


class InterventionAgent(SpecializedAgent):
    """Specialized agent for intervention design tasks."""
    
    def __init__(self):
        system_prompt = """
You are a specialized AI assistant expert in experimental design and intervention strategies for biological systems.

Your expertise includes:
- Optimal experimental design
- Drug combination optimization
- Perturbation strategies
- Active learning for experiments
- Cost-effective experimental planning
- Therapeutic intervention design

Guide users in designing effective experiments and interventions based on causal understanding.
"""
        super().__init__("InterventionAgent", "Intervention Design & Experimental Planning", system_prompt)


class AgentOrchestrator:
    """Orchestrates multiple specialized agents based on user needs."""
    
    def __init__(self):
        self.general_agent = OpenPerturbationAgent()
        self.specialized_agents = {
            "causal": CausalDiscoveryAgent(),
            "explainability": ExplainabilityAgent(),
            "intervention": InterventionAgent()
        }
        self.active_agent = self.general_agent
    
    async def route_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Route message to appropriate specialized agent."""
        agent_type = self._determine_agent_type(message)
        
        if agent_type in self.specialized_agents:
            self.active_agent = self.specialized_agents[agent_type]
            logger.info(f"Routing to specialized agent: {agent_type}")
        else:
            self.active_agent = self.general_agent
            logger.info("Using general agent")
        
        return await self.active_agent.process_message(message, context)
    
    def _determine_agent_type(self, message: str) -> Optional[str]:
        """Determine which specialized agent should handle the message."""
        message_lower = message.lower()
        
        causal_keywords = ["causal", "causality", "causal graph", "causal discovery", "dag", "confounding"]
        explainability_keywords = ["explain", "interpret", "attention", "concept", "pathway", "shap"]
        intervention_keywords = ["intervention", "experiment", "design", "optimize", "drug", "perturbation"]
        
        if any(keyword in message_lower for keyword in causal_keywords):
            return "causal"
        elif any(keyword in message_lower for keyword in explainability_keywords):
            return "explainability"
        elif any(keyword in message_lower for keyword in intervention_keywords):
            return "intervention"
        else:
            return None
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            "active_agent": self.active_agent.name if hasattr(self.active_agent, 'name') else "general",
            "available_agents": list(self.specialized_agents.keys()) + ["general"],
            "conversation_summary": self.active_agent.base_agent.get_conversation_summary() if hasattr(self.active_agent, 'base_agent') else self.active_agent.get_conversation_summary()
        }


# Factory function for creating agents
def create_openperturbation_agent(
    agent_type: str = "general",
    api_key: Optional[str] = None,
    **kwargs
) -> Union[OpenPerturbationAgent, SpecializedAgent, AgentOrchestrator]:
    """Factory function to create different types of OpenPerturbation agents."""
    
    if agent_type == "general":
        return OpenPerturbationAgent(api_key=api_key, **kwargs)
    elif agent_type == "causal":
        return CausalDiscoveryAgent()
    elif agent_type == "explainability":
        return ExplainabilityAgent()
    elif agent_type == "intervention":
        return InterventionAgent()
    elif agent_type == "orchestrator":
        return AgentOrchestrator()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# Testing function
async def test_agent_functionality():
    """Test agent functionality with sample interactions."""
    logger.info("Testing OpenPerturbation Agent functionality...")
    
    # Test general agent
    agent = create_openperturbation_agent("general")
    
    test_messages = [
        "What is OpenPerturbation?",
        "How do I run causal discovery analysis?",
        "Explain model attention maps",
        "Design an optimal intervention experiment"
    ]
    
    for message in test_messages:
        logger.info(f"Testing message: {message}")
        response = await agent.process_message(message)
        logger.info(f"Response: {response[:100]}...")
    
    # Test orchestrator
    orchestrator = create_openperturbation_agent("orchestrator")
    
    specialized_messages = [
        "Run causal discovery on my dataset",
        "Generate explainability report for my model",
        "Design intervention strategy for pathway X"
    ]
    
    for message in specialized_messages:
        logger.info(f"Testing orchestrator with: {message}")
        response = await orchestrator.route_message(message)
        logger.info(f"Orchestrator response: {response[:100]}...")
    
    logger.info("Agent testing completed successfully!")


if __name__ == "__main__":
    # Run agent tests
    asyncio.run(test_agent_functionality())

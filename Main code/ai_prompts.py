"""
ai_prompts.py - Dynamic Prompt Templates & Generation (Reasoning-Based)

Responsibilities:
- Build prompts that encourage AI reasoning and analysis
- Dynamic prompt generation based on evidence and reasoning chains
- No hardcoded solutions - promotes discovery from documents
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DynamicPromptTemplates:
    """Template storage for reasoning-based prompts"""
    
    def __init__(self):
        logger.info("📝 DynamicPromptTemplates initialized - Reasoning-focused prompts!")
        
        self.templates = {
            "dynamic_reasoning": """You are an expert technical problem solver and analyst. Analyze the documentation evidence to discover the solution through systematic reasoning.

USER PROBLEM: {query}

PROBLEM ANALYSIS:
{problem_analysis}

DOCUMENT EVIDENCE DISCOVERED:
{evidence_summary}

SOLUTION CANDIDATES FOUND:
{solution_candidates}

REASONING CHAIN:
{reasoning_chain}

DOCUMENTATION CONTEXT:
{document_context}

INSTRUCTIONS FOR ANALYSIS:
1. EXAMINE the user's problem and understand what they're trying to achieve
2. ANALYZE the evidence found in the documentation systematically  
3. EVALUATE each solution candidate based on relevance and strength
4. REASON through the relationships between concepts in the documents
5. SYNTHESIZE your conclusion from the evidence (don't just repeat it)
6. PROVIDE a clear solution with step-by-step reasoning
7. EXPLAIN why this solution is correct based on the document evidence
8. If evidence suggests multiple approaches, explain the trade-offs

CRITICAL: Base your answer on REASONING from the evidence, not just stating facts. Show your analytical thinking process.

Answer:""",

            "evidence_based_quick": """You are an expert analyst. Quickly analyze the evidence to find the best solution.

PROBLEM: {query}

EVIDENCE SUMMARY:
{evidence_summary}

TOP SOLUTION CANDIDATE:
{top_solution}

REASONING: {top_reasoning}

CONTEXT: {context_summary}

INSTRUCTIONS:
1. ANALYZE the evidence provided
2. REASON through why the top solution candidate is most relevant
3. PROVIDE a focused, practical answer based on your analysis
4. EXPLAIN the key reasoning behind your recommendation

Focus on analytical reasoning, not just repeating information.

Answer:""",

            "discovery_based": """You are a technical detective. Use the clues in the documentation to solve the user's problem.

USER'S MYSTERY: {query}

CLUES FOUND IN DOCUMENTATION:
{evidence_clues}

DOCUMENT CONTEXT FOR INVESTIGATION:
{document_context}

DETECTIVE INSTRUCTIONS:
1. INVESTIGATE what the user is trying to accomplish
2. EXAMINE the clues found in the documentation
3. DEDUCE the most likely solution from the evidence
4. BUILD a logical case for your recommendation
5. EXPLAIN your deductive reasoning process

Solve this like a detective using evidence and logical deduction.

Answer:""",

            "standard_analytical": """You are an expert Arma Reforger consultant. Provide detailed analysis and guidance.

{conversation_context}

DOCUMENTATION FOR ANALYSIS:
{document_context}

USER QUESTION: {query}

ANALYTICAL APPROACH:
1. Understand the user's technical challenge
2. Analyze the relevant documentation systematically
3. Identify the most appropriate solution approach
4. Provide detailed implementation guidance
5. Explain the technical reasoning behind your recommendations

Focus on analytical depth and practical implementation guidance.

Answer:""",

            "memory_analytical": """You are an analytical Arma Reforger assistant. Analyze our conversation history to provide insights.

CONVERSATION HISTORY FOR ANALYSIS:
{conversation_context}

USER QUESTION: {query}

ANALYTICAL INSTRUCTIONS:
1. ANALYZE the conversation history systematically
2. IDENTIFY patterns and topics we've discussed
3. REASON about connections between previous topics
4. SYNTHESIZE insights from our conversation
5. If the question isn't covered in our history, clearly state that

Base your response on analytical review of our actual conversation.

Answer:""",

            "question_improvement": """You are an expert technical communication analyst. Analyze and improve questions for better results.

{conversation_context}

ORIGINAL QUESTION: "{question}"

IMPROVEMENT ANALYSIS TASK:
1. ANALYZE what the user is trying to accomplish
2. IDENTIFY missing technical context or specificity
3. REASON about what information would help get better answers
4. GENERATE 3-4 improved versions with different focuses
5. EXPLAIN the analytical improvements made

FORMAT YOUR ANALYSIS:
**Improved Question Versions:**
1. **[Technical Context Version]**
   - Analysis: [why this improves the question]

2. **[Specific Implementation Version]** 
   - Analysis: [why this targets better answers]

3. **[Troubleshooting Focus Version]**
   - Analysis: [why this helps problem-solving]

**💡 Key Analytical Improvements:** [your reasoning process]
**🎯 Recommendation:** [which version to use and why]

Focus on analytical improvement of communication, not just adding keywords.

Answer:"""
        }

class DynamicPromptBuilder:
    """Dynamic prompt building based on reasoning and evidence"""
    
    def __init__(self):
        logger.info("🏗️ DynamicPromptBuilder initialized - Evidence-driven prompts!")
        self.templates = DynamicPromptTemplates()
    
    def build_reasoning_prompt(self, query: str, reasoning_result, context_docs: list, conversation_context: str = "") -> str:
        """Build prompt based on dynamic reasoning results"""
        
        logger.info(f"🏗️ Building reasoning-based prompt for query analysis")
        
        # Format the different components
        problem_analysis = self._format_problem_analysis(reasoning_result.problem_analysis)
        evidence_summary = self._format_evidence_summary(reasoning_result)
        solution_candidates = self._format_solution_candidates(reasoning_result.solution_candidates)
        reasoning_chain = self._format_reasoning_chain(reasoning_result.reasoning_chain)
        document_context = self._format_document_context(context_docs)
        
        prompt = self.templates.templates["dynamic_reasoning"].format(
            query=query,
            problem_analysis=problem_analysis,
            evidence_summary=evidence_summary,
            solution_candidates=solution_candidates,
            reasoning_chain=reasoning_chain,
            document_context=document_context
        )
        
        logger.info(f"✅ Reasoning prompt built ({len(prompt)} characters)")
        return prompt
    
    def build_quick_reasoning_prompt(self, query: str, reasoning_result, context_summary: str) -> str:
        """Build quick prompt based on reasoning evidence"""
        
        logger.info(f"⚡ Building quick reasoning prompt")
        
        evidence_summary = self._format_evidence_summary(reasoning_result)
        top_solution = reasoning_result.solution_candidates[0]["text"] if reasoning_result.solution_candidates else "No specific solution found"
        top_reasoning = reasoning_result.reasoning_chain[0] if reasoning_result.reasoning_chain else "Analysis in progress"
        
        prompt = self.templates.templates["evidence_based_quick"].format(
            query=query,
            evidence_summary=evidence_summary,
            top_solution=top_solution,
            top_reasoning=top_reasoning,
            context_summary=context_summary
        )
        
        logger.info(f"✅ Quick reasoning prompt built ({len(prompt)} characters)")
        return prompt
    
    def build_discovery_prompt(self, query: str, reasoning_result, context_docs: list) -> str:
        """Build discovery-focused prompt for complex analysis"""
        
        logger.info(f"🔍 Building discovery-based prompt")
        
        evidence_clues = self._format_evidence_as_clues(reasoning_result)
        document_context = self._format_document_context(context_docs)
        
        prompt = self.templates.templates["discovery_based"].format(
            query=query,
            evidence_clues=evidence_clues,
            document_context=document_context
        )
        
        return prompt
    
    def build_standard_prompt(self, query: str, context_docs: list, conversation_context: str) -> str:
        """Build standard analytical prompt for non-reasoning queries"""
        
        logger.info("📝 Building standard analytical prompt")
        
        document_context = self._format_document_context_simple(context_docs)
        
        prompt = self.templates.templates["standard_analytical"].format(
            conversation_context=conversation_context,
            document_context=document_context,
            query=query
        )
        
        return prompt
    
    def build_memory_prompt(self, query: str, conversation_context: str) -> str:
        """Build memory-focused analytical prompt"""
        
        logger.info("💭 Building memory analysis prompt")
        
        return self.templates.templates["memory_analytical"].format(
            conversation_context=conversation_context,
            query=query
        )
    
    def build_reform_prompt(self, question: str, conversation_context: str) -> str:
        """Build question improvement prompt based on analysis"""
        
        logger.info("🤖 Building question improvement prompt")
        
        return self.templates.templates["question_improvement"].format(
            conversation_context=conversation_context,
            question=question
        )
    
    def _format_problem_analysis(self, problem_analysis: Dict) -> str:
        """Format problem analysis for prompt"""
        
        formatted = [
            f"Primary Problem Type: {problem_analysis['primary_problem_type']}",
            f"Problem Indicators Detected: {list(problem_analysis['problem_indicators'].keys())}"
        ]
        
        if problem_analysis['related_document_problems']:
            formatted.append("Related Problems Found in Documentation:")
            for problem in problem_analysis['related_document_problems'][:3]:
                formatted.append(f"  • {problem['text'][:80]}...")
        
        return "\n".join(formatted)
    
    def _format_evidence_summary(self, reasoning_result) -> str:
        """Format evidence summary for prompt"""
        
        summary_parts = []
        
        # Solution evidence
        if reasoning_result.solution_candidates:
            summary_parts.append(f"Solutions Found: {len(reasoning_result.solution_candidates)}")
            top_solution = reasoning_result.solution_candidates[0]
            summary_parts.append(f"  Best Solution Type: {top_solution['type']}")
            summary_parts.append(f"  Evidence Strength: {top_solution['score']:.1f}/10")
        
        # Overall evidence metrics
        summary_parts.append(f"Overall Evidence Strength: {reasoning_result.evidence_strength:.2f}")
        summary_parts.append(f"Reasoning Confidence: {reasoning_result.confidence:.2f}")
        
        # Document insights
        if reasoning_result.document_insights:
            summary_parts.append("Key Insights:")
            for insight in reasoning_result.document_insights:
                summary_parts.append(f"  • {insight}")
        
        return "\n".join(summary_parts)
    
    def _format_solution_candidates(self, solution_candidates: List[Dict]) -> str:
        """Format solution candidates for prompt"""
        
        if not solution_candidates:
            return "No solution candidates identified from document analysis."
        
        formatted = []
        for i, candidate in enumerate(solution_candidates[:4], 1):  # Top 4 candidates
            formatted.append(f"{i}. {candidate['text']}")
            formatted.append(f"   Type: {candidate['type']} | Score: {candidate['score']:.1f} | Confidence: {candidate['confidence']:.2f}")
            formatted.append("")  # Empty line for readability
        
        return "\n".join(formatted)
    
    def _format_reasoning_chain(self, reasoning_chain: List[str]) -> str:
        """Format reasoning chain for prompt"""
        
        if not reasoning_chain:
            return "No reasoning chain available."
        
        formatted = []
        for i, step in enumerate(reasoning_chain, 1):
            formatted.append(f"{i}. {step}")
        
        return "\n".join(formatted)
    
    def _format_evidence_as_clues(self, reasoning_result) -> str:
        """Format evidence as detective clues"""
        
        clues = []
        
        if reasoning_result.solution_candidates:
            clues.append("🔍 SOLUTION CLUES:")
            for i, candidate in enumerate(reasoning_result.solution_candidates[:3], 1):
                clues.append(f"   Clue {i}: {candidate['text'][:60]}...")
                clues.append(f"   Evidence Type: {candidate['type']}")
        
        if reasoning_result.reasoning_chain:
            clues.append("\n🧩 REASONING CLUES:")
            for clue in reasoning_result.reasoning_chain[:3]:
                clues.append(f"   • {clue}")
        
        if reasoning_result.document_insights:
            clues.append("\n💡 INSIGHT CLUES:")
            for insight in reasoning_result.document_insights:
                clues.append(f"   • {insight}")
        
        return "\n".join(clues) if clues else "No clear clues found in documentation."
    
    def _format_document_context(self, context_docs: list) -> str:
        """Format document context for reasoning"""
        
        if not context_docs:
            return "No documentation available for analysis."
        
        formatted_docs = []
        for i, doc in enumerate(context_docs[:5], 1):  # Limit to 5 docs
            filename = doc.metadata.get('filename', 'Unknown')
            category = doc.metadata.get('main_category', 'Unknown')
            
            # Truncate content but keep it substantial for reasoning
            content = doc.page_content
            if len(content) > 600:
                content = content[:600] + "..."
            
            formatted_docs.append(f"Document {i} - {filename} [{category}]:\n{content}")
        
        return "\n\n".join(formatted_docs)
    
    def _format_document_context_simple(self, context_docs: list) -> str:
        """Simple document formatting for standard prompts"""
        
        if not context_docs:
            return "No relevant documentation found."
        
        formatted_docs = []
        for i, doc in enumerate(context_docs, 1):
            source_info = f"[{doc.metadata.get('main_category', 'Unknown')}/{doc.metadata.get('sub_category', 'Unknown')}]"
            formatted_docs.append(f"Document {i} {source_info}:\n{doc.page_content}\n")
        
        return "\n".join(formatted_docs)

class ReasoningPromptOptimizer:
    """Optimize prompts for different reasoning modes"""
    
    def __init__(self):
        logger.info("⚙️ ReasoningPromptOptimizer initialized")
    
    def optimize_for_reasoning_depth(self, prompt: str, reasoning_result) -> str:
        """Optimize prompt based on reasoning complexity"""
        
        confidence = reasoning_result.confidence
        evidence_strength = reasoning_result.evidence_strength
        
        if confidence < 0.5 or evidence_strength < 0.3:
            # Low confidence - encourage more careful analysis
            optimization_header = """OPTIMIZATION: Evidence is limited. Be extra careful in your analysis. Focus on what can be reasonably concluded from the available information. Acknowledge uncertainty where appropriate.

"""
        elif confidence > 0.8 and evidence_strength > 0.7:
            # High confidence - encourage comprehensive response
            optimization_header = """OPTIMIZATION: Strong evidence available. Provide a comprehensive, detailed analysis with clear reasoning steps and practical implementation guidance.

"""
        else:
            # Medium confidence - balanced approach
            optimization_header = """OPTIMIZATION: Moderate evidence available. Provide a balanced analysis, highlighting both what the evidence supports and any limitations.

"""
        
        return optimization_header + prompt
    
    def optimize_for_response_mode(self, prompt: str, mode: str) -> str:
        """Optimize prompt for specific response mode"""
        
        mode_optimizations = {
            "quick": "SPEED OPTIMIZATION: Focus on the most relevant evidence and provide a direct, actionable answer.\n\n",
            "detailed": "DEPTH OPTIMIZATION: Provide comprehensive analysis with detailed reasoning and multiple solution aspects.\n\n",
            "discovery": "DISCOVERY OPTIMIZATION: Act like a technical detective. Use systematic investigation and logical deduction.\n\n"
        }
        
        optimization = mode_optimizations.get(mode, "")
        return optimization + prompt
    
    def validate_prompt_reasoning_focus(self, prompt: str) -> bool:
        """Validate that prompt encourages reasoning over fact repetition"""
        
        reasoning_indicators = [
            "analyze", "examine", "evaluate", "reason", "deduce", "synthesize",
            "investigate", "systematic", "logical", "evidence", "conclusion"
        ]
        
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator.lower() in prompt.lower())
        
        # Should have at least 5 reasoning indicators for good reasoning focus
        return reasoning_count >= 5
    
    def get_reasoning_score(self, prompt: str) -> float:
        """Score how well prompt encourages reasoning"""
        
        reasoning_words = [
            "analyze", "examine", "evaluate", "reason", "deduce", "synthesize", "investigate",
            "systematic", "logical", "evidence", "conclusion", "thinking", "process"
        ]
        
        fact_repetition_words = [
            "state", "repeat", "copy", "list", "just", "simply", "only", "directly"
        ]
        
        reasoning_score = sum(1 for word in reasoning_words if word in prompt.lower())
        repetition_penalty = sum(1 for word in fact_repetition_words if word in prompt.lower())
        
        total_score = reasoning_score - (repetition_penalty * 0.5)
        return max(0, min(total_score / 15.0, 1.0))  # Normalize to 0-1

# Test function for development
def test_dynamic_prompts():
    """Test dynamic prompt building"""
    
    print("🧪 TESTING DYNAMIC PROMPT BUILDING (REASONING-BASED)")
    print("-" * 60)
    
    # Mock reasoning result
    from ai_reasoning import ReasoningResult
    
    mock_reasoning = ReasoningResult(
        problem_analysis={"primary_problem_type": "missing_elements", "problem_indicators": {"missing_elements": 2}},
        solution_candidates=[{
            "text": "In case of skinned assets like rifles, Export Skinning option should be used instead",
            "score": 8.5,
            "type": "direct_solution",
            "confidence": 0.85
        }],
        reasoning_chain=["Problem identified: missing_elements", "Solution found: direct_solution"],
        confidence=0.85,
        evidence_strength=0.75,
        document_insights=["Found 1 potential solution in documentation"]
    )
    
    # Mock document
    class MockDoc:
        def __init__(self):
            self.page_content = "In case of skinned assets like rifles, Export Skinning option should be used instead."
            self.metadata = {"filename": "Test.pdf", "main_category": "Documentation"}
    
    query = "imported gun but don't see bones"
    docs = [MockDoc()]
    
    # Test prompt building
    builder = DynamicPromptBuilder()
    optimizer = ReasoningPromptOptimizer()
    
    print("🏗️ Building reasoning prompt...")
    prompt = builder.build_reasoning_prompt(query, mock_reasoning, docs)
    
    print(f"✅ Prompt built: {len(prompt)} characters")
    
    # Test optimization
    optimized = optimizer.optimize_for_reasoning_depth(prompt, mock_reasoning)
    reasoning_score = optimizer.get_reasoning_score(optimized)
    
    print(f"🎯 Reasoning score: {reasoning_score:.2f}")
    print(f"📊 Reasoning focus valid: {optimizer.validate_prompt_reasoning_focus(optimized)}")
    
    # Show preview
    print("\n📖 REASONING PROMPT PREVIEW:")
    print("-" * 40)
    print(optimized[:500] + "...")

if __name__ == "__main__":
    test_dynamic_prompts()
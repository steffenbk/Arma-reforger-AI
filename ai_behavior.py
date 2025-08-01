"""
ai_behavior.py - Dynamic AI Behavior Coordination & Orchestration

Responsibilities:
- Coordinate dynamic reasoning + evidence-based prompts
- Handle different AI analysis modes (reasoning, discovery, analytical)
- Orchestrate intelligent problem-solving process
- No hardcoded solutions - purely evidence-driven
"""

import logging
from typing import Dict, Any, List, Optional
from langchain.schema import Document

from ai_reasoning import DynamicReasoningEngine, ReasoningResult
from ai_prompts import DynamicPromptBuilder, ReasoningPromptOptimizer

logger = logging.getLogger(__name__)

class DynamicAIBehaviorCoordinator:
    """Coordinates dynamic AI reasoning, evidence analysis, and intelligent response generation"""
    
    def __init__(self):
        logger.info("🤖 DynamicAIBehaviorCoordinator initializing...")
        
        self.reasoning_engine = DynamicReasoningEngine()
        self.prompt_builder = DynamicPromptBuilder()
        self.prompt_optimizer = ReasoningPromptOptimizer()
        
        # Analysis modes for different types of queries
        self.analysis_modes = {
            "reasoning": "Deep analytical reasoning with evidence chains",
            "discovery": "Detective-style problem solving",
            "quick_analysis": "Fast evidence-based analysis",
            "standard": "Standard analytical approach"
        }
        
        logger.info("✅ DynamicAIBehaviorCoordinator ready for intelligent analysis!")
    
    def analyze_and_respond(self, query: str, context_docs: List[Document], 
                           conversation_context: str, response_mode: str = "reasoning") -> Dict[str, Any]:
        """Main coordination method - intelligent analysis without hardcoded solutions"""
        
        logger.info(f"🤖 Dynamic AI analysis starting - Mode: {response_mode}")
        logger.info(f"📝 Query: '{query[:50]}...' | Documents: {len(context_docs)}")
        
        # Step 1: Dynamic reasoning analysis
        reasoning_result = self.reasoning_engine.reason_about_problem(query, context_docs)
        
        # Step 2: Determine optimal response approach
        optimal_mode = self._determine_optimal_analysis_mode(reasoning_result, response_mode)
        logger.info(f"🎯 Optimal analysis mode: {optimal_mode}")
        
        # Step 3: Build evidence-based prompt
        prompt = self._build_evidence_based_prompt(
            query, reasoning_result, context_docs, conversation_context, optimal_mode
        )
        
        # Step 4: Optimize prompt for reasoning quality
        optimized_prompt = self.prompt_optimizer.optimize_for_reasoning_depth(prompt, reasoning_result)
        optimized_prompt = self.prompt_optimizer.optimize_for_response_mode(optimized_prompt, optimal_mode)
        
        # Step 5: Validate reasoning focus
        reasoning_quality = self.prompt_optimizer.get_reasoning_score(optimized_prompt)
        logger.info(f"🧠 Reasoning quality score: {reasoning_quality:.2f}")
        
        # Step 6: Generate analysis metadata
        analysis_metadata = self._generate_analysis_metadata(reasoning_result, optimal_mode, reasoning_quality)
        
        # Step 7: Create suggested follow-ups based on evidence
        suggested_follow_ups = self._generate_evidence_based_follow_ups(reasoning_result, query)
        
        logger.info(f"✅ Dynamic analysis complete - Confidence: {reasoning_result.confidence:.2f}")
        
        return {
            "prompt": optimized_prompt,
            "reasoning_result": reasoning_result,
            "analysis_mode": optimal_mode,
            "analysis_confidence": reasoning_result.confidence,
            "evidence_strength": reasoning_result.evidence_strength,
            "reasoning_quality": reasoning_quality,
            "suggested_follow_ups": suggested_follow_ups,
            "response_mode": response_mode,
            "analysis_metadata": analysis_metadata,
            "prompt_valid": self.prompt_optimizer.validate_prompt_reasoning_focus(optimized_prompt),
            "token_estimate": len(optimized_prompt) // 4  # Rough estimate
        }
    
    def should_use_intelligent_analysis(self, query: str) -> bool:
        """Determine if query needs intelligent analysis - FINAL REFINED VERSION"""
        query_lower = query.lower().strip()
        
        # FIRST: Check for explanation/understanding requests about concepts
        explanation_indicators = [
            "explain", "what does", "how does", "what is this", "what is the",
            "break down", "walk through", "tell me about", "what happens here", 
            "meaning of", "describe", "show me what", "what's this", "can you explain", 
            "help me understand", "clarify", "definition of", "purpose of", "role of", "function of"
        ]
        
        # Enhanced explanation detection with better context awareness
        for indicator in explanation_indicators:
            if indicator in query_lower:
                # Exception: if it's asking for help with a specific implementation or fix
                action_words = ["how to create", "how to make", "how to build", "how to implement", 
                              "how to fix", "how to solve", "how to repair", "how to configure"]
                is_action_request = any(action in query_lower for action in action_words)
                
                # Exception: if asking about specific broken thing with personal pronouns
                personal_pronouns = ["my", "i'm", "i am", "i have", "i get"]
                specific_problem = any(word in query_lower for word in ["this error", "this issue", "this problem", "why this broke"])
                has_personal = any(pronoun in query_lower for pronoun in personal_pronouns)
                
                if not is_action_request and not (specific_problem and has_personal):
                    # Additional check: educational questions about problem concepts vs actual problems
                    educational_patterns = [
                        "what are common", "what are typical", "how do errors work", "how does debugging work",
                        "what types of", "teach me about", "learn about", "understand how",
                        "what does this error mean", "what does this mean", "meaning of this"
                    ]
                    is_educational = any(pattern in query_lower for pattern in educational_patterns)
                    
                    # Check if it's a general conceptual question (not about user's specific problem)
                    general_question = not has_personal and (is_educational or 
                                     any(word in query_lower for word in ["errors", "debugging", "troubleshooting", "common"]))
                    
                    if is_educational or general_question:
                        logger.info(f"📖 Explanation/educational request detected: '{indicator}' - using standard analysis")
                        return False
        
        # SECOND: Check for learning/educational requests (general knowledge)
        learning_indicators = [
            "teach me", "learn about", "tutorial", "guide me", "introduction to", 
            "basics of", "overview of", "summary of", "what are the types",
            "difference between", "compare", "contrast", "advantages", "disadvantages"
        ]
        
        for indicator in learning_indicators:
            if indicator in query_lower:
                # Exception: "teach me to fix this specific thing" is still a problem
                specific_fix = any(word in query_lower for word in ["fix this", "solve this", "repair this"])
                personal_learning = any(word in query_lower for word in ["my", "i'm", "i have"])
                
                if not specific_fix and not (personal_learning and "fix" in query_lower):
                    logger.info(f"📚 Learning request detected: '{indicator}' - using standard analysis")
                    return False
        
        # THIRD: Check for conceptual/theoretical questions
        conceptual_indicators = [
            "what are", "what is", "why do", "why does", "why is",
            "when do", "when does", "when is", "where do", "where does", 
            "which is", "which are", "similarities", "pros and cons"
        ]
        
        conceptual_score = sum(1 for indicator in conceptual_indicators if indicator in query_lower)
        if conceptual_score > 0:
            # Only treat as problem if there's a clear personal issue mentioned
            active_problems = ["not working", "broken", "error", "issue", "problem", "failing"]
            has_active_problem = any(problem in query_lower for problem in active_problems)
            personal_pronouns = ["my", "i'm", "i am", "i have", "i get"]
            has_personal = any(pronoun in query_lower for pronoun in personal_pronouns)
            
            # General questions about errors/problems are educational, not problem-solving
            general_error_questions = ["what are common errors", "how do errors work", "what types of errors"]
            is_general_error_question = any(pattern in query_lower for pattern in general_error_questions)
            
            if not has_active_problem or not has_personal or is_general_error_question:
                logger.info(f"🎓 Conceptual question detected - using standard analysis")
                return False
        
        # FOURTH: Check for comparison/analysis requests (non-problematic)
        comparison_indicators = [
            "vs", "versus", "compare", "difference between", "better than",
            "which is better", "pros and cons", "advantages of", "disadvantages of"
        ]
        
        for indicator in comparison_indicators:
            if indicator in query_lower:
                logger.info(f"⚖️ Comparison request detected: '{indicator}' - using standard analysis")
                return False
        
        # FIFTH: Check for actual technical problems that need solving
        problem_indicators = [
            "not working", "not compiling", "broken", "error", "problem", "issue", "trouble",
            "can't", "won't", "doesn't work", "failing", "stuck", "crashes", "fails", 
            "incorrect", "wrong", "missing", "don't see", "can't find", "won't load"
        ]
        
        problem_score = sum(1 for indicator in problem_indicators if indicator in query_lower)
        
        # SIXTH: Check for implementation/creation requests - IMPROVED DETECTION
        implementation_patterns = [
            "how do i create", "how to create", "how do i make", "how to make", 
            "how do i build", "how to build", "how do i implement", "how to implement",
            "how do i setup", "how to setup", "how do i configure", "how to configure",
            "need to create", "need to make", "need to build", "need to implement",
            "want to create", "want to make", "want to build", "want to implement",
            "trying to create", "trying to make", "trying to build", "trying to implement",
            "attempting to", "looking to create", "looking to make"
        ]
        
        implementation_score = sum(1 for pattern in implementation_patterns if pattern in query_lower)
        
        # Additional implementation indicators
        simple_implementation = ["create", "make", "build", "implement", "develop", "design"]
        context_words = ["custom", "new", "own", "personal", "specific"]
        
        has_simple_impl = any(word in query_lower for word in simple_implementation)
        has_context = any(word in query_lower for word in context_words)
        
        if has_simple_impl and has_context:
            implementation_score += 1
        
        # SEVENTH: Check for configuration/setup requests  
        config_patterns = [
            "configure", "setup", "set up", "install", "customize", "modify",
            "adjust", "tune", "optimize", "initialize"
        ]
        
        config_score = sum(1 for pattern in config_patterns if pattern in query_lower)
        
        # EIGHTH: Check for complex technical analysis needs
        complex_patterns = [
            "best way to", "proper way to", "correct way to", "recommended approach",
            "which approach", "what should i do", "improve performance",
            "most efficient", "best practice", "optimize"
        ]
        
        complexity_score = sum(1 for pattern in complex_patterns if pattern in query_lower)
        
        # NINTH: Multi-part questions (these often need problem-solving)
        multipart_indicators = ["and also", "but also", "however", "additionally", "furthermore"]
        multipart_score = sum(1 for indicator in multipart_indicators if indicator in query_lower)
        
        # TENTH: Check for troubleshooting context with personal pronouns
        troubleshooting_context = [
            "my", "i'm", "i am", "i have", "i get", "i'm getting", "i'm trying",
            "when i", "after i", "before i", "during"
        ]
        
        personal_context = any(context in query_lower for context in troubleshooting_context)
        has_tech_problem = any(problem in query_lower for problem in ["error", "issue", "problem", "not working"])
        
        context_score = 1 if personal_context and has_tech_problem else 0
        
        # Calculate total score from problem-solving categories
        total_score = (problem_score + implementation_score + config_score + 
                      complexity_score + multipart_score + context_score)
        
        # Decision logic with better thresholds
        if total_score >= 2:
            logger.info(f"🎯 Intelligent analysis needed - Score: {total_score} (problem: {problem_score}, implementation: {implementation_score}, config: {config_score}, complex: {complexity_score}, multipart: {multipart_score}, context: {context_score})")
            return True
        elif total_score == 1:
            # Single strong indicators that should trigger intelligent analysis
            if problem_score == 1 or implementation_score == 1 or config_score == 1:
                # But check for exceptions: general educational questions about these topics
                educational_exceptions = [
                    "what are common", "how do errors work", "how does", "what does",
                    "what types of", "learn about", "teach me about"
                ]
                is_educational_exception = any(pattern in query_lower for pattern in educational_exceptions)
                
                if not is_educational_exception:
                    logger.info(f"🔧 Single strong indicator detected - using intelligent analysis")
                    return True
        
        logger.info(f"📝 Standard processing sufficient - Score: {total_score} (explanation/learning/conceptual request)")
        return False
    
    def get_response_mode(self, query: str, prefix_used: str = None) -> str:
        """Determine appropriate response mode based on query complexity"""
        
        if prefix_used:
            if prefix_used.startswith("quick_"):
                return "quick_analysis"
            elif prefix_used == "reform":
                return "reform"
            elif prefix_used.startswith("force_") or prefix_used.startswith("dynamic_"):
                return "discovery"  # Use discovery mode for complex searches
        
        # Check for memory queries
        if query.startswith("MEMORY_ONLY:") or query.lower().startswith("memory "):
            return "memory_only"
        
        # Check for reform queries
        if query.lower().startswith("reform "):
            return "reform"
        
        # Analyze query complexity for mode selection
        query_lower = query.lower()
        
        # Discovery mode indicators
        discovery_indicators = ["debug", "investigate", "analyze", "why", "how does", "what causes"]
        if any(indicator in query_lower for indicator in discovery_indicators):
            return "discovery"
        
        # Quick analysis indicators
        quick_indicators = ["quick", "fast", "simple", "just", "only"]
        if any(indicator in query_lower for indicator in quick_indicators):
            return "quick_analysis"
        
        # Default to reasoning mode
        return "reasoning"
    
    def _determine_optimal_analysis_mode(self, reasoning_result: ReasoningResult, requested_mode: str) -> str:
        """Determine optimal analysis mode based on evidence quality"""
        
        confidence = reasoning_result.confidence
        evidence_strength = reasoning_result.evidence_strength
        solution_count = len(reasoning_result.solution_candidates)
        
        # Override mode based on evidence quality
        if confidence < 0.4 or evidence_strength < 0.3:
            # Low confidence - use discovery mode for deeper investigation
            if requested_mode in ["reasoning", "standard"]:
                logger.info("🔍 Switching to discovery mode due to low confidence")
                return "discovery"
        
        elif confidence > 0.8 and evidence_strength > 0.7 and solution_count > 1:
            # High confidence with multiple solutions - use reasoning mode
            if requested_mode == "quick_analysis":
                logger.info("🧠 Upgrading to reasoning mode due to high evidence quality")
                return "reasoning"
        
        # Use requested mode if evidence supports it
        return requested_mode
    
    def _build_evidence_based_prompt(self, query: str, reasoning_result: ReasoningResult, 
                                   context_docs: List[Document], conversation_context: str, 
                                   analysis_mode: str) -> str:
        """Build prompt based on evidence and analysis mode"""
        
        if analysis_mode == "quick_analysis":
            context_summary = self._create_context_summary(context_docs)
            return self.prompt_builder.build_quick_reasoning_prompt(query, reasoning_result, context_summary)
        
        elif analysis_mode == "discovery":
            return self.prompt_builder.build_discovery_prompt(query, reasoning_result, context_docs)
        
        elif analysis_mode == "memory_only":
            return self.prompt_builder.build_memory_prompt(query, conversation_context)
        
        elif analysis_mode == "reform":
            return self.prompt_builder.build_reform_prompt(query, conversation_context)
        
        elif analysis_mode == "standard":
            return self.prompt_builder.build_standard_prompt(query, context_docs, conversation_context)
        
        else:  # reasoning mode (default)
            return self.prompt_builder.build_reasoning_prompt(query, reasoning_result, context_docs, conversation_context)
    
    def _generate_analysis_metadata(self, reasoning_result: ReasoningResult, analysis_mode: str, reasoning_quality: float) -> Dict[str, Any]:
        """Generate metadata about the analysis process"""
        
        return {
            "problem_type": reasoning_result.problem_analysis.get("primary_problem_type", "unknown"),
            "solution_candidates_found": len(reasoning_result.solution_candidates),
            "reasoning_steps": len(reasoning_result.reasoning_chain),
            "evidence_sources": len(reasoning_result.document_insights),
            "analysis_mode_used": analysis_mode,
            "reasoning_quality_score": reasoning_quality,
            "analysis_complexity": self._calculate_analysis_complexity(reasoning_result),
            "recommendation_strength": self._calculate_recommendation_strength(reasoning_result)
        }
    
    def _calculate_analysis_complexity(self, reasoning_result: ReasoningResult) -> str:
        """Calculate complexity level of the analysis"""
        
        complexity_score = (
            len(reasoning_result.solution_candidates) * 2 +
            len(reasoning_result.reasoning_chain) +
            len(reasoning_result.document_insights)
        )
        
        if complexity_score >= 10:
            return "high"
        elif complexity_score >= 5:
            return "medium"
        else:
            return "low"
    
    def _calculate_recommendation_strength(self, reasoning_result: ReasoningResult) -> str:
        """Calculate strength of recommendations"""
        
        if reasoning_result.confidence >= 0.8 and reasoning_result.evidence_strength >= 0.7:
            return "strong"
        elif reasoning_result.confidence >= 0.6 and reasoning_result.evidence_strength >= 0.5:
            return "moderate"
        else:
            return "weak"
    
    def _generate_evidence_based_follow_ups(self, reasoning_result: ReasoningResult, original_query: str) -> List[str]:
        """Generate follow-up questions based on evidence analysis"""
        
        follow_ups = []
        
        # Based on problem type
        problem_type = reasoning_result.problem_analysis.get("primary_problem_type", "")
        
        if "missing" in problem_type:
            follow_ups.extend([
                "What are the complete troubleshooting steps for this issue?",
                "Are there alternative approaches to solve this problem?",
                "How can I verify that the solution worked correctly?"
            ])
        
        elif "import_export" in problem_type:
            follow_ups.extend([
                "What are the recommended export settings for this type of asset?",
                "How do I verify my import configuration is correct?",
                "What are common mistakes to avoid in this workflow?"
            ])
        
        elif "configuration" in problem_type:
            follow_ups.extend([
                "Can you walk me through the complete configuration process?",
                "What settings should I double-check?",
                "How do I test if the configuration is working properly?"
            ])
        
        # Based on solution candidates
        if len(reasoning_result.solution_candidates) > 1:
            follow_ups.append("What are the trade-offs between the different solution approaches?")
        
        # Based on confidence level
        if reasoning_result.confidence < 0.6:
            follow_ups.append("Can you help me gather more information to clarify this issue?")
        
        # Generic helpful follow-ups
        follow_ups.extend([
            "Can you explain the technical reasoning behind this solution?",
            "What documentation should I reference for more details?"
        ])
        
        return follow_ups[:4]  # Limit to 4 follow-ups
    
    def _create_context_summary(self, context_docs: List[Document]) -> str:
        """Create brief context summary for quick responses"""
        if not context_docs:
            return "No specific documentation found"
        
        doc_count = len(context_docs)
        categories = set()
        key_topics = set()
        
        for doc in context_docs[:3]:  # Check first 3 docs
            category = doc.metadata.get('main_category', 'Unknown')
            categories.add(category)
            
            # Extract key topics from content
            content_lower = doc.page_content.lower()
            if "export" in content_lower or "import" in content_lower:
                key_topics.add("import/export")
            if "bones" in content_lower or "skeleton" in content_lower:
                key_topics.add("rigging")
            if "skinning" in content_lower:
                key_topics.add("skinning")
            if "weapon" in content_lower or "rifle" in content_lower:
                key_topics.add("weapons")
        
        category_str = ', '.join(categories) if categories else "Unknown"
        topics_str = f" covering {', '.join(key_topics)}" if key_topics else ""
        
        return f"{doc_count} documents from {category_str}{topics_str}"

class AnalysisPerformanceTracker:
    """Track performance of dynamic analysis system"""
    
    def __init__(self):
        self.metrics = {
            "analyses_performed": 0,
            "confidence_scores": [],
            "evidence_strengths": [],
            "analysis_modes_used": {},
            "reasoning_quality_scores": [],
            "successful_solutions": 0
        }
    
    def record_analysis(self, analysis_result: Dict[str, Any]):
        """Record metrics from a dynamic analysis"""
        
        self.metrics["analyses_performed"] += 1
        
        # Track confidence and evidence
        self.metrics["confidence_scores"].append(analysis_result["analysis_confidence"])
        self.metrics["evidence_strengths"].append(analysis_result["evidence_strength"])
        self.metrics["reasoning_quality_scores"].append(analysis_result["reasoning_quality"])
        
        # Track analysis mode usage
        mode = analysis_result["analysis_mode"]
        self.metrics["analysis_modes_used"][mode] = self.metrics["analysis_modes_used"].get(mode, 0) + 1
        
        # Track successful solutions (high confidence + strong evidence)
        if analysis_result["analysis_confidence"] > 0.7 and analysis_result["evidence_strength"] > 0.6:
            self.metrics["successful_solutions"] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of dynamic analysis"""
        
        if not self.metrics["analyses_performed"]:
            return {"status": "No analyses performed yet"}
        
        avg_confidence = sum(self.metrics["confidence_scores"]) / len(self.metrics["confidence_scores"])
        avg_evidence = sum(self.metrics["evidence_strengths"]) / len(self.metrics["evidence_strengths"])
        avg_reasoning = sum(self.metrics["reasoning_quality_scores"]) / len(self.metrics["reasoning_quality_scores"])
        
        success_rate = self.metrics["successful_solutions"] / self.metrics["analyses_performed"]
        
        return {
            "total_analyses": self.metrics["analyses_performed"],
            "average_confidence": avg_confidence,
            "average_evidence_strength": avg_evidence,
            "average_reasoning_quality": avg_reasoning,
            "success_rate": success_rate,
            "modes_used": self.metrics["analysis_modes_used"],
            "performance_grade": self._calculate_performance_grade(avg_confidence, avg_evidence, success_rate)
        }
    
    def _calculate_performance_grade(self, avg_confidence: float, avg_evidence: float, success_rate: float) -> str:
        """Calculate overall performance grade"""
        
        overall_score = (avg_confidence + avg_evidence + success_rate) / 3
        
        if overall_score >= 0.8:
            return "Excellent"
        elif overall_score >= 0.7:
            return "Good"
        elif overall_score >= 0.6:
            return "Satisfactory"
        else:
            return "Needs Improvement"

# Test function for development
def test_dynamic_behavior():
    """Test dynamic AI behavior coordination"""
    
    print("🧪 TESTING DYNAMIC AI BEHAVIOR COORDINATION")
    print("-" * 60)
    
    # Mock document with realistic content
    class MockDoc:
        def __init__(self):
            self.page_content = "In case of skinned assets like rifles, Export Skinning option should be used instead. If for some reason you don't see bones icon on SampleWeapon_01.xob even after checking Export Skinning and reimporting resource, make sure that you have properly skinned your model in 3D software of your choice."
            self.metadata = {"filename": "Weapon Creation_Asset Preparation.pdf", "main_category": "Documentation"}
    
    context_docs = [MockDoc()]
    
    # Test different query types
    queries = [
        "can you explain this code",  # Should NOT use intelligent analysis
        "have imported my gun from blender, however i dont see any bones",  # Should use intelligent analysis
        "what does this function do",  # Should NOT use intelligent analysis
        "how do i fix this error"  # Should use intelligent analysis
    ]
    
    conversation_context = "No conversation history available."
    
    # Test dynamic coordination
    coordinator = DynamicAIBehaviorCoordinator()
    
    for query in queries:
        print(f"\n🔍 Testing query: '{query}'")
        should_analyze = coordinator.should_use_intelligent_analysis(query)
        print(f"   Should use intelligent analysis: {should_analyze}")
        
        if should_analyze:
            print("   → Will use problem-solving mode")
        else:
            print("   → Will use standard explanation mode")

if __name__ == "__main__":
    test_dynamic_behavior()
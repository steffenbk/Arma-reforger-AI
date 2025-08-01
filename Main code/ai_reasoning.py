"""
ai_reasoning.py - Dynamic AI Reasoning & Analysis (No Hardcoded Solutions)

Responsibilities:
- Dynamic document analysis and relationship extraction
- Pattern-based reasoning without hardcoded answers
- Evidence-based solution discovery
- Adaptive learning from document content
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ReasoningResult:
    """Result of dynamic AI reasoning analysis"""
    problem_analysis: Dict[str, Any]
    solution_candidates: List[Dict[str, Any]]
    reasoning_chain: List[str]
    confidence: float
    evidence_strength: float
    document_insights: List[str]
    
    @property
    def problem_type(self) -> str:
        """Compatibility property for accessing problem type"""
        return self.problem_analysis.get("primary_problem_type", "unknown")

class DynamicDocumentAnalyzer:
    """Analyze documents to find relationships and patterns dynamically"""
    
    def __init__(self):
        logger.info("🧠 DynamicDocumentAnalyzer initialized - No hardcoded solutions!")
        
        # Reasoning patterns for dynamic analysis
        self.reasoning_patterns = {
            "cause_effect": [
                "because", "since", "due to", "as a result", "therefore", "thus", 
                "consequently", "leads to", "causes", "results in"
            ],
            "conditional": [
                "if", "when", "unless", "provided that", "in case of", "should",
                "must", "need to", "requires", "depends on"
            ],
            "problem_solution": [
                "to fix", "to resolve", "solution is", "enable", "disable", "set to", 
                "configure", "adjust", "modify", "change", "use", "apply", "check"
            ],
            "requirements": [
                "must", "should", "required", "necessary", "essential", "important",
                "need", "have to", "ensure", "make sure", "verify"
            ],
            "alternatives": [
                "instead", "alternatively", "or", "either", "option", "can also",
                "another way", "different approach", "also possible"
            ],
            "negation_problems": [
                "not working", "missing", "don't see", "can't", "won't", "doesn't",
                "failed", "error", "issue", "problem", "broken", "incorrect"
            ]
        }
    
    def analyze_documents_dynamically(self, context_docs: list, user_query: str) -> Dict[str, Any]:
        """Dynamically analyze documents to understand relationships and find solutions"""
        
        logger.info(f"🔍 Analyzing {len(context_docs)} documents for query: '{user_query[:50]}...'")
        
        # Extract key concepts from user query
        query_concepts = self._extract_key_concepts(user_query)
        logger.info(f"📝 Key concepts extracted: {query_concepts}")
        
        # Analyze each document for relevant information
        document_analysis = {
            "problems_identified": [],
            "solutions_found": [],
            "conditions_discovered": [],
            "requirements_extracted": [],
            "alternatives_available": [],
            "concept_relationships": {}
        }
        
        for i, doc in enumerate(context_docs):
            logger.info(f"📄 Analyzing document {i+1}: {doc.metadata.get('filename', 'Unknown')}")
            
            doc_analysis = self._analyze_single_document(doc, query_concepts)
            
            # Merge results
            for key in document_analysis:
                if key == "concept_relationships":
                    document_analysis[key].update(doc_analysis[key])
                else:
                    document_analysis[key].extend(doc_analysis[key])
        
        logger.info(f"✅ Document analysis complete:")
        logger.info(f"   Problems: {len(document_analysis['problems_identified'])}")
        logger.info(f"   Solutions: {len(document_analysis['solutions_found'])}")
        logger.info(f"   Conditions: {len(document_analysis['conditions_discovered'])}")
        
        return document_analysis
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from user query using intelligent parsing"""
        
        # Remove common stop words
        stop_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", 
            "by", "from", "up", "about", "into", "through", "during", "before", "after", 
            "above", "below", "between", "among", "i", "my", "me", "we", "our", "you", "your", 
            "he", "she", "it", "they", "their", "is", "are", "was", "were", "be", "been", 
            "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", 
            "should", "may", "might", "can", "cant", "dont", "wont", "however", "when", "where"
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        key_concepts = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Group related concepts
        concept_groups = self._group_related_concepts(key_concepts)
        
        return key_concepts, concept_groups
    
    def _group_related_concepts(self, concepts: List[str]) -> Dict[str, List[str]]:
        """Group related concepts together"""
        
        concept_groups = {
            "import_export": [],
            "model_structure": [],
            "software_tools": [],
            "problems": [],
            "settings": []
        }
        
        # Categorize concepts
        for concept in concepts:
            if concept in ["import", "imported", "importing", "export", "exported", "exporting"]:
                concept_groups["import_export"].append(concept)
            elif concept in ["bones", "sockets", "armature", "skeleton", "mesh", "model", "fbx"]:
                concept_groups["model_structure"].append(concept)
            elif concept in ["blender", "enfusion", "workbench", "editor"]:
                concept_groups["software_tools"].append(concept)
            elif concept in ["missing", "broken", "error", "problem", "issue", "dont", "cant"]:
                concept_groups["problems"].append(concept)
            elif concept in ["settings", "options", "configuration", "skinning", "hierarchy"]:
                concept_groups["settings"].append(concept)
        
        return {k: v for k, v in concept_groups.items() if v}  # Remove empty groups
    
    def _analyze_single_document(self, doc, query_concepts) -> Dict[str, Any]:
        """Analyze a single document for relevant patterns"""
        
        content = doc.page_content.lower()
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        doc_analysis = {
            "problems_identified": [],
            "solutions_found": [],
            "conditions_discovered": [],
            "requirements_extracted": [],
            "alternatives_available": [],
            "concept_relationships": {}
        }
        
        key_concepts, concept_groups = query_concepts
        
        for sentence in sentences:
            # Calculate sentence relevance to query
            relevance = self._calculate_sentence_relevance(sentence, key_concepts)
            
            if relevance == 0:  # Skip irrelevant sentences
                continue
            
            # Look for different types of information
            self._extract_problems(sentence, relevance, doc_analysis["problems_identified"])
            self._extract_solutions(sentence, relevance, doc_analysis["solutions_found"])
            self._extract_conditions(sentence, relevance, doc_analysis["conditions_discovered"])
            self._extract_requirements(sentence, relevance, doc_analysis["requirements_extracted"])
            self._extract_alternatives(sentence, relevance, doc_analysis["alternatives_available"])
            
            # Build concept relationships
            self._build_concept_relationships(sentence, key_concepts, doc_analysis["concept_relationships"])
        
        return doc_analysis
    
    def _calculate_sentence_relevance(self, sentence: str, key_concepts: List[str]) -> float:
        """Calculate how relevant a sentence is to the user's query"""
        
        # Count direct concept matches
        direct_matches = sum(1 for concept in key_concepts if concept in sentence)
        
        # Boost for related terms
        related_boost = 0
        if any(concept in key_concepts for concept in ["bones", "sockets"]) and "skinning" in sentence:
            related_boost += 1
        if any(concept in key_concepts for concept in ["import", "export"]) and "settings" in sentence:
            related_boost += 1
        
        return direct_matches + related_boost
    
    def _extract_problems(self, sentence: str, relevance: float, problems_list: List):
        """Extract problem descriptions from sentence"""
        
        for pattern in self.reasoning_patterns["negation_problems"]:
            if pattern in sentence:
                problems_list.append({
                    "text": sentence,
                    "pattern": pattern,
                    "relevance": relevance,
                    "type": "problem_statement"
                })
                break
    
    def _extract_solutions(self, sentence: str, relevance: float, solutions_list: List):
        """Extract solution suggestions from sentence"""
        
        for pattern in self.reasoning_patterns["problem_solution"]:
            if pattern in sentence:
                # Look for specific actionable instructions
                action_strength = 1
                if any(word in sentence for word in ["must", "should", "need to"]):
                    action_strength = 2
                if any(word in sentence for word in ["enable", "disable", "set", "configure"]):
                    action_strength = 3
                
                solutions_list.append({
                    "text": sentence,
                    "pattern": pattern,
                    "relevance": relevance,
                    "action_strength": action_strength,
                    "type": "solution_instruction"
                })
                break
    
    def _extract_conditions(self, sentence: str, relevance: float, conditions_list: List):
        """Extract conditional statements that might contain solutions"""
        
        for pattern in self.reasoning_patterns["conditional"]:
            if pattern in sentence:
                conditions_list.append({
                    "text": sentence,
                    "condition": pattern,
                    "relevance": relevance,
                    "type": "conditional_logic"
                })
                break
    
    def _extract_requirements(self, sentence: str, relevance: float, requirements_list: List):
        """Extract requirements from sentence"""
        
        for pattern in self.reasoning_patterns["requirements"]:
            if pattern in sentence:
                requirements_list.append({
                    "text": sentence,
                    "requirement": pattern,
                    "relevance": relevance,
                    "type": "requirement"
                })
                break
    
    def _extract_alternatives(self, sentence: str, relevance: float, alternatives_list: List):
        """Extract alternative approaches"""
        
        for pattern in self.reasoning_patterns["alternatives"]:
            if pattern in sentence:
                alternatives_list.append({
                    "text": sentence,
                    "alternative": pattern,
                    "relevance": relevance,
                    "type": "alternative_approach"
                })
                break
    
    def _build_concept_relationships(self, sentence: str, key_concepts: List[str], relationships: Dict):
        """Build relationships between concepts found in the same sentence"""
        
        sentence_concepts = [concept for concept in key_concepts if concept in sentence]
        
        if len(sentence_concepts) > 1:
            # Create relationships between concepts that appear together
            for i, concept1 in enumerate(sentence_concepts):
                for concept2 in sentence_concepts[i+1:]:
                    relationship_key = f"{concept1}+{concept2}"
                    if relationship_key not in relationships:
                        relationships[relationship_key] = []
                    
                    relationships[relationship_key].append({
                        "context": sentence,
                        "strength": len(sentence_concepts)
                    })

class DynamicReasoningEngine:
    """Apply reasoning patterns to find solutions dynamically"""
    
    def __init__(self):
        logger.info("🔧 DynamicReasoningEngine initialized - Pure reasoning approach!")
        self.document_analyzer = DynamicDocumentAnalyzer()
    
    def reason_about_problem(self, query: str, context_docs: list) -> ReasoningResult:
        """Apply dynamic reasoning to find solutions without hardcoded answers"""
        
        logger.info(f"🧠 Starting dynamic reasoning for query: '{query[:50]}...'")
        
        # Step 1: Analyze documents dynamically
        document_analysis = self.document_analyzer.analyze_documents_dynamically(context_docs, query)
        
        # Step 2: Understand the user's problem
        problem_analysis = self._analyze_user_problem(query, document_analysis)
        
        # Step 3: Find and score solution candidates
        solution_candidates = self._find_solution_candidates(document_analysis, problem_analysis)
        
        # Step 4: Build reasoning chain
        reasoning_chain = self._build_reasoning_chain(problem_analysis, solution_candidates, document_analysis)
        
        # Step 5: Calculate confidence based on evidence
        confidence = self._calculate_reasoning_confidence(solution_candidates, document_analysis)
        
        # Step 6: Extract key insights
        document_insights = self._extract_document_insights(document_analysis, solution_candidates)
        
        result = ReasoningResult(
            problem_analysis=problem_analysis,
            solution_candidates=solution_candidates,
            reasoning_chain=reasoning_chain,
            confidence=confidence,
            evidence_strength=self._calculate_evidence_strength(document_analysis),
            document_insights=document_insights
        )
        
        logger.info(f"✅ Dynamic reasoning complete - Confidence: {confidence:.2f}")
        logger.info(f"🎯 Top solution: {solution_candidates[0]['text'][:50]}..." if solution_candidates else "No solutions found")
        
        return result
    
    def _analyze_user_problem(self, query: str, document_analysis: Dict) -> Dict[str, Any]:
        """Analyze what problem the user is actually facing"""
        
        query_lower = query.lower()
        
        problem_indicators = {
            "missing_elements": ["missing", "don't see", "not showing", "0", "zero", "empty"],
            "import_export_issue": ["import", "imported", "export", "exported"],
            "functionality_broken": ["not working", "broken", "failed", "error"],
            "configuration_needed": ["how to", "configure", "set up", "setup"]
        }
        
        detected_problems = {}
        for problem_type, indicators in problem_indicators.items():
            score = sum(1 for indicator in indicators if indicator in query_lower)
            if score > 0:
                detected_problems[problem_type] = score
        
        # Find most likely problem type
        primary_problem = max(detected_problems.items(), key=lambda x: x[1])[0] if detected_problems else "general_question"
        
        return {
            "primary_problem_type": primary_problem,
            "problem_indicators": detected_problems,
            "query_analysis": query_lower,
            "related_document_problems": [p for p in document_analysis["problems_identified"] if p["relevance"] > 0]
        }
    
    def _find_solution_candidates(self, document_analysis: Dict, problem_analysis: Dict) -> List[Dict[str, Any]]:
        """Find solution candidates from document analysis"""
        
        candidates = []
        
        # Primary solutions from documents
        for solution in document_analysis["solutions_found"]:
            score = solution["relevance"] * solution["action_strength"]
            
            candidates.append({
                "text": solution["text"],
                "score": score,
                "type": "direct_solution",
                "evidence": solution,
                "confidence": min(score / 10.0, 1.0)
            })
        
        # Conditional solutions
        for condition in document_analysis["conditions_discovered"]:
            if condition["relevance"] > 1:  # Only high-relevance conditions
                score = condition["relevance"] * 1.5  # Lower weight than direct solutions
                
                candidates.append({
                    "text": condition["text"],
                    "score": score,
                    "type": "conditional_solution",
                    "evidence": condition,
                    "confidence": min(score / 15.0, 0.8)  # Cap conditional confidence lower
                })
        
        # Requirements that might imply solutions
        for requirement in document_analysis["requirements_extracted"]:
            if requirement["relevance"] > 1:
                score = requirement["relevance"] * 1.2
                
                candidates.append({
                    "text": requirement["text"],
                    "score": score,
                    "type": "requirement_based",
                    "evidence": requirement,
                    "confidence": min(score / 12.0, 0.7)
                })
        
        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        
        return candidates
    
    def _build_reasoning_chain(self, problem_analysis: Dict, solution_candidates: List, document_analysis: Dict) -> List[str]:
        """Build a chain of reasoning steps"""
        
        reasoning_steps = []
        
        # Step 1: Problem identification
        reasoning_steps.append(f"Problem identified: {problem_analysis['primary_problem_type']}")
        
        if problem_analysis["related_document_problems"]:
            top_doc_problem = max(problem_analysis["related_document_problems"], key=lambda x: x["relevance"])
            reasoning_steps.append(f"Document evidence: '{top_doc_problem['text'][:60]}...'")
        
        # Step 2: Solution discovery
        if solution_candidates:
            top_solution = solution_candidates[0]
            reasoning_steps.append(f"Best solution found: {top_solution['type']}")
            reasoning_steps.append(f"Solution text: '{top_solution['text'][:60]}...'")
            reasoning_steps.append(f"Evidence strength: {top_solution['score']:.1f}")
        
        # Step 3: Supporting evidence
        if len(solution_candidates) > 1:
            reasoning_steps.append(f"Additional solutions available: {len(solution_candidates) - 1}")
        
        # Step 4: Concept relationships
        relationships = document_analysis["concept_relationships"]
        if relationships:
            key_relationship = list(relationships.keys())[0]
            reasoning_steps.append(f"Key concept relationship: {key_relationship}")
        
        return reasoning_steps
    
    def _calculate_reasoning_confidence(self, solution_candidates: List, document_analysis: Dict) -> float:
        """Calculate confidence based on reasoning strength"""
        
        if not solution_candidates:
            return 0.0
        
        # Base confidence from top solution
        base_confidence = solution_candidates[0]["confidence"]
        
        # Boost for multiple supporting solutions
        if len(solution_candidates) > 1:
            base_confidence += 0.1
        
        # Boost for strong document evidence
        total_solutions = len(document_analysis["solutions_found"])
        total_conditions = len(document_analysis["conditions_discovered"])
        
        evidence_boost = min((total_solutions + total_conditions) * 0.05, 0.2)
        base_confidence += evidence_boost
        
        # Boost for concept relationships
        if document_analysis["concept_relationships"]:
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _calculate_evidence_strength(self, document_analysis: Dict) -> float:
        """Calculate overall strength of evidence"""
        
        total_evidence = (
            len(document_analysis["solutions_found"]) * 3 +
            len(document_analysis["conditions_discovered"]) * 2 +
            len(document_analysis["requirements_extracted"]) * 1 +
            len(document_analysis["problems_identified"]) * 1
        )
        
        return min(total_evidence / 20.0, 1.0)  # Normalize to 0-1
    
    def _extract_document_insights(self, document_analysis: Dict, solution_candidates: List) -> List[str]:
        """Extract key insights from the analysis"""
        
        insights = []
        
        if solution_candidates:
            insights.append(f"Found {len(solution_candidates)} potential solutions in documentation")
        
        if document_analysis["concept_relationships"]:
            insights.append(f"Identified {len(document_analysis['concept_relationships'])} concept relationships")
        
        total_problems = len(document_analysis["problems_identified"])
        if total_problems > 0:
            insights.append(f"Matched {total_problems} problem patterns in documentation")
        
        return insights

# Test function for development
def test_dynamic_reasoning():
    """Test dynamic reasoning with real query"""
    
    print("🧪 TESTING DYNAMIC REASONING (NO HARDCODED SOLUTIONS)")
    print("-" * 60)
    
    # Mock document with the actual content
    class MockDoc:
        def __init__(self):
            self.page_content = "In case of skinned assets like rifles, Export Skinning option should be used instead. If for some reason you don't see bones icon on SampleWeapon_01.xob even after checking Export Skinning and reimporting resource, make sure that you have properly skinned your model in 3D software of your choice."
            self.metadata = {"filename": "Weapon Creation_Asset Preparation.pdf", "main_category": "Documentation"}
    
    context_docs = [MockDoc()]
    query = "have imported my gun from blender, however i dont see any bones or sockets it says 0 when i view it in enfusion"
    
    # Test dynamic reasoning
    reasoning_engine = DynamicReasoningEngine()
    result = reasoning_engine.reason_about_problem(query, context_docs)
    
    print(f"Problem Analysis: {result.problem_analysis['primary_problem_type']}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Evidence Strength: {result.evidence_strength:.2f}")
    
    if result.solution_candidates:
        print(f"\nTop Solution: {result.solution_candidates[0]['text']}")
        print(f"Solution Type: {result.solution_candidates[0]['type']}")
        print(f"Solution Score: {result.solution_candidates[0]['score']:.1f}")
    
    print(f"\nReasoning Chain:")
    for step in result.reasoning_chain:
        print(f"  • {step}")
    
    print(f"\nDocument Insights:")
    for insight in result.document_insights:
        print(f"  • {insight}")

if __name__ == "__main__":
    test_dynamic_reasoning()
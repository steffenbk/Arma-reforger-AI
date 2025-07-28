# =============================================================================
# FRESH SMART_PROCESSING.PY - UNIFORM NAMING VERSION 2.0
# =============================================================================
print("🚀🚀🚀 LOADING NEW SMART_PROCESSING.PY WITH UNIFORM NAMING! 🚀🚀🚀")

from typing import List, Dict, Any, Optional

class QueryClassifier:
    """Classify user queries into different types for better processing"""
    
    def __init__(self):
        print("✅ NEW QueryClassifier initialized!")
    
    def classify_query(self, query: str) -> str:
        query_lower = query.lower()
        
        # Reform queries (AI-powered question improvement)
        if "reform" in query_lower:
            return "ai_reform"
        
        # How-to questions
        elif any(phrase in query_lower for phrase in ["how to", "how do", "how can", "how should"]):
            return "how-to"
        
        # Definition questions
        elif any(phrase in query_lower for phrase in ["what is", "what are", "define", "explain what"]):
            return "definition"
        
        # Code example requests
        elif any(phrase in query_lower for phrase in ["show me", "example", "sample", "demonstrate", "code for"]):
            return "code-example"
        
        # Explanation questions
        elif any(query_lower.startswith(word) for word in ["why", "when", "where", "which"]):
            return "explanation"
        
        # Recommendation questions
        elif "?" in query and any(word in query_lower for word in ["better", "best", "recommend", "should i", "which one"]):
            return "recommendation"
        
        # Troubleshooting
        elif any(phrase in query_lower for phrase in ["error", "problem", "issue", "not working", "broken", "fix"]):
            return "troubleshooting"
        
        # Comparison
        elif any(phrase in query_lower for phrase in ["vs", "versus", "difference", "compare"]):
            return "comparison"
        
        else:
            return "general"
    
    def get_search_strategy(self, query_type: str, max_docs_from_config: int) -> Dict[str, Any]:
        """Get optimized search strategy based on query type"""
        strategies = {
            "how-to": {
                "preferred_categories": ["Documentation", "Source_Code"],
                "boost_keywords": ["tutorial", "guide", "step", "process"]
            },
            "code-example": {
                "preferred_categories": ["Source_Code", "API_Reference"],
                "boost_keywords": ["class", "function", "method", "example"]
            },
            "definition": {
                "preferred_categories": ["API_Reference", "Documentation"],
                "boost_keywords": ["definition", "interface", "class", "component"]
            },
            "troubleshooting": {
                "preferred_categories": ["Documentation", "Source_Code"],
                "boost_keywords": ["error", "fix", "solution", "problem"]
            },
            "ai_reform": {
                "preferred_categories": [],  # No document search - AI only
                "boost_keywords": []
            },
            "general": {
                "preferred_categories": None,
                "boost_keywords": []
            }
        }
        
        strategy = strategies.get(query_type, strategies["general"]).copy()
        strategy["max_docs"] = max_docs_from_config
        
        # Reform queries should have 0 docs
        if query_type == "ai_reform":
            strategy["max_docs"] = 0
        
        return strategy

class QueryProcessor:
    """Process and enhance user queries"""
    
    def __init__(self):
        print("✅ NEW QueryProcessor with UNIFORM NAMING initialized!")
        
        # Category prefixes - NEW UNIFORM NAMING SYSTEM!
        self.category_prefixes = {
            # Quick prefixes - Ultra-fast (NEW!)
            "quick_doc": ["Documentation"],
            "quick_code": ["Source_Code"],
            "quick_api": ["API_Reference"],
            "quick_all": ["Source_Code", "API_Reference", "Documentation"],
            
            # Standard prefixes - Uniform naming
            "standard_doc": ["Documentation"],
            "standard_code": ["Source_Code"],
            "standard_api": ["API_Reference"],
            "standard_code+api": ["Source_Code", "API_Reference"],
            "standard_all": ["Source_Code", "API_Reference", "Documentation"],
            
            # Force prefixes
            "force_doc": ["Documentation"],
            "force_code": ["Source_Code"],
            "force_api": ["API_Reference"],
            "force_code+api": ["Source_Code", "API_Reference"],
            "force_all": ["Source_Code", "API_Reference", "Documentation"],
            "force_benchmark": ["Source_Code", "API_Reference"],
            
            # Dynamic prefixes
            "dynamic_doc": ["Documentation"],
            "dynamic_code": ["Source_Code"],
            "dynamic_api": ["API_Reference"],
            "dynamic_code+api": ["Source_Code", "API_Reference"],
            "dynamic_all": ["Source_Code", "API_Reference", "Documentation"],
        }
        
        # LOUD DEBUG MESSAGE
        print(f"🚀🚀🚀 NEW UNIFORM PREFIXES LOADED: {list(self.category_prefixes.keys())}")
        print(f"🎯 Total prefixes: {len(self.category_prefixes)}")
        
        # Arma-specific terminology corrections
        self.arma_terms = {
            "weaponmanager": "WeaponManager",
            "scr_weapon": "SCR_WeaponComponent",
            "weaponcomponent": "SCR_WeaponComponent",
            "ai_agent": "SCR_AIAgent",
            "aiagent": "SCR_AIAgent",
            "vehiclecontroller": "VehicleController",
            "gamemode": "SCR_BaseGameMode",
            "playercontroller": "SCR_PlayerController",
            "inventorymanager": "SCR_InventoryStorageManagerComponent",
            "damagemanager": "SCR_DamageManagerComponent",
            "enfusion": "Enfusion Script",
            "workbench": "Arma Reforger Tools",
            "reforger": "Arma Reforger"
        }
        
        # Common abbreviations
        self.abbreviations = {
            "ui": "user interface",
            "ai": "artificial intelligence",
            "scr": "script",
            "api": "application programming interface",
            "gui": "graphical user interface",
            "hud": "heads up display",
            "npc": "non-player character"
        }
        
        # Synonym expansions for better search
        self.synonyms = {
            "create": ["make", "build", "implement", "develop", "generate"],
            "modify": ["change", "edit", "alter", "update", "customize"],
            "setup": ["configure", "initialize", "install", "prepare"],
            "use": ["utilize", "employ", "apply", "work with"],
            "issue": ["problem", "error", "bug", "trouble"]
        }
        
        # Intent detection patterns
        self.intent_patterns = {
            "expand_request": [
                "expand on that", "tell me more", "more details", "elaborate",
                "go deeper", "more about", "explain further", "continue"
            ],
            "follow_up": [
                "what about", "how about", "what if", "also", "and",
                "what's next", "then what", "after that"
            ],
            "clarification": [
                "what do you mean", "clarify", "explain that", "i don't understand",
                "what is", "what does that mean", "confused"
            ],
            "alternative": [
                "another way", "different approach", "alternative", "other method",
                "else", "instead", "better way"
            ],
            "reform_request": [
                "reform", "improve", "better question", "rephrase", "fix question"
            ]
        }
    
    def detect_intent(self, query: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Detect user intent and provide context"""
        query_lower = query.lower().strip()
        
        intent_result = {
            "intent_type": "direct_question",
            "needs_context": False,
            "reference_last": False,
            "context_hint": "",
            "validation_issues": []
        }
        
        # Check for reform requests first
        for pattern in self.intent_patterns["reform_request"]:
            if pattern in query_lower:
                intent_result.update({
                    "intent_type": "ai_reform_request",
                    "needs_context": False,
                    "reference_last": False,
                    "context_hint": "User wants AI-powered question improvement"
                })
                break
        
        # Check for expansion requests
        if intent_result["intent_type"] == "direct_question":
            for pattern in self.intent_patterns["expand_request"]:
                if pattern in query_lower:
                    intent_result.update({
                        "intent_type": "expand_request",
                        "needs_context": True,
                        "reference_last": True,
                        "context_hint": "User wants more details about the previous topic"
                    })
                    break
        
        # Check for follow-up questions
        if intent_result["intent_type"] == "direct_question":
            for pattern in self.intent_patterns["follow_up"]:
                if query_lower.startswith(pattern):
                    intent_result.update({
                        "intent_type": "follow_up",
                        "needs_context": True,
                        "reference_last": True,
                        "context_hint": f"User is asking a follow-up about previous discussion"
                    })
                    break
        
        # Check for clarification requests
        if intent_result["intent_type"] == "direct_question":
            for pattern in self.intent_patterns["clarification"]:
                if pattern in query_lower:
                    intent_result.update({
                        "intent_type": "clarification",
                        "needs_context": True,
                        "reference_last": True,
                        "context_hint": "User needs clarification about previous response"
                    })
                    break
        
        # Check for alternative requests
        if intent_result["intent_type"] == "direct_question":
            for pattern in self.intent_patterns["alternative"]:
                if pattern in query_lower:
                    intent_result.update({
                        "intent_type": "alternative",
                        "needs_context": True,
                        "reference_last": True,
                        "context_hint": "User wants alternative approaches to previous topic"
                    })
                    break
        
        # Query validation (skip for reform requests)
        if intent_result["intent_type"] != "ai_reform_request":
            validation_issues = self._validate_query(query, conversation_history)
            if validation_issues:
                intent_result["validation_issues"] = validation_issues
        
        return intent_result
    
    def _validate_query(self, query: str, conversation_history: List[Dict]) -> List[str]:
        """Validate query and suggest improvements"""
        issues = []
        query_words = query.strip().split()
        
        # Too short
        if len(query_words) <= 2:
            issues.append("Query seems very short. Could you provide more details?")
        
        # Too vague
        vague_patterns = ["this", "that", "it", "stuff", "thing", "something"]
        if any(pattern in query.lower() for pattern in vague_patterns) and len(conversation_history) == 0:
            issues.append("Your question contains vague references. Could you be more specific?")
        
        # No question structure
        question_indicators = ["how", "what", "why", "when", "where", "which", "can", "should", "do", "is", "are", "?"]
        if not any(indicator in query.lower() for indicator in question_indicators):
            issues.append("This doesn't seem like a question. Try asking 'How do I...' or 'What is...'")
        
        # Completely unrelated to Arma
        arma_keywords = [
            "arma", "reforger", "enfusion", "weapon", "vehicle", "ai", "script", "mod", "modding",
            "scr_", "component", "class", "workbench", "editor", "game", "military", "soldier"
        ]
        if not any(keyword in query.lower() for keyword in arma_keywords) and len(query_words) > 5:
            issues.append("This seems unrelated to Arma Reforger modding. I specialize in Arma Reforger assistance.")
        
        return issues
    
    def enhance_context_query(self, query: str, last_assistant_message: str, intent_result: Dict) -> str:
        """Enhance query with context from previous conversation"""
        
        if not intent_result["needs_context"] or not last_assistant_message:
            return query
        
        # Extract key topics from last response
        last_response_lower = last_assistant_message.lower()
        
        # Find Arma-specific terms mentioned
        mentioned_terms = []
        for term in ["weapon", "vehicle", "ai", "component", "script", "class"]:
            if term in last_response_lower:
                mentioned_terms.append(term)
        
        # Find SCR_ classes mentioned
        import re
        scr_classes = re.findall(r'SCR_\w+', last_assistant_message)
        
        # Enhance query based on intent
        if intent_result["intent_type"] == "expand_request":
            if mentioned_terms:
                return f"{query} (about {', '.join(mentioned_terms[:2])} from previous discussion)"
            elif scr_classes:
                return f"{query} (about {scr_classes[0]} from previous discussion)"
        
        elif intent_result["intent_type"] == "follow_up":
            if mentioned_terms:
                return f"{query} regarding {mentioned_terms[0]} (following up on previous discussion)"
        
        elif intent_result["intent_type"] == "clarification":
            return f"{query} (referring to previous explanation about {mentioned_terms[0] if mentioned_terms else 'the topic discussed'})"
        
        elif intent_result["intent_type"] == "alternative":
            if mentioned_terms:
                return f"{query} for {mentioned_terms[0]} (alternative to previous suggestion)"
        
        return query
    
    def detect_category_prefix(self, query: str) -> tuple[str, Optional[List[str]], str]:
        """Detect category prefix and return cleaned query + forced categories + prefix used"""
        query_stripped = query.strip()
        query_lower = query_stripped.lower()
        
        # LOUD DEBUG
        print(f"🚀🚀🚀 NEW PREFIX DETECTION: Checking '{query_stripped}'")
        print(f"🎯🎯🎯 NEW AVAILABLE PREFIXES: {list(self.category_prefixes.keys())}")
        
        for prefix, categories in self.category_prefixes.items():
            # Check for "prefix " or "prefix," at start of query
            if query_lower.startswith(f"{prefix} ") or query_lower.startswith(f"{prefix},"):
                # Remove prefix from query
                cleaned_query = query_stripped[len(prefix):].strip().lstrip(",").strip()
                print(f"✅✅✅ NEW PREFIX FOUND: '{prefix}' → '{cleaned_query}'")
                return cleaned_query, categories, prefix
        
        print(f"❌❌❌ NEW PREFIX DETECTION: No prefix found")
        return query, None, ""  # No prefix detected
    
    def correct_and_expand(self, query: str) -> tuple[str, List[str]]:
        """Correct typos and expand query for better search"""
        corrections_made = []
        processed = query
        
        # Fix Arma-specific terms
        for wrong, correct in self.arma_terms.items():
            if wrong.lower() in processed.lower():
                processed = processed.replace(wrong.lower(), correct)
                processed = processed.replace(wrong.title(), correct)
                corrections_made.append(f"'{wrong}' → '{correct}'")
        
        # Expand abbreviations
        words = processed.split()
        for i, word in enumerate(words):
            clean_word = word.lower().strip(".,!?")
            if clean_word in self.abbreviations:
                words[i] = word.replace(clean_word, self.abbreviations[clean_word])
                corrections_made.append(f"'{clean_word}' → '{self.abbreviations[clean_word]}'")
        
        processed = " ".join(words)
        
        # Add synonyms for key terms to improve search
        for term, synonyms in self.synonyms.items():
            if term in processed.lower():
                # Add synonyms to the end for search expansion
                processed += " " + " ".join(synonyms)
        
        return processed, corrections_made

class MultiPartHandler:
    """Handle multi-part questions"""
    
    def detect_multi_part(self, query: str) -> bool:
        """Detect if query contains multiple questions"""
        indicators = [
            " and ", " also ", " additionally ", " furthermore ",
            "1.", "2.", "3.", "first", "second", "third",
            "?", "\n"
        ]
        
        # Count question indicators
        question_count = 0
        for indicator in ["?", " and ", " also "]:
            question_count += query.count(indicator)
        
        # Check for numbered lists
        if any(f"{i}." in query for i in range(1, 6)):
            return True
        
        # Check for multiple sentences with question words
        sentences = query.split(". ")
        question_words = ["how", "what", "why", "when", "where", "which", "can", "should", "do"]
        question_sentences = 0
        
        for sentence in sentences:
            if any(sentence.lower().strip().startswith(word) for word in question_words):
                question_sentences += 1
        
        return question_sentences > 1 or question_count > 1
    
    def split_questions(self, query: str) -> List[str]:
        """Split multi-part query into individual questions"""
        # First try splitting on numbered lists
        if any(f"{i}." in query for i in range(1, 6)):
            parts = []
            for i in range(1, 10):
                if f"{i}." in query:
                    if parts:
                        # Split the previous part at this number
                        before_parts = parts[-1].split(f"{i}.")
                        parts[-1] = before_parts[0].strip()
                        if len(before_parts) > 1:
                            parts.append(f"{i}.{before_parts[1]}")
                    else:
                        parts = query.split(f"{i}.")
                        if len(parts) > 1:
                            parts[1] = f"{i}.{parts[1]}"
            
            if len(parts) > 1:
                return [part.strip() for part in parts if len(part.strip()) > 5]
        
        # Try splitting on common separators
        separators = [" and ", " also ", " additionally ", " furthermore "]
        
        parts = [query]
        for sep in separators:
            new_parts = []
            for part in parts:
                if sep in part:
                    split_parts = part.split(sep)
                    new_parts.extend(split_parts)
                else:
                    new_parts.append(part)
            parts = new_parts
        
        # Split on sentence boundaries if they look like questions
        if len(parts) == 1:
            sentences = query.split(". ")
            question_words = ["how", "what", "why", "when", "where", "which", "can", "should"]
            question_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and any(sentence.lower().strip().startswith(word) for word in question_words):
                    question_sentences.append(sentence)
            
            if len(question_sentences) > 1:
                parts = question_sentences
        
        # Clean and filter
        clean_parts = []
        for part in parts:
            clean_part = part.strip().strip(".,!?")
            if len(clean_part) > 10:  # Minimum question length
                clean_parts.append(clean_part)
        
        return clean_parts if len(clean_parts) > 1 else [query]

class SuggestionEngine:
    """Generate follow-up suggestions based on conversation context"""
    
    def __init__(self):
        self.topic_suggestions = {
            "weapon": [
                "How do I customize weapon properties?",
                "Show me weapon attachment examples",
                "What about weapon animations?",
                "How does weapon damage work?"
            ],
            "ai": [
                "How does AI decision making work?",
                "Show me AI behavior tree examples", 
                "What about AI pathfinding?",
                "How to create custom AI behaviors?"
            ],
            "vehicle": [
                "How do vehicle physics work?",
                "Show me vehicle customization examples",
                "What about vehicle damage systems?",
                "How to create custom vehicles?"
            ],
            "component": [
                "How do components communicate?",
                "Show me component inheritance examples",
                "What about component lifecycle?",
                "How to create custom components?"
            ],
            "script": [
                "What are scripting best practices?",
                "Show me advanced scripting examples",
                "How does script debugging work?",
                "What about script performance?"
            ],
            "tutorial": [
                "Are there more advanced tutorials?",
                "What about video tutorials?",
                "Show me step-by-step guides",
                "What should I learn next?"
            ],
            "reform": [
                "How do I ask better questions?",
                "What makes a good Arma Reforger question?",
                "Can you help me with question structure?",
                "How to be more specific in my questions?"
            ]
        }
    
    def extract_topics(self, messages: List[Dict]) -> List[str]:
        """Extract topics from recent conversation"""
        topics = set()
        keywords = ["weapon", "ai", "vehicle", "component", "script", "class", "function", "tutorial", "guide", "reform"]
        
        # Look at recent messages (last 6 messages = 3 exchanges)
        recent_messages = messages[-6:] if len(messages) > 6 else messages
        
        for msg in recent_messages:
            content_lower = msg["content"].lower()
            for keyword in keywords:
                if keyword in content_lower:
                    topics.add(keyword)
        
        return list(topics)
    
    def generate_follow_ups(self, conversation_history: List[Dict], query_type: str, last_answer: str = "") -> List[str]:
        """Generate contextual follow-up suggestions"""
        suggestions = []
        
        # Extract topics from conversation
        topics = self.extract_topics(conversation_history)
        
        # Add topic-specific suggestions
        for topic in topics[:2]:  # Limit to top 2 topics
            if topic in self.topic_suggestions:
                suggestions.extend(self.topic_suggestions[topic][:2])
        
        # Add query-type specific suggestions
        type_suggestions = {
            "how-to": [
                "Can you show me an example?",
                "What are the common pitfalls?",
                "Are there alternative approaches?"
            ],
            "code-example": [
                "Can you explain this code?",
                "How can I modify this?",
                "What are the best practices?"
            ],
            "definition": [
                "How do I use this?",
                "Can you show examples?",
                "What are related concepts?"
            ],
            "ai_reform": [
                "Try one of the improved questions above",
                "How do I ask better technical questions?",
                "What makes a good Arma Reforger question?",
                "Can you help me be more specific?"
            ]
        }
        
        if query_type in type_suggestions:
            suggestions.extend(type_suggestions[query_type][:1])
        
        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:4]  # Return top 4 suggestions

print("🎉🎉🎉 NEW SMART_PROCESSING.PY FULLY LOADED WITH UNIFORM NAMING! 🎉🎉🎉")
"""
Expanded Vocabulary and Grammar definitions for the LLM.
Contains word categories, synonyms, registers (formal/informal/academic/slang),
and sentence structure rules.
"""


class Register:
    """Word register/style classifications."""
    FORMAL = "formal"
    INFORMAL = "informal"
    ACADEMIC = "academic"
    SLANG = "slang"
    NEUTRAL = "neutral"
    TECHNICAL = "technical"
    LITERARY = "literary"


class Vocabulary:
    """
    Manages vocabulary with word-to-index mappings, category information,
    synonyms, and register classifications.
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    # ==========================================================================
    # EXPANDED WORD CATEGORIES
    # ==========================================================================
    
    CATEGORIES = {
        # ----------------------------------------------------------------------
        # PRONOUNS - Personal, demonstrative, interrogative, reflexive
        # ----------------------------------------------------------------------
        "pronoun": [
            # Personal pronouns
            "i", "you", "he", "she", "it", "we", "they",
            # Demonstrative
            "this", "that", "these", "those",
            # Interrogative
            "who", "whom", "what", "which", "whose",
            # Indefinite
            "someone", "anyone", "everyone", "nobody", "somebody", "anybody",
            "something", "anything", "everything", "nothing",
            "one", "each", "either", "neither", "both", "all", "none",
            # Reflexive
            "myself", "yourself", "himself", "herself", "itself", "ourselves", "themselves",
        ],
        
        # ----------------------------------------------------------------------
        # VERBS - Action, state, mental, communication, movement
        # ----------------------------------------------------------------------
        "verb": [
            # Original verbs
            "run", "drive", "exist", "become", "have", "do", "make", "go", "see", "believe",
            # Movement verbs
            "walk", "move", "travel", "fly", "swim", "jump", "climb", "fall", "rise", "turn",
            "arrive", "depart", "enter", "exit", "approach", "retreat", "advance", "proceed",
            # Communication verbs
            "say", "tell", "speak", "talk", "ask", "answer", "reply", "explain", "describe",
            "announce", "declare", "state", "mention", "whisper", "shout", "yell", "scream",
            # Mental verbs
            "think", "know", "understand", "remember", "forget", "learn", "teach", "study",
            "consider", "imagine", "wonder", "doubt", "realize", "recognize", "notice",
            # Action verbs
            "take", "give", "get", "put", "bring", "carry", "hold", "keep", "leave", "send",
            "write", "read", "show", "build", "create", "destroy", "break", "fix", "repair",
            "open", "close", "start", "stop", "begin", "end", "finish", "complete",
            # State verbs
            "be", "seem", "appear", "look", "feel", "sound", "taste", "smell", "remain", "stay",
            # Possession verbs
            "own", "possess", "belong", "contain", "include", "consist", "comprise",
            # Emotion verbs
            "love", "hate", "like", "enjoy", "prefer", "want", "need", "wish", "hope", "fear",
            # Work verbs
            "work", "manage", "lead", "follow", "help", "support", "assist", "serve", "perform",
            # Change verbs
            "change", "develop", "grow", "improve", "increase", "decrease", "reduce", "expand",
            "transform", "evolve", "adapt", "modify", "adjust", "alter", "vary",
            # Technical verbs
            "analyze", "compute", "calculate", "process", "optimize", "implement", "deploy",
            "configure", "integrate", "synchronize", "validate", "verify", "test", "debug",
        ],
        
        # ----------------------------------------------------------------------
        # ADJECTIVES - Descriptive, evaluative, quantitative
        # ----------------------------------------------------------------------
        "adjective": [
            # Original adjectives
            "wealthy", "fast", "sustainable", "critical", "urban", "resilient",
            "transparent", "equitable", "dynamic", "immersive",
            # Size
            "big", "large", "huge", "enormous", "massive", "giant", "vast",
            "small", "little", "tiny", "miniature", "compact", "slim",
            # Quality
            "good", "great", "excellent", "outstanding", "superb", "wonderful", "fantastic",
            "bad", "poor", "terrible", "awful", "horrible", "dreadful",
            # Age/Time
            "new", "old", "young", "ancient", "modern", "contemporary", "recent", "current",
            # Appearance
            "beautiful", "pretty", "handsome", "elegant", "gorgeous", "attractive",
            "ugly", "plain", "ordinary", "simple", "complex", "complicated",
            # Physical properties
            "hot", "cold", "warm", "cool", "wet", "dry", "hard", "soft", "rough", "smooth",
            "heavy", "light", "thick", "thin", "deep", "shallow", "wide", "narrow",
            # Colors
            "red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray",
            # Emotions/Feelings
            "happy", "sad", "angry", "excited", "calm", "nervous", "confident", "afraid",
            "proud", "ashamed", "grateful", "jealous", "curious", "bored", "interested",
            # Intelligence/Ability
            "smart", "intelligent", "clever", "brilliant", "wise", "stupid", "foolish",
            "skilled", "talented", "capable", "competent", "expert", "professional",
            # Speed/Pace
            "quick", "rapid", "swift", "slow", "gradual", "steady", "constant",
            # Importance
            "important", "significant", "essential", "vital", "crucial", "key", "major",
            "minor", "trivial", "irrelevant", "unnecessary",
            # Technical/Academic
            "technical", "scientific", "academic", "theoretical", "practical", "empirical",
            "quantitative", "qualitative", "systematic", "comprehensive", "extensive",
            "innovative", "revolutionary", "groundbreaking", "cutting-edge", "state-of-the-art",
            # Business/Formal
            "strategic", "operational", "tactical", "efficient", "effective", "productive",
            "profitable", "viable", "feasible", "scalable", "robust", "flexible", "agile",
            # Informal/Slang
            "cool", "awesome", "amazing", "incredible", "crazy", "insane", "wild", "sick",
            "dope", "lit", "epic", "legendary", "iconic",
        ],
        
        # ----------------------------------------------------------------------
        # NOUNS - People, places, things, concepts
        # ----------------------------------------------------------------------
        "noun": [
            # Original nouns
            "city", "data", "risk", "citizen", "car", "model", "twin", "governance",
            "algorithm", "resilience",
            # People
            "person", "people", "man", "woman", "child", "adult", "student", "teacher",
            "doctor", "engineer", "scientist", "artist", "writer", "developer", "designer",
            "manager", "leader", "worker", "employee", "boss", "client", "customer",
            "friend", "family", "parent", "colleague", "partner", "team", "group", "community",
            # Places
            "place", "country", "state", "region", "area", "district", "neighborhood",
            "building", "house", "home", "office", "school", "university", "hospital",
            "store", "shop", "restaurant", "hotel", "airport", "station", "park",
            "street", "road", "highway", "bridge", "town", "village", "suburb",
            # Technology
            "computer", "phone", "device", "machine", "system", "network", "internet",
            "software", "hardware", "application", "program", "code", "database", "server",
            "platform", "interface", "framework", "technology", "innovation", "solution",
            # Business
            "company", "business", "organization", "corporation", "enterprise", "startup",
            "market", "industry", "sector", "economy", "finance", "investment", "revenue",
            "product", "service", "brand", "strategy", "project", "process", "operation",
            # Abstract concepts
            "idea", "concept", "theory", "principle", "method", "approach", "technique",
            "problem", "solution", "challenge", "opportunity", "goal", "objective", "target",
            "success", "failure", "progress", "development", "growth", "change", "trend",
            "future", "past", "present", "time", "moment", "period", "era", "century",
            # Education/Knowledge
            "knowledge", "information", "fact", "truth", "evidence", "research", "study",
            "analysis", "report", "document", "paper", "article", "book", "chapter",
            "lesson", "course", "degree", "skill", "ability", "experience", "expertise",
            # Nature
            "world", "earth", "nature", "environment", "climate", "weather", "water",
            "air", "fire", "land", "ocean", "sea", "river", "mountain", "forest", "tree",
            # Objects
            "thing", "object", "item", "piece", "part", "component", "element", "material",
            "tool", "equipment", "resource", "asset", "property", "value", "quality",
            # Events
            "event", "meeting", "conference", "session", "presentation", "discussion",
            "decision", "action", "activity", "task", "work", "job", "effort", "result",
        ],
        
        # ----------------------------------------------------------------------
        # PARTICLES - Phrasal verb particles
        # ----------------------------------------------------------------------
        "particle": [
            "to", "up", "out", "off", "down", "over", "away", "in", "back", "on",
            "through", "around", "about", "along", "across", "forward", "behind",
            "apart", "aside", "together", "ahead", "by",
        ],
        
        # ----------------------------------------------------------------------
        # CONJUNCTIONS - Coordinating, subordinating, correlative
        # ----------------------------------------------------------------------
        "conjunction": [
            # Coordinating
            "and", "but", "or", "nor", "so", "yet", "for",
            # Subordinating
            "because", "although", "when", "if", "while", "since", "unless",
            "after", "before", "until", "once", "whenever", "wherever", "whereas",
            "whether", "though", "even", "as", "than",
            # Informal connectors
            "plus", "besides", "however", "therefore", "thus", "hence", "moreover",
            "furthermore", "nevertheless", "nonetheless", "meanwhile", "otherwise",
        ],
        
        # ----------------------------------------------------------------------
        # DETERMINERS - Articles, demonstratives, quantifiers
        # ----------------------------------------------------------------------
        "determiner": [
            "the", "a", "an",
            "this", "that", "these", "those",
            "my", "your", "his", "her", "its", "our", "their",
            "some", "any", "no", "every", "each", "either", "neither",
            "much", "many", "more", "most", "few", "little", "several",
            "all", "both", "half", "enough", "another", "other", "such",
        ],
        
        # ----------------------------------------------------------------------
        # AUXILIARIES - Helping verbs for tenses
        # ----------------------------------------------------------------------
        "auxiliary": [
            "is", "are", "am", "was", "were", "been", "being",
            "has", "have", "had", "having",
            "do", "does", "did",
            "will", "would", "shall", "should",
            "can", "could", "may", "might", "must",
        ],
        
        # ----------------------------------------------------------------------
        # ADVERBS - Manner, time, place, degree, frequency
        # ----------------------------------------------------------------------
        "adverb": [
            # Manner
            "quickly", "slowly", "carefully", "easily", "hardly", "simply",
            "quietly", "loudly", "softly", "strongly", "gently", "firmly",
            "clearly", "obviously", "apparently", "certainly", "definitely",
            # Time
            "now", "then", "soon", "later", "before", "after", "already", "still",
            "yet", "recently", "finally", "eventually", "immediately", "suddenly",
            "always", "never", "often", "sometimes", "usually", "rarely", "seldom",
            # Place
            "here", "there", "everywhere", "somewhere", "nowhere", "anywhere",
            "inside", "outside", "upstairs", "downstairs", "nearby", "far",
            # Degree
            "very", "really", "quite", "rather", "fairly", "pretty", "extremely",
            "absolutely", "completely", "totally", "entirely", "fully", "partly",
            "almost", "nearly", "barely", "hardly", "just", "only", "even",
            # Frequency
            "daily", "weekly", "monthly", "yearly", "annually", "constantly",
            "frequently", "occasionally", "periodically", "regularly",
        ],
        
        # ----------------------------------------------------------------------
        # PREPOSITIONS - Location, time, direction
        # ----------------------------------------------------------------------
        "preposition": [
            "in", "on", "at", "to", "from", "by", "with", "without",
            "for", "about", "of", "into", "onto", "out", "off",
            "up", "down", "over", "under", "above", "below", "between", "among",
            "through", "across", "around", "along", "behind", "before", "after",
            "during", "until", "since", "toward", "towards", "against", "within",
        ],
        
        # ----------------------------------------------------------------------
        # INTERJECTIONS - Exclamations, fillers
        # ----------------------------------------------------------------------
        "interjection": [
            "oh", "ah", "wow", "hey", "hi", "hello", "well", "okay", "ok",
            "yes", "no", "yeah", "nah", "uh", "um", "hmm", "oops", "ouch",
            "please", "thanks", "sorry", "exactly", "indeed", "absolutely",
        ],
    }
    
    # ==========================================================================
    # SYNONYM SYSTEM WITH REGISTERS
    # ==========================================================================
    
    SYNONYMS = {
        # Verb synonyms with register information
        "run": {
            Register.NEUTRAL: ["run", "jog"],
            Register.FORMAL: ["proceed", "execute", "operate"],
            Register.INFORMAL: ["dash", "sprint", "bolt"],
            Register.SLANG: ["book it", "zoom"],
        },
        "go": {
            Register.NEUTRAL: ["go", "move", "travel"],
            Register.FORMAL: ["proceed", "advance", "depart"],
            Register.INFORMAL: ["head", "leave", "take off"],
            Register.SLANG: ["bounce", "dip", "jet"],
        },
        "see": {
            Register.NEUTRAL: ["see", "look", "watch"],
            Register.FORMAL: ["observe", "perceive", "witness"],
            Register.ACADEMIC: ["examine", "analyze", "scrutinize"],
            Register.INFORMAL: ["spot", "catch", "check out"],
        },
        "think": {
            Register.NEUTRAL: ["think", "consider"],
            Register.FORMAL: ["contemplate", "deliberate", "ponder"],
            Register.ACADEMIC: ["hypothesize", "theorize", "postulate"],
            Register.INFORMAL: ["figure", "reckon", "guess"],
            Register.SLANG: ["vibe", "feel"],
        },
        "say": {
            Register.NEUTRAL: ["say", "tell", "speak"],
            Register.FORMAL: ["state", "declare", "articulate"],
            Register.INFORMAL: ["chat", "mention", "go"],
            Register.SLANG: ["spill", "drop"],
        },
        "make": {
            Register.NEUTRAL: ["make", "create", "build"],
            Register.FORMAL: ["construct", "fabricate", "manufacture"],
            Register.ACADEMIC: ["synthesize", "formulate", "generate"],
            Register.INFORMAL: ["put together", "whip up"],
        },
        "get": {
            Register.NEUTRAL: ["get", "obtain", "receive"],
            Register.FORMAL: ["acquire", "procure", "attain"],
            Register.INFORMAL: ["grab", "snag", "score"],
            Register.SLANG: ["cop", "snatch"],
        },
        "understand": {
            Register.NEUTRAL: ["understand", "know", "realize"],
            Register.FORMAL: ["comprehend", "grasp", "appreciate"],
            Register.ACADEMIC: ["discern", "cognize", "apprehend"],
            Register.INFORMAL: ["get", "catch", "follow"],
            Register.SLANG: ["vibe with", "feel"],
        },
        "help": {
            Register.NEUTRAL: ["help", "assist", "support"],
            Register.FORMAL: ["facilitate", "aid", "accommodate"],
            Register.INFORMAL: ["give a hand", "pitch in"],
            Register.SLANG: ["hook up", "back up"],
        },
        "want": {
            Register.NEUTRAL: ["want", "need", "wish"],
            Register.FORMAL: ["desire", "require", "seek"],
            Register.INFORMAL: ["wanna", "gotta have"],
            Register.SLANG: ["crave", "be dying for"],
        },
        
        # Adjective synonyms
        "good": {
            Register.NEUTRAL: ["good", "nice", "fine"],
            Register.FORMAL: ["excellent", "outstanding", "superb"],
            Register.ACADEMIC: ["optimal", "exemplary", "superior"],
            Register.INFORMAL: ["great", "awesome", "cool"],
            Register.SLANG: ["sick", "dope", "fire", "lit"],
        },
        "bad": {
            Register.NEUTRAL: ["bad", "poor"],
            Register.FORMAL: ["inadequate", "unsatisfactory", "deficient"],
            Register.INFORMAL: ["terrible", "awful", "lousy"],
            Register.SLANG: ["trash", "garbage", "weak", "mid"],
        },
        "big": {
            Register.NEUTRAL: ["big", "large"],
            Register.FORMAL: ["substantial", "considerable", "significant"],
            Register.ACADEMIC: ["extensive", "comprehensive", "voluminous"],
            Register.INFORMAL: ["huge", "massive", "giant"],
            Register.SLANG: ["ginormous", "mega"],
        },
        "small": {
            Register.NEUTRAL: ["small", "little"],
            Register.FORMAL: ["minimal", "minor", "modest"],
            Register.ACADEMIC: ["negligible", "marginal", "diminutive"],
            Register.INFORMAL: ["tiny", "mini", "teeny"],
        },
        "fast": {
            Register.NEUTRAL: ["fast", "quick"],
            Register.FORMAL: ["rapid", "swift", "expeditious"],
            Register.ACADEMIC: ["accelerated"],
            Register.INFORMAL: ["speedy", "zippy", "snappy"],
            Register.SLANG: ["blazing", "insane"],
        },
        "important": {
            Register.NEUTRAL: ["important", "significant"],
            Register.FORMAL: ["crucial", "vital", "essential"],
            Register.ACADEMIC: ["paramount", "pivotal", "fundamental"],
            Register.INFORMAL: ["big", "major", "key"],
            Register.SLANG: ["huge", "massive"],
        },
        "smart": {
            Register.NEUTRAL: ["smart", "intelligent"],
            Register.FORMAL: ["astute", "sagacious", "perspicacious"],
            Register.ACADEMIC: ["erudite", "scholarly", "intellectual"],
            Register.INFORMAL: ["clever", "bright", "sharp"],
            Register.SLANG: ["brainy", "genius"],
        },
        "happy": {
            Register.NEUTRAL: ["happy", "glad", "pleased"],
            Register.FORMAL: ["delighted", "gratified", "content"],
            Register.LITERARY: ["elated", "jubilant", "euphoric"],
            Register.INFORMAL: ["thrilled", "pumped", "psyched"],
            Register.SLANG: ["stoked", "hyped", "vibing"],
        },
        "sad": {
            Register.NEUTRAL: ["sad", "unhappy"],
            Register.FORMAL: ["melancholic", "despondent", "sorrowful"],
            Register.LITERARY: ["woeful", "dejected", "forlorn"],
            Register.INFORMAL: ["down", "bummed", "blue"],
            Register.SLANG: ["gutted", "bumming"],
        },
        
        # Noun synonyms
        "person": {
            Register.NEUTRAL: ["person", "individual"],
            Register.FORMAL: ["individual", "entity", "party"],
            Register.INFORMAL: ["guy", "dude", "folks"],
            Register.SLANG: ["homie", "peeps", "fam"],
        },
        "money": {
            Register.NEUTRAL: ["money", "cash", "funds"],
            Register.FORMAL: ["capital", "finances", "resources"],
            Register.ACADEMIC: ["currency", "monetary assets"],
            Register.INFORMAL: ["bucks", "dough"],
            Register.SLANG: ["bread", "cheddar", "guap", "bands", "racks"],
        },
        "problem": {
            Register.NEUTRAL: ["problem", "issue"],
            Register.FORMAL: ["challenge", "difficulty", "complication"],
            Register.ACADEMIC: ["impediment", "obstacle", "constraint"],
            Register.INFORMAL: ["trouble", "mess", "headache"],
            Register.SLANG: ["drama", "beef"],
        },
        "idea": {
            Register.NEUTRAL: ["idea", "thought"],
            Register.FORMAL: ["concept", "notion", "proposition"],
            Register.ACADEMIC: ["hypothesis", "thesis", "postulation"],
            Register.INFORMAL: ["plan", "scheme"],
            Register.SLANG: ["vibe", "move"],
        },
        "work": {
            Register.NEUTRAL: ["work", "job"],
            Register.FORMAL: ["employment", "occupation", "profession"],
            Register.ACADEMIC: ["vocation", "endeavor"],
            Register.INFORMAL: ["gig", "hustle"],
            Register.SLANG: ["grind", "9-to-5"],
        },
        "friend": {
            Register.NEUTRAL: ["friend", "companion"],
            Register.FORMAL: ["associate", "colleague", "acquaintance"],
            Register.INFORMAL: ["buddy", "pal", "mate"],
            Register.SLANG: ["homie", "bro", "bestie", "fam", "dawg"],
        },
        "car": {
            Register.NEUTRAL: ["car", "vehicle", "automobile"],
            Register.FORMAL: ["motor vehicle", "transport"],
            Register.INFORMAL: ["ride", "wheels"],
            Register.SLANG: ["whip", "beater"],
        },
        "house": {
            Register.NEUTRAL: ["house", "home"],
            Register.FORMAL: ["residence", "dwelling", "domicile"],
            Register.INFORMAL: ["place", "pad"],
            Register.SLANG: ["crib", "spot"],
        },
    }
    
    # ==========================================================================
    # VERB FORMS
    # ==========================================================================
    
    # Verb to present participle (-ing form)
    VERB_TO_ING = {
        # Original
        "run": "running", "drive": "driving", "exist": "existing",
        "become": "becoming", "have": "having", "do": "doing",
        "make": "making", "go": "going", "see": "seeing", "believe": "believing",
        # Movement
        "walk": "walking", "move": "moving", "travel": "traveling",
        "fly": "flying", "swim": "swimming", "jump": "jumping",
        "climb": "climbing", "fall": "falling", "rise": "rising", "turn": "turning",
        "arrive": "arriving", "depart": "departing", "enter": "entering", "exit": "exiting",
        # Communication
        "say": "saying", "tell": "telling", "speak": "speaking", "talk": "talking",
        "ask": "asking", "answer": "answering", "explain": "explaining",
        # Mental
        "think": "thinking", "know": "knowing", "understand": "understanding",
        "remember": "remembering", "forget": "forgetting", "learn": "learning",
        "consider": "considering", "imagine": "imagining", "wonder": "wondering",
        # Action
        "take": "taking", "give": "giving", "get": "getting", "put": "putting",
        "bring": "bringing", "carry": "carrying", "hold": "holding",
        "write": "writing", "read": "reading", "show": "showing",
        "build": "building", "create": "creating", "break": "breaking",
        "open": "opening", "close": "closing", "start": "starting", "stop": "stopping",
        # State
        "be": "being", "seem": "seeming", "appear": "appearing",
        "look": "looking", "feel": "feeling", "remain": "remaining",
        # Possession
        "own": "owning", "belong": "belonging", "contain": "containing",
        # Emotion
        "love": "loving", "hate": "hating", "like": "liking", "enjoy": "enjoying",
        "want": "wanting", "need": "needing", "wish": "wishing", "hope": "hoping",
        # Work
        "work": "working", "manage": "managing", "lead": "leading",
        "help": "helping", "support": "supporting", "perform": "performing",
        # Change
        "change": "changing", "develop": "developing", "grow": "growing",
        "improve": "improving", "increase": "increasing", "expand": "expanding",
        "transform": "transforming", "evolve": "evolving", "adapt": "adapting",
        # Technical
        "analyze": "analyzing", "compute": "computing", "process": "processing",
        "optimize": "optimizing", "implement": "implementing", "deploy": "deploying",
        "test": "testing", "debug": "debugging", "validate": "validating",
    }
    
    # Verb to past participle
    VERB_TO_PAST_PARTICIPLE = {
        # Irregular verbs
        "run": "run", "drive": "driven", "become": "become", "have": "had",
        "do": "done", "make": "made", "go": "gone", "see": "seen",
        "be": "been", "take": "taken", "give": "given", "get": "gotten",
        "write": "written", "read": "read", "break": "broken", "speak": "spoken",
        "know": "known", "grow": "grown", "fly": "flown", "swim": "swum",
        "fall": "fallen", "rise": "risen", "begin": "begun", "hold": "held",
        "bring": "brought", "think": "thought", "tell": "told", "say": "said",
        "feel": "felt", "keep": "kept", "leave": "left", "find": "found",
        "understand": "understood", "forget": "forgotten",
        # Regular verbs (add -ed)
        "exist": "existed", "believe": "believed", "walk": "walked",
        "move": "moved", "travel": "traveled", "jump": "jumped", "climb": "climbed",
        "turn": "turned", "arrive": "arrived", "depart": "departed",
        "enter": "entered", "exit": "exited", "ask": "asked", "answer": "answered",
        "explain": "explained", "remember": "remembered", "learn": "learned",
        "consider": "considered", "imagine": "imagined", "wonder": "wondered",
        "carry": "carried", "show": "showed", "build": "built", "create": "created",
        "open": "opened", "close": "closed", "start": "started", "stop": "stopped",
        "seem": "seemed", "appear": "appeared", "look": "looked", "remain": "remained",
        "own": "owned", "belong": "belonged", "contain": "contained",
        "love": "loved", "hate": "hated", "like": "liked", "enjoy": "enjoyed",
        "want": "wanted", "need": "needed", "wish": "wished", "hope": "hoped",
        "work": "worked", "manage": "managed", "lead": "led", "help": "helped",
        "support": "supported", "perform": "performed",
        "change": "changed", "develop": "developed", "improve": "improved",
        "increase": "increased", "expand": "expanded", "transform": "transformed",
        "evolve": "evolved", "adapt": "adapted",
        "analyze": "analyzed", "compute": "computed", "process": "processed",
        "optimize": "optimized", "implement": "implemented", "deploy": "deployed",
        "test": "tested", "debug": "debugged", "validate": "validated",
    }
    
    def __init__(self):
        """Initialize vocabulary with all words and mappings."""
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_to_category = {}
        self.category_to_words = {}
        self.category_to_indices = {}
        self.word_to_register = {}  # Track word register
        self.word_to_synonyms = {}  # Quick synonym lookup
        
        self._build_vocabulary()
        self._build_synonym_index()
    
    def _build_vocabulary(self):
        """Build all vocabulary mappings."""
        idx = 0
        
        # Add special tokens first
        for token in [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]:
            self.word_to_idx[token] = idx
            self.idx_to_word[idx] = token
            self.word_to_category[token] = "special"
            idx += 1
        
        # Add category words
        for category, words in self.CATEGORIES.items():
            self.category_to_words[category] = words
            self.category_to_indices[category] = []
            
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
                    self.word_to_category[word] = category
                    idx += 1
                self.category_to_indices[category].append(self.word_to_idx[word])
        
        # Add -ing forms
        for base, ing_form in self.VERB_TO_ING.items():
            if ing_form not in self.word_to_idx:
                self.word_to_idx[ing_form] = idx
                self.idx_to_word[idx] = ing_form
                self.word_to_category[ing_form] = "verb_ing"
                idx += 1
        
        # Add past participles that aren't already in vocab
        for base, pp_form in self.VERB_TO_PAST_PARTICIPLE.items():
            if pp_form not in self.word_to_idx:
                self.word_to_idx[pp_form] = idx
                self.idx_to_word[idx] = pp_form
                self.word_to_category[pp_form] = "past_participle"
                idx += 1
        
        # Add synonym words that might not be in categories
        for word, registers in self.SYNONYMS.items():
            for register, synonyms in registers.items():
                for syn in synonyms:
                    # Handle multi-word synonyms by splitting
                    for part in syn.split():
                        if part not in self.word_to_idx:
                            self.word_to_idx[part] = idx
                            self.idx_to_word[idx] = part
                            self.word_to_category[part] = "synonym"
                            self.word_to_register[part] = register
                            idx += 1
        
        self.vocab_size = idx
    
    def _build_synonym_index(self):
        """Build reverse index for synonym lookup."""
        for base_word, registers in self.SYNONYMS.items():
            all_synonyms = set()
            for register, syns in registers.items():
                all_synonyms.update(syns)
            
            # Map each synonym back to all other synonyms
            for syn in all_synonyms:
                self.word_to_synonyms[syn] = list(all_synonyms)
    
    def get_synonyms(self, word, register=None):
        """
        Get synonyms for a word, optionally filtered by register.
        
        Args:
            word: The word to find synonyms for
            register: Optional register to filter by
            
        Returns:
            List of synonyms
        """
        if word not in self.SYNONYMS:
            return [word]
        
        if register:
            return self.SYNONYMS[word].get(register, [word])
        
        # Return all synonyms from all registers
        all_syns = set()
        for reg_syns in self.SYNONYMS[word].values():
            all_syns.update(reg_syns)
        return list(all_syns)
    
    def get_word_register(self, word):
        """Get the register of a word."""
        return self.word_to_register.get(word, Register.NEUTRAL)
    
    def encode(self, text):
        """Convert text to list of token indices."""
        words = text.lower().split()
        return [self.word_to_idx.get(w, self.word_to_idx[self.UNK_TOKEN]) for w in words]
    
    def decode(self, indices):
        """Convert list of token indices to text."""
        words = [self.idx_to_word.get(idx, self.UNK_TOKEN) for idx in indices]
        # Filter out special tokens for display
        words = [w for w in words if w not in [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]]
        return " ".join(words)
    
    def get_category_indices(self, category):
        """Get all token indices for a category."""
        return self.category_to_indices.get(category, [])
    
    def get_words_by_category(self, category):
        """Get all words in a category."""
        return self.category_to_words.get(category, [])
    
    def get_ing_form(self, verb):
        """Get -ing form of a verb."""
        return self.VERB_TO_ING.get(verb, verb + "ing")
    
    def get_past_participle(self, verb):
        """Get past participle form of a verb."""
        return self.VERB_TO_PAST_PARTICIPLE.get(verb, verb + "ed")
    
    def get_vocab_stats(self):
        """Get vocabulary statistics."""
        stats = {
            "total_size": self.vocab_size,
            "categories": {},
            "synonym_count": len(self.SYNONYMS),
        }
        for cat, words in self.category_to_words.items():
            stats["categories"][cat] = len(words)
        return stats
    
    def __len__(self):
        return self.vocab_size


class GrammarRule:
    """
    Defines grammar rules for sentence generation.
    """
    
    # Present continuous pattern:
    # pronoun + verb(ing) + determiner + adjective + noun + conjunction + 
    # pronoun + verb(ing) + particle + verb(infinitive) + noun + noun
    PRESENT_CONTINUOUS = [
        ("pronoun", None),
        ("auxiliary", None),  # is/are
        ("verb", "ing"),      # -ing form
        ("determiner", None),
        ("adjective", None),
        ("noun", None),
        ("conjunction", None),
        ("pronoun", None),
        ("auxiliary", None),
        ("verb", "ing"),
        ("particle", None),
        ("verb", "base"),     # infinitive
        ("noun", None),
        ("noun", None),
    ]
    
    # Past participle pattern:
    # determiner + noun + verb(aux) + past_participle
    PAST_PARTICIPLE = [
        ("determiner", None),
        ("noun", None),
        ("auxiliary", None),  # was/were
        ("verb", "past_participle"),
    ]
    
    # Extended patterns
    SIMPLE_PRESENT = [
        ("pronoun", None),
        ("verb", "base"),
        ("determiner", None),
        ("adjective", None),
        ("noun", None),
    ]
    
    COMPLEX_SENTENCE = [
        ("pronoun", None),
        ("auxiliary", None),
        ("adverb", None),
        ("verb", "ing"),
        ("determiner", None),
        ("adjective", None),
        ("adjective", None),
        ("noun", None),
        ("preposition", None),
        ("determiner", None),
        ("noun", None),
    ]
    
    # Pronoun to auxiliary verb mapping (for grammatical correctness)
    PRONOUN_TO_AUX = {
        "i": "am", "you": "are", "he": "is", "she": "is", "it": "is",
        "we": "are", "they": "are", "this": "is", "that": "is", "who": "is",
        "everyone": "is", "someone": "is", "anyone": "is", "nobody": "is",
        "everybody": "is", "somebody": "is", "one": "is", "each": "is",
    }
    
    # Singular/plural auxiliaries for past tense
    SINGULAR_PAST_AUX = "was"
    PLURAL_PAST_AUX = "were"
    
    @staticmethod
    def get_pattern(pattern_name):
        """Get a grammar pattern by name."""
        patterns = {
            "present_continuous": GrammarRule.PRESENT_CONTINUOUS,
            "past_participle": GrammarRule.PAST_PARTICIPLE,
            "simple_present": GrammarRule.SIMPLE_PRESENT,
            "complex_sentence": GrammarRule.COMPLEX_SENTENCE,
        }
        return patterns.get(pattern_name, [])
    
    @staticmethod
    def get_auxiliary(pronoun, tense="present"):
        """Get appropriate auxiliary verb for a pronoun."""
        if tense == "present":
            return GrammarRule.PRONOUN_TO_AUX.get(pronoun.lower(), "is")
        elif tense == "past":
            # Simplified: use 'was' for singular pronouns
            plural = pronoun.lower() in ["we", "they", "you"]
            return GrammarRule.PLURAL_PAST_AUX if plural else GrammarRule.SINGULAR_PAST_AUX
        return "is"

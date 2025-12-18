"""
Training loop for the language model.
Includes grammar rules, example sentences, and proper article selection.
"""

import time
from .math_utils import Loss, Activations, Random
from .vocabulary import Vocabulary, GrammarRule


# =============================================================================
# EXAMPLE SENTENCES FOR TRAINING
# These teach the model proper grammar patterns
# =============================================================================

EXAMPLE_SENTENCES = [
    # A vs AN examples (consonant vs vowel sounds)
    "a car is fast",
    "an apple is red",
    "a big house",
    "an old building",
    "a university student",  # "u" sounds like "y" = a
    "an umbrella is useful",  # "u" sounds like "uh" = an  
    "a European city",  # "E" sounds like "Y" = a
    "an hour ago",  # silent "h" = an
    "a happy person",
    "an honest answer",  # silent "h" = an
    "a unique idea",  # "u" sounds like "y" = a
    "an unusual event",
    
    # Present continuous examples
    "i am running quickly",
    "you are driving carefully",
    "he is working hard",
    "she is speaking loudly", 
    "it is growing slowly",
    "we are learning fast",
    "they are building something",
    
    # Past participle examples  
    "the car was driven",
    "the book was written",
    "the work was done",
    "the problem was solved",
    "the code was tested",
    "the system was built",
    "the data was analyzed",
    
    # Subject-verb agreement
    "he runs every day",
    "she works at home",
    "it seems important",
    "they run together",
    "we work as a team",
    "i think therefore",
    
    # Pronoun + auxiliary agreement
    "i am happy",
    "you are smart", 
    "he is tall",
    "she is kind",
    "it is big",
    "we are ready",
    "they are coming",
    
    # Complex sentences
    "the smart developer is building an innovative solution",
    "a skilled engineer was developing the new system",
    "an experienced scientist is analyzing the important data",
    "the efficient algorithm is processing the complex information",
    "a creative designer is creating an amazing interface",
    
    # Formal vs informal
    "the manager is proceeding with the strategic plan",
    "she is helping her friend with the difficult problem",
    "an expert is examining the critical evidence",
    "the team is working on an important project",
    
    # Technical examples
    "the system is processing data efficiently",
    "an algorithm is optimizing the performance",
    "the developer is debugging the complex code",
    "a program is analyzing the user input",
    "the network is transmitting information securely",
    
    # Common patterns
    "someone is waiting outside",
    "everyone is working together", 
    "nobody was expecting this",
    "something was happening quickly",
    "anyone can learn this",
    
    # Adverb placement
    "he is quickly running",
    "she always works hard",
    "they never stop learning",
    "we often work together",
    "i sometimes think deeply",
]


# =============================================================================
# CONVERSATION PAIRS FOR Q&A TRAINING
# Format: (input, response) - teaches model to respond appropriately
# =============================================================================

CONVERSATION_PAIRS = [
    # Greetings
    ("hello", "hello how are you"),
    ("hi", "hi there how can i help"),
    ("hey", "hey what is up"),
    ("good morning", "good morning how are you"),
    ("good evening", "good evening nice to see you"),
    
    # How are you variations
    ("how are you", "i am doing well thank you"),
    ("how is it going", "it is going great thanks"),
    ("what is up", "not much just working here"),
    ("are you okay", "yes i am fine thank you"),
    
    # Questions about identity
    ("who are you", "i am a simple language model"),
    ("what are you", "i am an algorithm that processes text"),
    ("what is your name", "i am a language model without a name"),
    ("are you human", "no i am a computer program"),
    ("are you a robot", "i am a language model made with code"),
    
    # Questions about capabilities
    ("what can you do", "i can generate text and answer questions"),
    ("can you help me", "yes i can try to help you"),
    ("do you understand me", "i am trying to understand you"),
    ("can you think", "i process information but do not think like you"),
    ("are you smart", "i am learning to be helpful"),
    
    # Simple questions
    ("what is a city", "a city is a large urban area with many people"),
    ("what is data", "data is information that is stored and processed"),
    ("what is an algorithm", "an algorithm is a set of steps to solve a problem"),
    ("what is a car", "a car is a vehicle that people drive"),
    ("what is a computer", "a computer is a machine that processes information"),
    
    # Opinion questions
    ("do you like cars", "i find cars interesting as technology"),
    ("what do you think", "i think this is an interesting question"),
    ("is this good", "it depends on the context and goals"),
    ("is it important", "yes understanding is always important"),
    
    # Action requests
    ("tell me something", "the world is full of interesting data"),
    ("say something", "hello i am here to help you"),
    ("explain this", "i will try to explain it simply"),
    ("help me understand", "let me help you understand this"),
    
    # Agreements and confirmations
    ("okay", "great let me know if you need more help"),
    ("thanks", "you are welcome happy to help"),
    ("thank you", "you are welcome anytime"),
    ("yes", "understood i will continue"),
    ("no", "okay i understand your answer"),
    
    # Farewells
    ("goodbye", "goodbye have a great day"),
    ("bye", "bye take care"),
    ("see you", "see you later"),
    ("i have to go", "okay goodbye and take care"),
    
    # Technical conversations
    ("how does this work", "it works by processing data step by step"),
    ("what is happening", "the system is processing your input"),
    ("is it working", "yes everything is working correctly"),
    ("something is wrong", "let me help you find the problem"),
    
    # Learning and knowledge
    ("i want to learn", "learning is a great goal keep going"),
    ("teach me", "i will try to teach you something new"),
    ("i do not understand", "let me explain it in a different way"),
    ("this is hard", "yes but with practice it becomes easier"),
    ("i am confused", "that is okay let me help clarify"),
]


class Trainer:
    """
    Trainer class for the SimpleLLM model.
    Handles data generation, training loop, and loss computation.
    """
    
    # Words that start with vowel SOUNDS (not just vowel letters)
    VOWEL_SOUND_WORDS = {
        # Regular vowel starts
        "a", "e", "i", "o", "u",
        # Words starting with vowel sounds
        "apple", "algorithm", "amazing", "ancient", "angry", "animal",
        "elephant", "elegant", "electric", "element", "excellent", "expert",
        "idea", "important", "immersive", "incredible", "individual", "information",
        "object", "obvious", "ocean", "office", "old", "open", "opportunity",
        "ugly", "umbrella", "understanding", "unusual", "urban", "urgent",
        # Silent H words (use "an")
        "hour", "honest", "honor", "heir",
    }
    
    # Words that start with consonant SOUNDS despite vowel letters
    CONSONANT_SOUND_WORDS = {
        # "U" sounding like "Y" 
        "university", "unique", "uniform", "unit", "united", "useful", "user",
        # "E" sounding like "Y"
        "european", "euphoria",
        # "O" sounding like "W"
        "one", "once",
    }
    
    def __init__(self, model, vocabulary, learning_rate=0.001):
        """
        Initialize trainer.
        
        Args:
            model: SimpleLLM model instance
            vocabulary: Vocabulary instance
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.vocabulary = vocabulary
        self.learning_rate = learning_rate
        self.rng = Random()
        
        # Training history
        self.loss_history = []
        
        # Preprocess example sentences and conversations
        self.example_sentences = self._prepare_examples()
        self.conversation_pairs = self._prepare_conversations()
    
    def _prepare_examples(self):
        """Convert example sentences to token indices."""
        examples = []
        for sent in EXAMPLE_SENTENCES:
            indices = self.vocabulary.encode(sent)
            # Only include if all words are in vocabulary
            if self.vocabulary.word_to_idx[Vocabulary.UNK_TOKEN] not in indices:
                examples.append(indices)
        return examples
    
    def _prepare_conversations(self):
        """Convert conversation pairs to token indices for Q&A training."""
        pairs = []
        sep_token = self.vocabulary.word_to_idx.get(":", None)  # Optional separator
        
        for user_input, response in CONVERSATION_PAIRS:
            input_indices = self.vocabulary.encode(user_input)
            response_indices = self.vocabulary.encode(response)
            
            # Check if all words are in vocabulary
            unk_idx = self.vocabulary.word_to_idx[Vocabulary.UNK_TOKEN]
            if unk_idx not in input_indices and unk_idx not in response_indices:
                # Combine as: input + response (model learns to continue)
                combined = input_indices + response_indices
                pairs.append((input_indices, response_indices, combined))
        
        return pairs
    
    def select_article(self, next_word):
        """
        Select correct article (a/an) based on the following word.
        
        Args:
            next_word: The word that follows the article
            
        Returns:
            'a' or 'an' based on phonetic rules
        """
        word_lower = next_word.lower()
        
        # Check exceptions first
        if word_lower in self.CONSONANT_SOUND_WORDS:
            return "a"
        
        if word_lower in self.VOWEL_SOUND_WORDS:
            return "an"
        
        # Default rule: vowel letter = an, consonant = a
        if word_lower and word_lower[0] in 'aeiou':
            return "an"
        
        return "a"
    
    def generate_training_sentence(self, pattern_name="present_continuous"):
        """
        Generate a random training sentence following grammar rules.
        With proper article (a/an) selection.
        
        Args:
            pattern_name: Name of grammar pattern to use
            
        Returns:
            List of words forming the sentence
        """
        pattern = GrammarRule.get_pattern(pattern_name)
        sentence = []
        last_pronoun = None
        
        for i, (category, modifier) in enumerate(pattern):
            if category == "pronoun":
                word = self.rng.choice(self.vocabulary.get_words_by_category("pronoun"))
                last_pronoun = word
                sentence.append(word)
            
            elif category == "auxiliary":
                # Get appropriate auxiliary based on last pronoun
                if last_pronoun:
                    word = GrammarRule.get_auxiliary(last_pronoun, "present")
                else:
                    word = "is"
                sentence.append(word)
            
            elif category == "verb":
                base_verb = self.rng.choice(self.vocabulary.get_words_by_category("verb"))
                
                if modifier == "ing":
                    word = self.vocabulary.get_ing_form(base_verb)
                elif modifier == "past_participle":
                    word = self.vocabulary.get_past_participle(base_verb)
                else:
                    word = base_verb
                sentence.append(word)
            
            elif category == "adjective":
                word = self.rng.choice(self.vocabulary.get_words_by_category("adjective"))
                sentence.append(word)
            
            elif category == "noun":
                word = self.rng.choice(self.vocabulary.get_words_by_category("noun"))
                sentence.append(word)
            
            elif category == "conjunction":
                word = self.rng.choice(self.vocabulary.get_words_by_category("conjunction"))
                sentence.append(word)
            
            elif category == "determiner":
                # Look ahead to next word to select a/an correctly
                next_word = None
                for j in range(i + 1, len(pattern)):
                    next_cat, next_mod = pattern[j]
                    if next_cat in ["adjective", "noun"]:
                        # Get a sample word from that category
                        next_words = self.vocabulary.get_words_by_category(next_cat)
                        if next_words:
                            next_word = self.rng.choice(next_words)
                            break
                
                if next_word:
                    # Select a or an based on next word
                    article = self.select_article(next_word)
                    # 50% chance to use "the" instead
                    if self.rng.random() < 0.3:
                        word = "the"
                    else:
                        word = article
                else:
                    word = self.rng.choice(["the", "a", "an"])
                sentence.append(word)
            
            elif category == "adverb":
                word = self.rng.choice(self.vocabulary.get_words_by_category("adverb"))
                sentence.append(word)
            
            elif category == "preposition":
                word = self.rng.choice(self.vocabulary.get_words_by_category("preposition"))
                sentence.append(word)
            
            elif category == "particle":
                word = self.rng.choice(self.vocabulary.get_words_by_category("particle"))
                sentence.append(word)
        
        return sentence
    
    def generate_past_participle_sentence(self):
        """
        Generate a past participle sentence with correct grammar.
        
        Returns:
            List of words forming the sentence
        """
        sentence = []
        
        # Get noun first to determine article
        noun = self.rng.choice(self.vocabulary.get_words_by_category("noun"))
        
        # Select correct article
        if self.rng.random() < 0.4:
            determiner = "the"
        else:
            determiner = self.select_article(noun)
        
        sentence.append(determiner)
        sentence.append(noun)
        
        # Auxiliary (was/were)
        aux = "was"  # Simplified - always use was
        sentence.append(aux)
        
        # Past participle
        verb = self.rng.choice(self.vocabulary.get_words_by_category("verb"))
        past_part = self.vocabulary.get_past_participle(verb)
        sentence.append(past_part)
        
        return sentence
    
    def create_training_batch(self, batch_size=32, pattern="mixed"):
        """
        Create a batch of training examples.
        Includes both generated sentences and example sentences.
        
        Args:
            batch_size: Number of examples to generate
            pattern: "present_continuous", "past_participle", or "mixed"
            
        Returns:
            List of (input_indices, target_indices) pairs
        """
        batch = []
        
        for idx in range(batch_size):
            rand_val = self.rng.random()
            
            # 25% chance to use conversation pairs for Q&A learning
            if self.conversation_pairs and rand_val < 0.25:
                _, _, combined = self.rng.choice(self.conversation_pairs)
                indices = combined
            # 25% chance to use example sentences for grammar learning
            elif self.example_sentences and rand_val < 0.50:
                indices = self.rng.choice(self.example_sentences)
            # 50% chance to use generated sentences
            else:
                if pattern == "mixed":
                    if self.rng.random() < 0.5:
                        words = self.generate_training_sentence("present_continuous")
                    else:
                        words = self.generate_past_participle_sentence()
                elif pattern == "present_continuous":
                    words = self.generate_training_sentence("present_continuous")
                else:
                    words = self.generate_past_participle_sentence()
                
                # Convert to indices
                indices = self.vocabulary.encode(" ".join(words))
            
            # Create input-target pairs (next token prediction)
            # Input: [BOS, word1, word2, ...], Target: [word1, word2, ..., EOS]
            bos_idx = self.vocabulary.word_to_idx[Vocabulary.BOS_TOKEN]
            eos_idx = self.vocabulary.word_to_idx[Vocabulary.EOS_TOKEN]
            
            input_seq = [bos_idx] + indices
            target_seq = indices + [eos_idx]
            
            batch.append((input_seq, target_seq))
        
        return batch

    
    def train_step(self, input_seq, target_seq, max_positions=None):
        """
        Perform one training step.
        
        Args:
            input_seq: Input token indices
            target_seq: Target token indices
            max_positions: Max positions to train on (None = all, speeds up training)
            
        Returns:
            Loss value for this step
        """
        total_loss = 0.0
        seq_len = len(input_seq)
        
        # Turbo mode: only train on last N positions (much faster!)
        if max_positions and seq_len > max_positions:
            # Train on first 2 and last (max_positions-2) positions
            positions = list(range(min(2, seq_len))) + list(range(seq_len - max_positions + 2, seq_len))
        else:
            positions = range(seq_len)
        
        num_trained = 0
        for t in positions:
            # Get context up to this point
            context = input_seq[:t + 1]
            target_idx = target_seq[t]
            
            # Forward pass
            logits = self.model.forward(context)
            
            # Compute softmax
            probs = Activations.softmax(logits)
            
            # Compute loss
            loss = Loss.cross_entropy(probs, target_idx)
            total_loss += loss
            num_trained += 1
            
            # Compute gradient
            grad = Loss.cross_entropy_gradient(probs, target_idx)
            
            # Backward pass
            self.model.backward(grad)
            
            # Update weights
            self.model.update(self.learning_rate)
        
        return total_loss / num_trained if num_trained > 0 else 0.0
    
    def train(self, num_epochs=100, batch_size=32, pattern="mixed", 
              verbose=True, print_every=10, turbo=False):
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Examples per batch
            pattern: Sentence pattern to train on
            verbose: Whether to print progress
            print_every: How often to print updates
            turbo: If True, only train on 4 positions per sequence (10x faster)
            
        Returns:
            List of average losses per epoch
        """
        # Turbo mode trains on fewer positions per sequence
        max_positions = 4 if turbo else None
        
        if turbo and verbose:
            print("TURBO MODE: Training on 4 positions per sequence (10x faster)")
        
        epoch_losses = []
        start_time = time.time()
        total_samples = num_epochs * batch_size
        samples_processed = 0
        
        for epoch in range(num_epochs):
            batch = self.create_training_batch(batch_size, pattern)
            epoch_loss = 0.0
            epoch_start = time.time()
            
            for sample_idx, (input_seq, target_seq) in enumerate(batch):
                sample_start = time.time()
                loss = self.train_step(input_seq, target_seq, max_positions=max_positions)
                epoch_loss += loss
                samples_processed += 1
                
                # Progress within epoch
                if verbose:
                    elapsed = time.time() - start_time
                    samples_per_sec = samples_processed / elapsed if elapsed > 0 else 0
                    remaining_samples = total_samples - samples_processed
                    eta = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
                    
                    # Progress bar
                    pct = (sample_idx + 1) / batch_size
                    bar_len = 20
                    filled = int(bar_len * pct)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    
                    print(f"\rEpoch {epoch + 1}/{num_epochs} [{bar}] {sample_idx + 1}/{batch_size} | "
                          f"Loss: {loss:.4f} | {samples_per_sec:.1f} samples/s | ETA: {eta:.0f}s", end='', flush=True)
            
            avg_loss = epoch_loss / batch_size
            epoch_losses.append(avg_loss)
            self.loss_history.append(avg_loss)
            
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            
            if verbose:
                print(f"\rEpoch {epoch + 1}/{num_epochs} completed | Avg Loss: {avg_loss:.4f} | "
                      f"Epoch time: {epoch_time:.1f}s | Total: {elapsed:.1f}s" + " " * 20)
        
        if verbose:
            total_time = time.time() - start_time
            print(f"\nTraining complete! Total time: {total_time:.1f}s")
        
        return epoch_losses
    
    def evaluate(self, num_samples=100, pattern="mixed"):
        """
        Evaluate model on generated samples.
        
        Args:
            num_samples: Number of samples to evaluate
            pattern: Sentence pattern to evaluate on
            
        Returns:
            Average loss and perplexity
        """
        import math
        
        total_loss = 0.0
        total_tokens = 0
        
        batch = self.create_training_batch(num_samples, pattern)
        
        for input_seq, target_seq in batch:
            for t in range(len(input_seq)):
                context = input_seq[:t + 1]
                target_idx = target_seq[t]
                
                logits = self.model.forward(context)
                probs = Activations.softmax(logits)
                loss = Loss.cross_entropy(probs, target_idx)
                
                total_loss += loss
                total_tokens += 1
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        
        return avg_loss, perplexity

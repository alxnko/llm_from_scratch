"""
Sentence generation following grammar rules.
"""

from .vocabulary import Vocabulary, GrammarRule
from .math_utils import Random, Activations


class SentenceGenerator:
    """
    Generates sentences using the trained language model,
    following the defined grammar patterns.
    """
    
    def __init__(self, model, vocabulary, seed=None):
        """
        Initialize sentence generator.
        
        Args:
            model: Trained SimpleLLM model
            vocabulary: Vocabulary instance
            seed: Random seed for reproducibility
        """
        self.model = model
        self.vocabulary = vocabulary
        self.rng = Random(seed) if seed else model.rng
    
    def generate_free(self, prompt=None, max_length=20, temperature=1.0):
        """
        Generate text freely without grammar constraints.
        
        Args:
            prompt: Optional starting text
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text string
        """
        # Initialize with BOS token or prompt
        if prompt:
            tokens = self.vocabulary.encode(prompt)
            tokens = [self.vocabulary.word_to_idx[Vocabulary.BOS_TOKEN]] + tokens
        else:
            tokens = [self.vocabulary.word_to_idx[Vocabulary.BOS_TOKEN]]
        
        eos_idx = self.vocabulary.word_to_idx[Vocabulary.EOS_TOKEN]
        generated = []
        
        for _ in range(max_length):
            # Predict next token
            next_idx, probs = self.model.predict_next(tokens, temperature)
            
            # Stop if EOS
            if next_idx == eos_idx:
                break
            
            generated.append(next_idx)
            tokens.append(next_idx)
        
        # Convert to text
        return self.vocabulary.decode(generated)
    
    def generate_present_continuous(self, temperature=0.8):
        """
        Generate a present continuous sentence following the grammar pattern.
        Uses model to select within grammar constraints.
        
        Pattern: pronoun + aux + verb(ing) + det + adj + noun + conj + 
                 pronoun + aux + verb(ing) + particle + verb + noun + noun
        
        Args:
            temperature: Sampling temperature
            
        Returns:
            Generated sentence string
        """
        sentence = []
        tokens = [self.vocabulary.word_to_idx[Vocabulary.BOS_TOKEN]]
        
        # Step 1: First pronoun
        pronoun1 = self._select_from_category("pronoun", tokens, temperature)
        sentence.append(pronoun1)
        tokens.append(self.vocabulary.word_to_idx[pronoun1])
        
        # Step 2: First auxiliary (based on pronoun)
        aux1 = GrammarRule.get_auxiliary(pronoun1)
        sentence.append(aux1)
        if aux1 in self.vocabulary.word_to_idx:
            tokens.append(self.vocabulary.word_to_idx[aux1])
        
        # Step 3: First verb (ing form)
        verb1 = self._select_from_category("verb", tokens, temperature)
        verb1_ing = self.vocabulary.get_ing_form(verb1)
        sentence.append(verb1_ing)
        if verb1_ing in self.vocabulary.word_to_idx:
            tokens.append(self.vocabulary.word_to_idx[verb1_ing])
        
        # Step 4: Determiner
        det = self._select_from_category("determiner", tokens, temperature)
        sentence.append(det)
        tokens.append(self.vocabulary.word_to_idx[det])
        
        # Step 5: Adjective
        adj = self._select_from_category("adjective", tokens, temperature)
        sentence.append(adj)
        tokens.append(self.vocabulary.word_to_idx[adj])
        
        # Step 6: First noun
        noun1 = self._select_from_category("noun", tokens, temperature)
        sentence.append(noun1)
        tokens.append(self.vocabulary.word_to_idx[noun1])
        
        # Step 7: Conjunction
        conj = self._select_from_category("conjunction", tokens, temperature)
        sentence.append(conj)
        tokens.append(self.vocabulary.word_to_idx[conj])
        
        # Step 8: Second pronoun
        pronoun2 = self._select_from_category("pronoun", tokens, temperature)
        sentence.append(pronoun2)
        tokens.append(self.vocabulary.word_to_idx[pronoun2])
        
        # Step 9: Second auxiliary
        aux2 = GrammarRule.get_auxiliary(pronoun2)
        sentence.append(aux2)
        if aux2 in self.vocabulary.word_to_idx:
            tokens.append(self.vocabulary.word_to_idx[aux2])
        
        # Step 10: Second verb (ing form)
        verb2 = self._select_from_category("verb", tokens, temperature)
        verb2_ing = self.vocabulary.get_ing_form(verb2)
        sentence.append(verb2_ing)
        if verb2_ing in self.vocabulary.word_to_idx:
            tokens.append(self.vocabulary.word_to_idx[verb2_ing])
        
        # Step 11: Particle
        particle = self._select_from_category("particle", tokens, temperature)
        sentence.append(particle)
        tokens.append(self.vocabulary.word_to_idx[particle])
        
        # Step 12: Third verb (infinitive/base)
        verb3 = self._select_from_category("verb", tokens, temperature)
        sentence.append(verb3)
        tokens.append(self.vocabulary.word_to_idx[verb3])
        
        # Step 13: Second noun
        noun2 = self._select_from_category("noun", tokens, temperature)
        sentence.append(noun2)
        tokens.append(self.vocabulary.word_to_idx[noun2])
        
        # Step 14: Third noun
        noun3 = self._select_from_category("noun", tokens, temperature)
        sentence.append(noun3)
        tokens.append(self.vocabulary.word_to_idx[noun3])
        
        return " ".join(sentence)
    
    def generate_past_participle(self, temperature=0.8):
        """
        Generate a past participle sentence following the grammar pattern.
        
        Pattern: determiner + noun + was/were + past_participle
        
        Args:
            temperature: Sampling temperature
            
        Returns:
            Generated sentence string
        """
        sentence = []
        tokens = [self.vocabulary.word_to_idx[Vocabulary.BOS_TOKEN]]
        
        # Step 1: Determiner
        det = self._select_from_category("determiner", tokens, temperature)
        sentence.append(det)
        tokens.append(self.vocabulary.word_to_idx[det])
        
        # Step 2: Noun
        noun = self._select_from_category("noun", tokens, temperature)
        sentence.append(noun)
        tokens.append(self.vocabulary.word_to_idx[noun])
        
        # Step 3: Auxiliary (was)
        aux = "was"
        sentence.append(aux)
        if aux in self.vocabulary.word_to_idx:
            tokens.append(self.vocabulary.word_to_idx[aux])
        
        # Step 4: Past participle
        verb = self._select_from_category("verb", tokens, temperature)
        past_part = self.vocabulary.get_past_participle(verb)
        sentence.append(past_part)
        
        return " ".join(sentence)
    
    def _select_from_category(self, category, context, temperature=1.0):
        """
        Select a word from a category using model probabilities.
        
        Args:
            category: Word category to select from
            context: Current token sequence
            temperature: Sampling temperature
            
        Returns:
            Selected word
        """
        # Get category words
        words = self.vocabulary.get_words_by_category(category)
        
        if not words:
            return ""
        
        # Get model logits for next token
        logits = self.model.forward(context)
        
        # Apply temperature
        if temperature != 1.0:
            logits = [l / temperature for l in logits]
        
        # Get probabilities only for words in this category
        category_probs = []
        for word in words:
            idx = self.vocabulary.word_to_idx.get(word, 0)
            category_probs.append((word, logits[idx]))
        
        # Softmax on category subset
        max_logit = max(p[1] for p in category_probs)
        exp_probs = [(w, _safe_exp(l - max_logit)) for w, l in category_probs]
        sum_exp = sum(p[1] for p in exp_probs)
        normalized = [(w, e / sum_exp) for w, e in exp_probs]
        
        # Sample from distribution
        r = self.rng.random()
        cumsum = 0.0
        for word, prob in normalized:
            cumsum += prob
            if r < cumsum:
                return word
        
        return words[-1]
    
    def generate_batch(self, num_sentences=5, pattern="mixed", temperature=0.8):
        """
        Generate multiple sentences.
        
        Args:
            num_sentences: Number of sentences to generate
            pattern: "present_continuous", "past_participle", or "mixed"
            temperature: Sampling temperature
            
        Returns:
            List of generated sentences
        """
        sentences = []
        
        for i in range(num_sentences):
            if pattern == "mixed":
                if self.rng.random() < 0.5:
                    sent = self.generate_present_continuous(temperature)
                else:
                    sent = self.generate_past_participle(temperature)
            elif pattern == "present_continuous":
                sent = self.generate_present_continuous(temperature)
            else:
                sent = self.generate_past_participle(temperature)
            
            sentences.append(sent)
        
        return sentences


def _safe_exp(x):
    """Safe exponential that avoids overflow."""
    import math
    x = max(-500, min(500, x))
    return math.exp(x)

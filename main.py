"""
LLM From Scratch - Main Entry Point

A minimal language model implemented in pure Python with no external dependencies.
Designed to run on CPU-only systems (tested on AMD Ryzen 5 5500U, 16GB RAM).

Usage:
    python main.py                    # Interactive mode
    python main.py --train            # Train the model
    python main.py --generate         # Generate sentences
    python main.py --test-forward     # Test forward pass
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.vocabulary import Vocabulary, GrammarRule
from llm.model import SimpleLLM
from llm.trainer import Trainer
from llm.generator import SentenceGenerator
from llm.math_utils import Random


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("  LLM From Scratch - Pure Python Implementation")
    print("  No External Dependencies")
    print("=" * 60)
    print()


def test_vocabulary():
    """Test vocabulary functionality."""
    print("\n--- Testing Vocabulary ---")
    vocab = Vocabulary()
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"\nCategories: {list(vocab.category_to_words.keys())}")
    
    # Test encoding/decoding
    test_sentence = "he is driving the fast car"
    encoded = vocab.encode(test_sentence)
    decoded = vocab.decode(encoded)
    
    print(f"\nOriginal: {test_sentence}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test -ing forms
    print("\n-ing forms:")
    for verb in ["run", "drive", "make", "see"]:
        print(f"  {verb} -> {vocab.get_ing_form(verb)}")
    
    # Test past participles
    print("\nPast participles:")
    for verb in ["run", "drive", "make", "see"]:
        print(f"  {verb} -> {vocab.get_past_participle(verb)}")
    
    return vocab


def test_forward_pass(vocab):
    """Test model forward pass."""
    print("\n--- Testing Forward Pass ---")
    
    # Create model
    model = SimpleLLM(
        vocab_size=len(vocab),
        embed_dim=32,  # Small for testing
        num_layers=1,
        max_seq_len=20,
        seed=42
    )
    
    # Test sentence
    test_input = [vocab.word_to_idx[Vocabulary.BOS_TOKEN], 
                  vocab.word_to_idx["he"],
                  vocab.word_to_idx["is"]]
    
    print(f"Input tokens: {test_input}")
    print(f"Input words: {[vocab.idx_to_word[i] for i in test_input]}")
    
    # Forward pass
    logits = model.forward(test_input)
    
    print(f"\nOutput logits shape: {len(logits)}")
    print(f"First 5 logits: {[f'{l:.4f}' for l in logits[:5]]}")
    
    # Get prediction
    next_idx, probs = model.predict_next(test_input, temperature=1.0)
    predicted_word = vocab.idx_to_word.get(next_idx, "<UNK>")
    
    print(f"\nPredicted next token: {next_idx} ({predicted_word})")
    print(f"Top 5 probabilities:")
    
    # Sort probabilities
    prob_pairs = [(i, p) for i, p in enumerate(probs)]
    prob_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for idx, prob in prob_pairs[:5]:
        word = vocab.idx_to_word.get(idx, "<UNK>")
        print(f"  {word}: {prob:.4f}")
    
    return model


def test_training(vocab):
    """Test training functionality."""
    print("\n--- Testing Training ---")
    
    # Create model with small dimensions for fast testing
    model = SimpleLLM(
        vocab_size=len(vocab),
        embed_dim=32,
        num_layers=1,
        max_seq_len=20,
        seed=42
    )
    
    # Create trainer
    trainer = Trainer(model, vocab, learning_rate=0.01)
    
    # Generate a sample sentence
    print("\nSample present continuous sentence:")
    sent = trainer.generate_training_sentence("present_continuous")
    print(f"  {' '.join(sent)}")
    
    print("\nSample past participle sentence:")
    sent = trainer.generate_past_participle_sentence()
    print(f"  {' '.join(sent)}")
    
    # Train for a few epochs
    print("\nTraining for 5 epochs (quick test)...")
    losses = trainer.train(
        num_epochs=5,
        batch_size=4,
        pattern="mixed",
        verbose=True,
        print_every=1
    )
    
    print(f"\nFinal loss: {losses[-1]:.4f}")
    
    return model, trainer


def test_generation(model, vocab):
    """Test sentence generation."""
    print("\n--- Testing Generation ---")
    
    generator = SentenceGenerator(model, vocab, seed=42)
    
    print("\nPresent continuous sentences:")
    for i in range(3):
        sent = generator.generate_present_continuous(temperature=0.8)
        print(f"  {i+1}. {sent}")
    
    print("\nPast participle sentences:")
    for i in range(3):
        sent = generator.generate_past_participle(temperature=0.8)
        print(f"  {i+1}. {sent}")
    
    print("\nFree generation (no grammar constraints):")
    sent = generator.generate_free(prompt="he is", max_length=10, temperature=1.0)
    print(f"  'he is' -> {sent}")


def train_model(epochs=100, embed_dim=64, num_layers=2, batch_size=16, 
                learning_rate=0.001, save_path="model.json", turbo=False):
    """Train the model with specified parameters."""
    print("\n--- Training Model ---")
    
    vocab = Vocabulary()
    print(f"Vocabulary size: {len(vocab)}")
    
    model = SimpleLLM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_layers=num_layers,
        max_seq_len=32,
        seed=42
    )
    
    print(f"\nModel configuration:")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Max sequence length: 32")
    
    trainer = Trainer(model, vocab, learning_rate=learning_rate)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    print("\nStarting training...\n")
    
    trainer.train(
        num_epochs=epochs,
        batch_size=batch_size,
        pattern="mixed",
        verbose=True,
        print_every=10,
        turbo=turbo
    )
    
    # Evaluate
    avg_loss, perplexity = trainer.evaluate(num_samples=50)
    print(f"\nEvaluation:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    
    # Save model
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")
    
    # Generate samples
    print("\nGenerated sentences after training:")
    generator = SentenceGenerator(model, vocab)
    
    print("\nPresent continuous:")
    for sent in generator.generate_batch(3, "present_continuous"):
        print(f"  - {sent}")
    
    print("\nPast participle:")
    for sent in generator.generate_batch(3, "past_participle"):
        print(f"  - {sent}")
    
    return model, vocab


def chat_mode(model_path="model.json"):
    """Interactive chat mode - talk with the trained model."""
    print("\n" + "=" * 60)
    print("  CHAT MODE - Talk with the LLM")
    print("  Type 'quit' or 'exit' to leave")
    print("  Type 'help' for commands")
    print("=" * 60 + "\n")
    
    vocab = Vocabulary()
    
    try:
        model = SimpleLLM.load(model_path)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"No model found at {model_path}. Training a quick model...")
        model = SimpleLLM(
            vocab_size=len(vocab),
            embed_dim=32,
            num_layers=1,
            max_seq_len=32,
            seed=42
        )
        from llm.trainer import Trainer
        trainer = Trainer(model, vocab, learning_rate=0.01)
        trainer.train(num_epochs=10, batch_size=4, turbo=True, verbose=True)
        model.save(model_path)
        print("Quick model trained and saved!")
    
    generator = SentenceGenerator(model, vocab)
    
    print("\nReady to chat! (responses are generated, quality depends on training)\n")
    
    while True:
        try:
            user_input = input("You: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input in ["quit", "exit", "bye", "goodbye"]:
            print("Bot: Goodbye! Have a great day!")
            break
        
        if user_input == "help":
            print("\nCommands:")
            print("  quit/exit - Leave chat mode")
            print("  help - Show this message")
            print("  generate - Generate random sentences")
            print("  temp <value> - Set temperature (0.1-2.0)")
            print()
            continue
        
        if user_input == "generate":
            print("\nBot: Here are some generated sentences:")
            for sent in generator.generate_batch(3, "mixed"):
                print(f"  - {sent}")
            print()
            continue
        
        if user_input.startswith("temp "):
            try:
                temp = float(user_input.split()[1])
                temp = max(0.1, min(2.0, temp))
                print(f"Bot: Temperature set to {temp}")
            except:
                print("Bot: Invalid temperature. Use 'temp 0.5' for example.")
            continue
        
        # Generate response based on input
        response = generate_response(model, vocab, user_input, generator)
        print(f"Bot: {response}\n")


def generate_response(model, vocab, user_input, generator, max_length=15, temperature=0.7):
    """
    Generate a response to user input.
    
    Args:
        model: Trained model
        vocab: Vocabulary
        user_input: User's message
        generator: SentenceGenerator instance
        max_length: Maximum response length
        temperature: Sampling temperature
        
    Returns:
        Generated response string
    """
    # Encode user input
    input_tokens = vocab.encode(user_input)
    
    # Add BOS token
    bos_idx = vocab.word_to_idx[Vocabulary.BOS_TOKEN]
    eos_idx = vocab.word_to_idx[Vocabulary.EOS_TOKEN]
    
    tokens = [bos_idx] + input_tokens
    generated = []
    
    # Generate response tokens
    for _ in range(max_length):
        next_idx, probs = model.predict_next(tokens, temperature)
        
        # Stop if EOS or special token
        if next_idx == eos_idx or next_idx == bos_idx:
            break
        
        # Skip if it's just repeating the input
        if len(generated) == 0 and next_idx in input_tokens[-3:]:
            # Try to get a different token
            sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
            for idx, prob in sorted_probs[1:5]:
                if idx != eos_idx and idx != bos_idx and idx not in input_tokens[-3:]:
                    next_idx = idx
                    break
        
        generated.append(next_idx)
        tokens.append(next_idx)
        
        # Stop at natural sentence boundaries
        word = vocab.idx_to_word.get(next_idx, "")
        if len(generated) >= 5 and word in [".", "!", "?"]:
            break
    
    if not generated:
        # Fallback: generate a random appropriate response
        fallback_responses = [
            "i understand what you mean",
            "that is an interesting point",
            "let me think about that",
            "i am here to help you",
            "please tell me more",
        ]
        return generator.rng.choice(fallback_responses)
    
    return vocab.decode(generated)


def interactive_mode():
    """Interactive mode for experimentation."""
    print_banner()
    
    print("Options:")
    print("  1. Run all tests")
    print("  2. Train model (quick - 20 epochs)")
    print("  3. Train model (full - 100 epochs)")
    print("  4. Load model and generate")
    print("  5. Show grammar rules")
    print("  6. Chat mode (talk with the model)")
    print("  0. Exit")
    
    vocab = Vocabulary()
    model = None
    
    while True:
        print()
        choice = input("Enter choice (0-6): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        
        elif choice == "1":
            vocab = test_vocabulary()
            model = test_forward_pass(vocab)
            model, trainer = test_training(vocab)
            test_generation(model, vocab)
        
        elif choice == "2":
            model, vocab = train_model(epochs=20, embed_dim=48, num_layers=1, 
                                       batch_size=8, learning_rate=0.005)
        
        elif choice == "3":
            model, vocab = train_model(epochs=100, embed_dim=64, num_layers=2,
                                       batch_size=16, learning_rate=0.001)
        
        elif choice == "4":
            try:
                model = SimpleLLM.load("model.json")
                print("Model loaded successfully!")
                generator = SentenceGenerator(model, vocab)
                
                print("\nGenerating sentences...")
                print("\nPresent continuous:")
                for sent in generator.generate_batch(5, "present_continuous"):
                    print(f"  - {sent}")
                
                print("\nPast participle:")
                for sent in generator.generate_batch(5, "past_participle"):
                    print(f"  - {sent}")
                    
            except FileNotFoundError:
                print("No saved model found. Train a model first!")
        
        elif choice == "5":
            print("\n--- Grammar Rules ---")
            print("\nPresent Continuous Sentence:")
            print("  pronoun + aux + verb(ing) + determiner + adjective + noun +")
            print("  conjunction + pronoun + aux + verb(ing) + particle + verb + noun + noun")
            print("\nExample: 'he is driving the fast car and she is going up make data risk'")
            
            print("\nPast Participle Sentence:")
            print("  determiner + noun + was + past_participle")
            print("\nExample: 'the car was driven'")
            
            print("\nVocabulary Categories:")
            for cat, words in vocab.CATEGORIES.items():
                print(f"  {cat}: {', '.join(words[:5])}...")
        
        elif choice == "6":
            chat_mode()
        
        else:
            print("Invalid choice. Try again.")


def main():
    """Main entry point."""
    args = sys.argv[1:]
    
    if not args:
        interactive_mode()
    
    elif "--train" in args:
        epochs = 100
        embed_dim = 64
        num_layers = 2
        batch_size = 16
        learning_rate = 0.001
        
        # Fast mode: smaller model for quicker testing
        if "--fast" in args:
            epochs = 30
            embed_dim = 32
            num_layers = 1
            batch_size = 8
            learning_rate = 0.005
            print("Using FAST mode (smaller model, fewer epochs)")
        
        # Tiny mode: ultra-fast for testing
        if "--tiny" in args:
            epochs = 10
            embed_dim = 16
            num_layers = 1
            batch_size = 4
            learning_rate = 0.01
            print("Using TINY mode (minimal model for testing)")
        
        if "--epochs" in args:
            idx = args.index("--epochs")
            if idx + 1 < len(args):
                epochs = int(args[idx + 1])
        
        # Turbo mode: 10x faster by training on fewer positions
        turbo = "--turbo" in args
        
        train_model(epochs=epochs, embed_dim=embed_dim, num_layers=num_layers,
                    batch_size=batch_size, learning_rate=learning_rate, turbo=turbo)
    
    elif "--generate" in args:
        vocab = Vocabulary()
        try:
            model = SimpleLLM.load("model.json")
            generator = SentenceGenerator(model, vocab)
            
            print("Generated sentences:")
            for sent in generator.generate_batch(10, "mixed"):
                print(f"  - {sent}")
        except FileNotFoundError:
            print("No saved model found. Train a model first with --train")
    
    elif "--chat" in args:
        chat_mode()
    
    elif "--test-forward" in args:
        vocab = test_vocabulary()
        test_forward_pass(vocab)
    
    elif "--help" in args or "-h" in args:
        print(__doc__)
        print("\nAdditional commands:")
        print("  --train [--fast|--tiny] [--turbo] [--epochs N]  Train the model")
        print("  --generate                                      Generate sentences")
        print("  --chat                                          Interactive chat mode")
        print("  --test-forward                                  Test the forward pass")
    
    else:
        print(f"Unknown arguments: {args}")
        print("Use --help for usage information")


if __name__ == "__main__":
    main()

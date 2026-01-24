"""
Curated article lists for diverse, high-quality Wikipedia content.
These articles are hand-picked to ensure topic diversity and quality.
"""

# Nobel Prize Winners (across all categories)
NOBEL_LAUREATES = [
    # Physics
    "Albert Einstein", "Marie Curie", "Niels Bohr", "Richard Feynman",
    "Max Planck", "Werner Heisenberg", "Erwin Schrödinger", "Paul Dirac",
    "Enrico Fermi", "Wolfgang Pauli", "Peter Higgs", "Stephen Hawking",
    # Chemistry
    "Linus Pauling", "Dorothy Hodgkin", "Ahmed Zewail", "Frances Arnold",
    "Fritz Haber", "Glenn T. Seaborg", "Robert Burns Woodward",
    # Medicine/Physiology
    "Alexander Fleming", "James Watson", "Francis Crick", "Rosalind Franklin",
    "Barbara McClintock", "Tu Youyou", "Robert Koch", "Louis Pasteur",
    # Literature
    "Ernest Hemingway", "Gabriel García Márquez", "Toni Morrison",
    "Rabindranath Tagore", "Albert Camus", "Samuel Beckett", "Bob Dylan",
    "Kazuo Ishiguro", "Doris Lessing", "V. S. Naipaul",
    # Peace
    "Nelson Mandela", "Martin Luther King Jr.", "Malala Yousafzai",
    "Mother Teresa", "Dalai Lama", "Desmond Tutu", "Aung San Suu Kyi",
    "Barack Obama", "Mikhail Gorbachev", "Kofi Annan",
    # Economics
    "John Nash", "Milton Friedman", "Paul Krugman", "Amartya Sen",
    "Daniel Kahneman", "Elinor Ostrom", "Joseph Stiglitz",
]

# Major Inventions and Discoveries
INVENTIONS = [
    "Printing press", "Steam engine", "Telephone", "Light bulb",
    "Radio", "Television", "Computer", "Internet", "World Wide Web",
    "Transistor", "Integrated circuit", "Smartphone", "Electric motor",
    "Internal combustion engine", "Airplane", "Automobile", "Bicycle",
    "Refrigerator", "Air conditioning", "Vaccination", "Antibiotics",
    "Penicillin", "X-ray", "MRI", "DNA sequencing", "CRISPR",
    "Solar cell", "Nuclear power", "Laser", "GPS",
]

# Major Historical Events
HISTORICAL_EVENTS = [
    "French Revolution", "American Revolution", "Industrial Revolution",
    "World War I", "World War II", "Cold War", "Fall of the Berlin Wall",
    "Moon landing", "Renaissance", "Protestant Reformation",
    "Russian Revolution", "Chinese Revolution", "Indian independence movement",
    "Abolition of slavery", "Women's suffrage", "Civil rights movement",
    "Apartheid", "September 11 attacks", "COVID-19 pandemic",
    "Ancient Rome", "Ancient Greece", "Ancient Egypt", "Mongol Empire",
    "British Empire", "Ottoman Empire", "Silk Road", "Age of Exploration",
    "Scientific Revolution", "Enlightenment",
]

# Scientific Concepts and Theories
SCIENTIFIC_CONCEPTS = [
    "Theory of relativity", "Quantum mechanics", "Evolution",
    "Big Bang", "Black hole", "DNA", "Photosynthesis", "Climate change",
    "Plate tectonics", "Periodic table", "Atom", "Molecule",
    "Gravity", "Electromagnetism", "Thermodynamics", "Entropy",
    "Gene", "Cell (biology)", "Virus", "Bacteria", "Ecosystem",
    "Artificial intelligence", "Machine learning", "Neural network",
    "Blockchain", "Cryptography", "Algorithm", "Data structure",
    "Calculus", "Algebra", "Geometry",
]

# Famous Artists, Authors, and Musicians
ARTISTS_AND_AUTHORS = [
    # Visual Artists
    "Leonardo da Vinci", "Michelangelo", "Vincent van Gogh", "Pablo Picasso",
    "Claude Monet", "Rembrandt", "Frida Kahlo", "Salvador Dalí",
    # Authors
    "William Shakespeare", "Jane Austen", "Charles Dickens", "Leo Tolstoy",
    "Fyodor Dostoevsky", "Mark Twain", "Virginia Woolf", "Franz Kafka",
    "Jorge Luis Borges", "Haruki Murakami", "Chinua Achebe",
    # Musicians/Composers
    "Wolfgang Amadeus Mozart", "Ludwig van Beethoven", "Johann Sebastian Bach",
    "The Beatles", "Elvis Presley", "Michael Jackson", "Bob Marley",
    "Frédéric Chopin", "Pyotr Ilyich Tchaikovsky",
]

# Countries and Major Cities
GEOGRAPHY = [
    # Major Countries
    "India", "China", "United States", "Japan", "Germany", "France",
    "United Kingdom", "Brazil", "Russia", "Australia", "Canada",
    "South Africa", "Egypt", "Nigeria", "Mexico", "Indonesia",
    # Major Cities
    "New York City", "London", "Tokyo", "Paris", "Mumbai", "Shanghai",
    "São Paulo", "Cairo", "Sydney", "Singapore", "Dubai", "Rome",
    # Natural Wonders
    "Amazon rainforest", "Himalayas", "Great Barrier Reef", "Sahara",
    "Grand Canyon", "Mount Everest", "Niagara Falls",
]

# Philosophy and Thinkers
PHILOSOPHY = [
    "Socrates", "Plato", "Aristotle", "Confucius", "Buddha",
    "Immanuel Kant", "Friedrich Nietzsche", "Karl Marx", "John Locke",
    "René Descartes", "David Hume", "Jean-Paul Sartre", "Simone de Beauvoir",
    "Mahatma Gandhi", "Existentialism", "Utilitarianism", "Stoicism",
]

# Technology Companies and Pioneers
TECH_PIONEERS = [
    "Steve Jobs", "Bill Gates", "Elon Musk", "Jeff Bezos", "Mark Zuckerberg",
    "Tim Berners-Lee", "Alan Turing", "Ada Lovelace", "Grace Hopper",
    "Apple Inc.", "Microsoft", "Google", "Amazon (company)", "Tesla, Inc.",
]

def get_all_curated_articles():
    """Return all curated articles as a flat list."""
    all_articles = []
    all_articles.extend(NOBEL_LAUREATES)
    all_articles.extend(INVENTIONS)
    all_articles.extend(HISTORICAL_EVENTS)
    all_articles.extend(SCIENTIFIC_CONCEPTS)
    all_articles.extend(ARTISTS_AND_AUTHORS)
    all_articles.extend(GEOGRAPHY)
    all_articles.extend(PHILOSOPHY)
    all_articles.extend(TECH_PIONEERS)
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for article in all_articles:
        if article.lower() not in seen:
            seen.add(article.lower())
            unique.append(article)
    
    return unique

def get_curated_by_category():
    """Return curated articles organized by category."""
    return {
        "nobel_laureates": NOBEL_LAUREATES,
        "inventions": INVENTIONS,
        "historical_events": HISTORICAL_EVENTS,
        "scientific_concepts": SCIENTIFIC_CONCEPTS,
        "artists_and_authors": ARTISTS_AND_AUTHORS,
        "geography": GEOGRAPHY,
        "philosophy": PHILOSOPHY,
        "tech_pioneers": TECH_PIONEERS,
    }


if __name__ == "__main__":
    articles = get_all_curated_articles()
    print(f"Total curated articles: {len(articles)}")
    
    by_cat = get_curated_by_category()
    for cat, items in by_cat.items():
        print(f"  {cat}: {len(items)}")

import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory: str) -> dict:
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

def transition_model(corpus: dict, page: str, damping_factor: float) -> dict:
    """
    Return a probability distribution over which page to visit next, given a current page.

    With probability `damping_factor`, choose a link at random linked to by `page`.
    With probability `1 - damping_factor`, choose a link at random chosen from all pages in the corpus.
    """
    links = corpus.get(page, set())
    N = len(corpus)
    transitions = {}

    if not links:
        # If there are no outgoing links, distribute probability uniformly across all pages
        for p in corpus:
            transitions[p] = 1 / N
    else:
        # Distribute probability
        for p in corpus:
            transitions[p] = (1 - damping_factor) / N  # Base probability for all pages
        for link in links:
            transitions[link] += damping_factor / len(links)  # Add linked-page probability

    return transitions

def sample_pagerank(corpus: dict, damping_factor: float, n: int) -> dict:
    """
    Return PageRank values for each page by sampling `n` pages
    according to the transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Initialize the PageRank counts for each page
    pagerank_counts = {page: 0 for page in corpus}

    # Start by choosing a random page
    current_page = random.choice(list(corpus.keys()))

    # Perform sampling for `n` iterations
    for _ in range(n):
        # Count the current page
        pagerank_counts[current_page] += 1

        # Get the transition model for the current page
        transitions = transition_model(corpus, current_page, damping_factor)

        # Choose the next page based on the transition model
        current_page = random.choices(list(transitions.keys()), weights=transitions.values())[0]

    # Normalize the counts to get probabilities
    pagerank = {page: count / n for page, count in pagerank_counts.items()}

    return pagerank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Step 1: Initialize PageRank values
    N = len(corpus)
    pagerank = {page: 1 / N for page in corpus}
    tolerance = 0.001

    # Step 2: Iteratively update PageRank until convergence
    converged = False
    while not converged:
        new_pagerank = {}
        for page in corpus:
            # Calculate the random jump probability
            new_rank = (1 - damping_factor) / N

            # Calculate the contribution from all pages that link to this page
            for potential_linker in corpus:
                if page in corpus[potential_linker]:
                    # Distribute the PageRank from the linking page
                    new_rank += damping_factor * (pagerank[potential_linker] / len(corpus[potential_linker]))

                # Handle the case where a page has no outgoing links (distribute equally to all pages)
                if not corpus[potential_linker]:
                    new_rank += damping_factor * (pagerank[potential_linker] / N)

            new_pagerank[page] = new_rank

        # Check for convergence (if all differences are below the tolerance)
        converged = all(abs(new_pagerank[page] - pagerank[page]) < tolerance for page in pagerank)

        # Update the PageRank values for the next iteration
        pagerank = new_pagerank

    return pagerank

if __name__ == "__main__":
    main()
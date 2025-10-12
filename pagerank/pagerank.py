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


def crawl(directory):
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


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    p = {}
    N = len(corpus)
    links = corpus[page]

    if links:
        link_prob = damping_factor / len(links)
        for link in links:
            p[link] = link_prob
    else:
        link_prob = damping_factor / N
        for pg in corpus:
            p[pg] = link_prob

    teleport = (1 - damping_factor) / N
    for pg in corpus:
        if pg in p:
            p[pg] += teleport
        else:
            p[pg] = teleport

    return p


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    current = random.choice(list(corpus.keys()))
    ranks = {pg: 0 for pg in corpus}
    ranks[current] += 1

    for _ in range(n - 1):
        trans = transition_model(corpus, current, damping_factor)
        next_page = random.choices(list(trans.keys()), weights=list(trans.values()))[0]
        current = next_page
        ranks[current] += 1

    total = sum(ranks.values())
    for pg in ranks:
        ranks[pg] /= total

    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    N = len(corpus)
    ranks = {pg: 1.0 / N for pg in corpus}

    while True:
        new_ranks = {pg: 0.0 for pg in corpus}

        uniform = (1 - damping_factor) / N
        for pg in corpus:
            new_ranks[pg] = uniform
            contrib = 0.0
            for q in corpus:
                if len(corpus[q]) > 0 and pg in corpus[q]:
                    contrib += ranks[q] / len(corpus[q])
            new_ranks[pg] += damping_factor * contrib

        zero_mass = sum(ranks[q] for q in corpus if len(corpus[q]) == 0)
        zero_contrib = damping_factor * zero_mass / N

        for pg in corpus:
            new_ranks[pg] += zero_contrib

        max_diff = 0.0
        for pg in corpus:
            diff = abs(new_ranks[pg] - ranks[pg])
            if diff > max_diff:
                max_diff = diff

        ranks = new_ranks

        if max_diff < 0.001:
            break

    return ranks


if __name__ == "__main__":
    main()

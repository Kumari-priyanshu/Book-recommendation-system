
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load cleaned books
books = pd.read_csv("data/processed/books_clean.csv")

# Load tag information
book_tags = pd.read_csv("data/raw/book_tags.csv")
tags = pd.read_csv("data/raw/tags.csv")

# Connect tag IDs to tag names
book_tags = book_tags.merge(tags, on="tag_id")

# Select the most frequent 10 tags per book
book_tags = book_tags.sort_values(
    ["goodreads_book_id", "count"],
    ascending=[True, False]
)

top_tags = (
    book_tags.groupby("goodreads_book_id")["tag_name"]
    .apply(lambda x: " ".join(x.head(10)))
    .reset_index()
)

# Connect tags with book information
books = books.merge(
    top_tags,
    on="goodreads_book_id",
    how="left"
)

books["tag_name"] = books["tag_name"].fillna("")

# Combine important book features
books["features"] = (
    books["title"].fillna("") + " " +
    books["authors"].fillna("") + " " +
    books["tag_name"]
)

# Convert text into numerical vectors
vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    books["features"]
)

# Calculate similarity between books
similarity = cosine_similarity(tfidf_matrix)


def recommend_books(book_title, top_n=5):

    matches = books[
        books["title"].str.lower() ==
        book_title.lower()
    ]

    if matches.empty:
        print("Book not found!")
        return

    index = matches.index[0]

    scores = list(enumerate(similarity[index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:top_n + 1]

    print("\nBooks recommended for:", book_title)

    for i, score in scores:
        print(
            books.iloc[i]["title"],
            "| Similarity:",
            round(score, 3)
        )


if __name__ == "__main__":
    recommend_books("The Hunger Games")


import pandas as pd

books = pd.read_csv("data/raw/books.csv")
ratings = pd.read_csv("data/raw/ratings.csv")

# Remove duplicate books
books = books.drop_duplicates(subset="book_id")

# Remove books without titles
books = books.dropna(subset=["title"])

# Remove duplicate user-book ratings
ratings = ratings.drop_duplicates(
    subset=["user_id", "book_id"],
    keep="last"
)

# Remove invalid ratings
ratings = ratings[
    ratings["rating"].between(1, 5)
]

# Save cleaned datasets
books.to_csv("data/processed/books_clean.csv", index=False)
ratings.to_csv("data/processed/ratings_clean.csv", index=False)

print("Cleaning completed!")
print("Books remaining:", len(books))
print("Ratings remaining:", len(ratings))

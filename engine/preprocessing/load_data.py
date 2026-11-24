
import pandas as pd

# Load datasets
books = pd.read_csv("data/raw/books.csv")
ratings = pd.read_csv("data/raw/ratings.csv")
tags = pd.read_csv("data/raw/tags.csv")
book_tags = pd.read_csv("data/raw/book_tags.csv")
to_read = pd.read_csv("data/raw/to_read.csv")

print("Books:", books.shape)
print("Ratings:", ratings.shape)
print("Tags:", tags.shape)
print("Book Tags:", book_tags.shape)
print("To Read:", to_read.shape)

print("\nFirst 5 books:")
print(books[['book_id', 'title', 'authors', 'average_rating']].head())

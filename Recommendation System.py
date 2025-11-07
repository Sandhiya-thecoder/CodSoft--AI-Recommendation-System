import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = {
    'Title': [
        'The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Pulp Fiction', 'Batman Begins', 'The Shawshank Redemption', 'Fight Club', 'Forrest Gump', 'The Godfather', 'The Lord of the Rings', 'The Social Network', 'Gladiator', 'The Avengers', 'Jurassic Park', 'Titanic', 'Avatar', 'The Wolf of Wall Street', 'La La Land', 'Mad Max: Fury Road'
    ],
    'Description': [
        'A computer hacker learns about the true nature of his reality and his role in the war against its controllers.',
        'A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
        'When the menace known as the Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.',
        'The lives of two mob hitmen, a boxer, a gangster\'s wife, and a pair of diner bandits intertwine in four tales of violence and redemption.',
        'After witnessing his parents\' murder as a child, Bruce Wayne trains himself to fight crime and becomes the masked vigilante known as Batman.',
        'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
        'An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into something much, much more.',
        'The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other history unfold through the eyes of an Alabama man with an IQ of 75.',
        'An organized crime dynasty\'s aging patriarch transfers control of his clandestine empire to his reluctant son.',
        'An unusual alliance forms when a group of hobbits set out to destroy a powerful ring sought by the Dark Lord Sauron.',
        'As Harvard students, Mark Zuckerberg and Eduardo Saverin created a social networking site that would become known as Facebook.',
        'A former Roman general sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery.',
        'Earth\'s mightiest heroes must come together and learn to fight as a team if they are going to stop the mischievous Loki and his alien army.',
        'During a preview tour, a theme park suffers a major power breakdown that allows its cloned dinosaur exhibits to run amok.',
        'A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.',
        'In the 22nd century, a paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.',
        'Based on the true story of Jordan Belfort, from his rise to a wealthy stockbroker living the high life to his fall involving crime, corruption and the federal government.',
        'Struggling with the aftermath of a failed relationship, a jazz musician discovers a new passion for life and love in the vibrant city of Los Angeles.',
        'In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland.'
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Description'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_movies(title, cosine_similarities=cosine_similarities):
    if title not in df['Title'].values:
        return "Movie not found in database."

    idx = df.index[df['Title'] == title][0]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    
    print("Top 5 movie recommendations for '{}':".format(title))
    for i, score in sim_scores:
        movie_title = df['Title'].iloc[i]
        similarity = score
        print(f"{movie_title} (Similarity Score: {similarity:.4f})")

user_input = input("Enter a movie title: ").strip()
recommend_movies(user_input)

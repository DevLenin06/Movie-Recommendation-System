# Movie Recommendation System (in-progress)

## Overview
This project aims to build a movie recommendation system using collaborative filtering. The system predicts which movies a user is likely to enjoy based on the ratings and preferences of similar users. The model is currently in progress and is being developed using data preprocessing, exploratory data analysis (EDA), and machine learning techniques. 

The dataset contains movie ratings by various users, along with details such as genres, titles, and years of release. The goal is to recommend movies to users based on their past interactions and preferences.

## Dataset
The dataset contains the following features:
- **userId**: Unique identifier for each user.
- **movieId**: Unique identifier for each movie.
- **rating**: Rating given by the user (scale: 0-5).
- **timestamp**: The time at which the rating was made.
- **genres**: Movie genres (such as Action, Comedy, Drama, etc.).
- **title**: Title of the movie.
- **year**: The release year of the movie.

### Dataset Highlights
- Total entries: 100,000+ (approx. 100,000 movie ratings).
- Features: 6 (userId, movieId, rating, timestamp, genres, title).
- Missing values: Minor missing data handled appropriately.

### Data Imbalance:
- There is a slight imbalance in the number of ratings across different movies. Popular movies have more ratings compared to less popular ones.

## Tools and Libraries Used

### Python Libraries:
- `pandas`, `numpy`: For data manipulation and analysis.
- `matplotlib`, `seaborn`: For data visualization.
- `scikit-learn`: For machine learning algorithms and preprocessing.
- `imbalanced-learn`: For oversampling techniques to handle data imbalance.
- `surprise`: For collaborative filtering model building (if used in the future).

## Process

### 1. Data Preprocessing
- **Handling Missing Values**: Missing values in the dataset were handled by imputing values (e.g., replacing missing ratings with zero or mean).
- **Feature Engineering**: I created new features such as extracting genres and release year from movie titles. I also converted categorical variables (genres, titles) into numerical representations using OneHotEncoding and Label Encoding.
- **Data Merging**: Combined the ratings data with movie data (titles and genres) for a better feature set.

### 2. Exploratory Data Analysis (EDA)
- Visualized rating distributions and their relationship with movie genres.
- Analyzed user rating patterns and movie popularity.
- Highlighted the imbalances in ratings for popular and non-popular movies.

### 3. Balancing the Data
- **Addressing Data Imbalance**: As popular movies tend to have more ratings, we considered using techniques like oversampling (SMOTE) or downsampling to balance the dataset before training models.

### 4. Building the Model
- **Model**: Collaborative filtering based on user ratings (could later include content-based filtering or hybrid models).
- **Preprocessing**:
  - Standardized or normalized ratings for model training.
  - Split data into training (80%) and testing (20%) sets.
- **Model**: Logistic Regression, Random Forest, or other algorithms for recommendations (still in-progress).
- **Evaluation**: Cross-validation to evaluate model accuracy, precision, recall, and other metrics.
  
### 5. Evaluating the Model (still in-progress)
- **Cross-validation Score**: 70-80% (based on current evaluation)
- **Metrics**:
  - Precision and Recall to evaluate the effectiveness of the recommendations.
  - ROC-AUC curve to understand the model's performance.
  - Confusion Matrix to evaluate correct and incorrect predictions.

## Key Results (in-progress)

## Visualizations (in-progress)
- Distribution of ratings for different genres.
- Count plots to visualize user ratings across genres.
- ROC and Precision-Recall curves to evaluate the model's performance.

## Future Work
- **Advanced Models**: Experiment with more advanced models such as Neural Networks for better prediction accuracy.
- **Hybrid Recommendation System**: Combine collaborative filtering with content-based methods to improve recommendation quality.
- **Include More Features**: Incorporate additional movie features, such as director, cast, or movie descriptions, for more even more personalized recommendations.
- **Exploring Other Data Balancing Methods**: Experiment with other oversampling and undersampling techniques to further improve model performance.

## How to Use This Project

### 1. Install Required Libraries:
Install all dependencies using `pip`:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn surprise
2. Run the Script:
Ensure that the dataset (ratings.csv and movies.csv) is located in the same directory as the script.
Execute the script to preprocess data, train the model, and view the results.
bash
Copy code
python movie_recommendation_model.py
3. View Visualizations:
The script generates several plots that help visualize the dataset distribution and model performance. You can modify the code to output different types of visualizations based on your needs.

File Structure
bash
Copy code
movie-recommendation-project/
│
├── data/
│   ├── ratings.csv         # User ratings data
│   ├── movies.csv          # Movie information data
│
├── src/
│   ├── movie_recommendation_model.py  # Python script for training the recommendation model
│   ├── data_preprocessing.py          # Data preprocessing script
│
├── notebooks/
│   ├── Movie_Recommendation_Analysis.ipynb  # Jupyter notebook for data analysis
│
├── requirements.txt        # Required Python dependencies
└── README.md               # Project documentation
Conclusion
This project demonstrates how a movie recommendation system can be built using collaborative filtering techniques. Although still in progress, it lays the groundwork for exploring more advanced recommendation algorithms and improving the system's performance. The next steps include optimizing the current model and experimenting with hybrid systems to deliver better recommendations. 

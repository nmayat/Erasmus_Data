# Where you should do your Erasmus programm

Currently, I'm living it up in Valencia, Spain, where I've landed an amazing internship opportunity. Thanks to my ongoing studies, I was able to snag an Erasmus+ grant to make it all possible. This isn't my first rodeo, though – I previously enjoyed an Erasmus semester as a student in Córdoba. I know, I know... I'm a total Spain enthusiast! But can you blame me? The weather's fantastic, the food is to die for, and the people are just so warm and welcoming. Plus, as an aspiring data scientist, what better way to dive into my passion than by exploring Erasmus data and turning it into a cool project?

For those of you who aren't familiar with the Erasmus+ program (https://data.europa.eu/data/datasets/erasmus-mobility-statistics-2014-2019-v2?locale=en), let me give you a quick rundown: Erasmus+ is the European Union's initiative to promote education, training, youth, and sport across Europe. The program offers various mobility opportunities, such as studying or interning abroad. With an impressive budget of around €26.2 billion since 2020, the European Commission allocates funds to support these projects and foster international cooperation.

Now, I can tell you from personal experience that picking your dream destination can be a bit of a headache. After all, Europe is a treasure trove of amazing places waiting to be explored. That's why I thought, "Why not create a recommendation system to help make that decision a little easier?" And so, the idea was born! As a bonus, the European Commission generously provides yearly data on the Erasmus+ program: https://data.europa.eu/data/datasets/erasmus-mobility-statistics-2014-2019-v2?locale=en.


## Data Exploration
After cleaning the data I used Plotly to create some engaging visualizations that offer insights into the dataset and help us understand the program better.

### Sending and Recieving countries
As expected, the majority of participants come from EU member countries, with more populous countries like Germany and France sending the most students. However, we also see grants awarded to people from outside Europe, such as Turkey, which had nearly as many participants as Poland. Surprisingly, even 35 people from Fiji took part in the program!

Looking at the receiving countries, Spain tops the list (I know, I'm not that unique). Over the five years analyzed, more than half a million people chose Spain as their Erasmus destination. Other popular countries include Germany, the UK, and Italy. Interestingly, while France sent the second-highest number of participants, only around 280,000 people chose to go there.

In a separate plot, I highlighted the mobility patterns of German participants. The UK emerged as their top choice, followed by France and Spain. The data also revealed that Erasmus+ funded programs within Germany, which might include events like national youth meetings.

### Activities
I was amazed by the diverse opportunities offered by the Erasmus+ program. While I was familiar with student mobility for studies and traineeships (having participated myself), I discovered that these were just the tip of the iceberg. Over 1.2 million students took part in the program for studies alone. However, the program also supports various other activities, such as national youth meetings and job shadowing.

### Participants
An interesting observation from the data is the higher participation of girls and women, with nearly 700,000 fewer men involved. This substantial difference is quite notable. As for the age distribution, a large portion of participants are in their early twenties, which makes sense given the program's focus on students and people in educational systems.

When analyzing the mean age per activity, we find that certain fields attract participants over thirty. Job shadowing and staff training abroad, for example, typically involve people already in their careers. This observation highlights the success of the Erasmus+ program in promoting lifelong learning, aligning with its mission to support a wide range of individuals.

## Recommendation system
Now that we've explored the dataset, let's dive into the machine learning aspect of our project – recommending countries for prospective participants. The idea is simple: a person provides their information, and we suggest possible destinations for their program based on similarity. Since we don't have feedback on individual experiences, we'll use the popular k-Nearest Neighbors (kNN) algorithm to make our recommendations.

kNN is a versatile machine learning algorithm employed for both classification and regression tasks. It functions by identifying the k nearest data points to a new observation and assigning the label or value of the majority of those neighbors to that observation.

Now let's dive into the code! We'll begin by encoding our categorical features using Scikit-learn's LabelEncoder:

encoder = LabelEncoder()
categorical_columns = ['Activity', 'Field of Education', 'Education Level', 'Sending Country', 'Receiving Country']
encoded_df = data.copy()
for column in categorical_columns:
    encoded_df[column] = encoder.fit_transform(data[column])

Next, we'll set up our data preprocessing pipeline using OneHotEncoder for categorical features and StandardScaler for numerical features:

cat_transformer = OneHotEncoder()
num_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, ['Mobility Duration', 'Participant Age']),
        ('cat', cat_transformer, ['Activity', 'Field of Education', 'Education Level', 'Sending Country'])
    ],sparse_threshold=0)

The receiving country is the feature we want to recommend, so we'll prepare our training and test sets accordingly:

X = encoded_df.drop('Receiving Country', axis=1)
y = encoded_df['Receiving Country']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)

preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)

We want our model to recommend the receiving country based on the 5 closest neighbors of our input:

model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='euclidean')
model.fit(X_train)

As with any ML model, we need a metric to evaluate its performance. For our recommendation system, one possible metric would be to see if the actual receiving country is among the list of recommended countries. However, we also want to measure if the actual country appears among the model's more popular recommendations. After discussing this with Chat GPT, I decided on using the Mean Reciprocal Rank as our evaluation metric. To calculate the Mean Reciprocal Rank (MRR), we'll use the following formula:

MRR = mean(1/rank)

For example, if the actual country is the second most popular recommendation, the rank would be 2, and the MRR value would be 0.5.

Considering the large size of our dataset, using a 20% test-train split would yield a significant number of rows for testing the algorithm. To speed up the testing process, we'll divide the testing dataset into smaller batches:

def process_test_data_in_batches(X_test, y_test, model, batch_size=10):
    reciprocal_ranks = []
    num_batches = int(np.ceil(X_test.shape[0] / batch_size))

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, X_test.shape[0])

        X_test_batch = X_test[start_idx:end_idx]
        y_test_batch = y_test.iloc[start_idx:end_idx]

        distances, indices = model.kneighbors(X_test_batch)


        y_pred_batch = [y_train.iloc[indices_row].values for indices_row in indices]
        y_true_batch = y_test_batch.values

        # Calculate reciprocal rank for each instance in the batch
        batch_reciprocal_ranks = [
            1 / (np.where(y_pred == y_true)[0][0] + 1)
            if y_true in y_pred else 0
            for y_true, y_pred in zip(y_true_batch, y_pred_batch)
        ]

        reciprocal_ranks.extend(batch_reciprocal_ranks)
    return np.mean(reciprocal_ranks)


Now let's create a function to recommend receiving countries based on an individual's information:

def recommend_receiving_country(student_info):
    # Encode the student_info
    encoded_student_info = student_info.copy()
    for i, column in enumerate(categorical_columns[:-1]):
        encoded_student_info[column] = encoder.fit(data[column]).transform([student_info[column]])[0]

    # Transform the student_info
    student_X = preprocessor.transform(encoded_student_info)

    # Find the k-nearest neighbors
    distances, indices = model.kneighbors(student_X)

    # Get the corresponding receiving countries
    recommendations = data.iloc[indices[0]]['Receiving Country'].values

    return np.unique(recommendations)


Finally, let's see the output of our recommendation function for a given student's information.

student_info = pd.DataFrame([{'Mobility Duration': 180, 'Activity': 'Student mobility for traineeships between Programme Countries', 'Field of Education': 'Physical sciences',
                              'Education Level': 'Second cycle / Master’s or equivalent level', 'Participant Age': 24,
                              'Sending Country': 'Germany'}])

['Italy' 'Portugal' 'Slovakia' 'Spain']

Not bad at all! I went to Spain, so it seems like a reasonable recommendation.

Now, let's try something a bit more exotic:

student_info = pd.DataFrame([{'Mobility Duration': 90, 'Activity': 'Individual Volunteering Activities', 'Field of Education': 'Fine arts',
                              'Education Level': 'Third cycle / Doctoral or equivalent level', 'Participant Age': 35,
                              'Sending Country': 'Lesotho'}])
['Lithuania' 'Netherlands' 'Portugal' 'Romania' 'Spain']

So, Lithuanians, if in the near future you get visited by a vast amount of doctoral students in Fine Arts from Lesotho, you'll know why!

## Summary
I had a blast diving into the Erasmus data provided by the EU. Of course, the decision to study or work in another country depends on various factors, such as language skills, available institutions for your program, and even personal preferences like weather. In my case, the sunshine definitely played a role in choosing Spain!

The recommendation engine I built is fairly simple, but it does a decent job. More sophisticated methods, like using deep learning, might yield better results. On the other hand, this recommendation model could spark new ideas for someone contemplating their next destination.

So, if you're considering an Erasmus+ experience, let this model serve as a fun starting point to explore the possibilities that Europe has to offer!

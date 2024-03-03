# Import necessary modules
from sklearn import svm  # Import Support Vector Machines module
from sklearn.feature_extraction.text import CountVectorizer  # Import CountVectorizer for text processing

# Sample training data
training_data = [
    {"input": "hello", "category": "greeting"},
    {"input": "how are you", "category": "greeting"},
    {"input": "bye", "category": "farewell"},
    {"input": "goodbye", "category": "farewell"},
    {"input": "tell me a joke", "category": "humor"},
    {"input": "knock knock", "category": "humor"}
]

# Extract features and labels from training data
corpus = [item["input"] for item in training_data]  # Extract input data from training data
labels = [item["category"] for item in training_data]  # Extract corresponding categories

# Initialize a CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer()  # Create an instance of CountVectorizer

# Transform the text data into a sparse matrix of token counts
X = vectorizer.fit_transform(corpus)  # Fit and transform the text data into numerical features

# Train an SVM classifier with a linear kernel
clf = svm.SVC(kernel='linear')  # Create an instance of SVM with a linear kernel
clf.fit(X, labels)  # Train the SVM classifier with the features and labels

# Function to get the category of a user input using the trained SVM model
def get_category(user_input):
    user_input_vectorized = vectorizer.transform([user_input])  # Vectorize the user input
    prediction = clf.predict(user_input_vectorized)  # Predict the category using the trained SVM model
    return prediction[0]

# Function to get a response based on the predicted category
def get_response(category):
    if category == "greeting":
        return "Hi there! How can I help you?"
    elif category == "farewell":
        return "Goodbye! Have a great day."
    elif category == "humor":
        return "Why did the computer a catch cold? bcz it had too many WINDOWS open!"

# Main function for running the chatbot
def main():
    print("Chatbot: Hi! I'm a simple SVM-based chatbot. Type 'bye' to exit.")

    while True:  # start of loop
        user_input = input("You: ")
        if user_input.lower() == 'bye':  # checking if user input is bye
            print("Chatbot: Goodbye!")  # print this
            break

        category = get_category(user_input)  # getting into any one category
        response = get_response(category)  # replying response
        print("Chatbot:", response)  # printing response

# Run the main function if the script is executed
if __name__ == "__main__":
    main()

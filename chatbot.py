# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import numpy as np
import re
from porter_stemmer import PorterStemmer


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moomin'

        self.creative = creative
        self.stemmer = PorterStemmer()
        self.negation = ["not", "didn't", "no", "don't", "never", "can't"]
        self.pos_words = ["love", "ador","enjoi","favorit","best","perfect"]
        self.neg_words = ["hate","loath","dispis","terrible","horrible","worst","appal","digust"]
        self.intensifier = ["so", "veri", "realli", "reealli", "extrem"]
        self.min_num_of_movies = 5
        self.num_ratings = 0

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.sentiment['enjoi'] = 'pos'
        self.sentiment['fun'] = 'pos'
        self.sentiment['cool'] = 'pos'

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        self.user_ratings = np.zeros(np.shape(self.ratings)[0])
        
        # Keeps track of number of recs given so far
        self.recs_given = 0

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "hi there! I'm Moomin, a movie bot! :) i'll help you find movies you'll like. can you start by listing some movies you've enjoyed in the past?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "bye! have a nice day ☀︎"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
       """Process a line of input from the REPL and generate a response.
 
       This is the method that is called by the REPL loop directly with user
       input.
 
       You should delegate most of the work of processing the user's input to
       the helper functions you write later in this class.
 
       Takes the input string from the REPL and call delegated functions that
         1) extract the relevant information, and
         2) transform the information into a response to the user.
 
       Example:
         resp = chatbot.process('I loved "The Notebook" so much!!')
         print(resp) // prints 'So you loved "The Notebook", huh?'
 
       :param line: a user-supplied line of text
       :returns: a string containing the chatbot's response to the user input
       """
       ########################################################################
       # TODO: Implement the extraction and transformation in this method,    #
       # possibly calling other functions. Although your code is not graded   #
       # directly based on how modular it is, we highly recommended writing   #
       # code in a modular fashion to make it easier to improve and debug.    #
       ########################################################################
 
       if self.creative:
           response = "i processed {} in creative mode!!".format(line)
       else:
 
           # Tracks number of recommendations given
           if self.recs_given == 0:
               title = self.extract_titles(line)
 
               # Deals with errors
               if len(title) > 1:
                   return "please give me just one title at a time (๑•﹏•)⋆* ⁑⋆* "
               elif len(title) == 0:
                   return "please put the movie title in quotation marks. for example, “Twilight (2008)”."
               else:
                   movies = self.find_movies_by_title(title[0])
 
                   # Deals with errors
                   if len(movies) > 1:
                       return "looks like there's than one movie matching that title. can you specify the year in which your movie was released?"
                   elif len(movies) == 0:
                       return "oops! i haven’t seen that movie yet (´･_･`). please try another title!"
                   else:
                       sentiment = self.extract_sentiment(str(line))
 
                       # Positive sentiment
                       if sentiment == 1:
                           response = "ok, you liked {}!".format(title[0])
                           self.user_ratings[movies[0]] = sentiment
                           self.num_ratings += 1
                       # Negative sentiment
                       elif sentiment == -1:
                           response = "so you didn't like {}. :(".format(title[0])
                           self.user_ratings[movies[0]] = sentiment
                           self.num_ratings += 1
                       # Neutral sentiment
                       elif sentiment == 0:
                           return "i’m unsure if you liked {}. tell me more about it".format(title[0])
                       # Unintelligible
                       else:
                           return "sorry, I didn't get that. tell me about a movie you've watched"
 
                       # Provide recommendation after 5 ratings
                       if self.num_ratings >= self.min_num_of_movies:
                           response +=  "i’ve gathered enough information to recommend you a movie. hang tight! ヽ(＾Д＾)ﾉ"
                           recommendation = self.recommend(self.user_ratings, self.ratings, k = 20).pop(0)
                           response += "i think you’ll really enjoy watching  \"{}\".".format(
                               self.titles[recommendation][0])
                           response += "want another?"
                           # Tracks number of recs given so far
                           self.recs_given += 1
                       else:
                           response += " tell me another"
 
           # Situation for when current number of recs given is > 0
           else:
               yeses = ['yes', 'yep', 'yah', 'yeah', 'sure', 'yup', 'yea', 'yuh', 'definitely']
               for yes in yeses:
                   nos = ['no', 'nah', 'nope', 'no thank you', 'no thanks']
                   if yes in line.lower():
                       if self.recs_given < 19:
                           # Give another recommendation
                           recommendation = self.recommend(self.user_ratings, self.ratings, k = 20).pop(self.recs_given)
                           response = "\"{}\" would be a good movie for you too.".format(self.titles[recommendation][0])
                           response += " want another?"
                           self.recs_given += 1
                           for no in nos:
                               if no in line.lower():
                                   response = "Okay thank you! Have a good day."
                                   # return response
                                   # quit()
                       elif self.recs_given == 19:
                           # Give rec
 
                           recommendation = self.recommend(self.user_ratings, self.ratings, k = 20).pop(self.recs_given)
                           response = "\"{}\" would be a good movie for you too.".format(self.titles[recommendation][0])
                           response += '\n' + " I am out of recommendations. Please tell me more preferences so I can give you more."
 
                           self.recs_given = 0
                       else:
                           # Start over / cancel recs
                           response = "ok let's go back to movies you like so i can give better recs. tell me more"
                           self.recs_given = 0
 
               nos = ['no', 'nah', 'nope', 'no thank you', 'no thanks']
               for no in nos:
                   if no in line.lower():
                       response = "okay thank you! have a good day :-)"
                       # return response
                       # quit()
 
               if line.lower() not in yeses and line.lower() not in nos:
                   response = "sorry i didn't get that. please enter 'yes' or 'no'."
 
 
       ########################################################################
       #                          END OF YOUR CODE                            #
       ########################################################################
       return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        result = re.findall('"([^"]*)"', preprocessed_input)

        return result

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        has_date = False
        title = title.title()
        pattern = "^(An|A|The)?\s?(.+?)\s?(\((?:\d{4})\))?$"
        m = re.search(pattern, title)
        processed_title = m.group(2)

        if m.group(1) != None:
            processed_title += ", " + m.group(1)

        if m.group(3) != None:
            has_date = True
            date = m.group(3)

        index = []

        for i in range(len(self.titles)):
            movie = self.titles[i][0]
            m2 = re.search("(.+?)(?:\s(\((?:\d{4})\)))?$", movie)
            
            if self.creative:
                if re.search(r'\b({0})\b'.format(title), movie) != None:
                    index.append(i)
            
            if processed_title.lower() == m2.group(1).lower():
                if has_date:
                    if m2.group(2) != None and date == m2.group(2):
                        if not i in index:
                            index.append(i)
                else:
                    if not i in index:
                        index.append(i)                

        return index

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        score = 0
        processed_input = re.sub('"([^"]*)"', "[movie]", preprocessed_input)
        processed_input = re.sub("[\.\,\!\?\:\;]", "", processed_input).split()
        negation_on = False
        intensifer_on = False

        for i in range(len(processed_input)):
            cur_word = processed_input[i].lower()

            if cur_word in self.negation:
                negation_on = True
            
            if not cur_word in self.sentiment:
                cur_word = self.stemmer.stem(cur_word)

            if cur_word in self.pos_words or cur_word in self.neg_words or cur_word in self.intensifier:
                intensifer_on = True

            if (cur_word in self.sentiment and self.sentiment[cur_word] == 'neg') or cur_word in self.neg_words:
                if negation_on: 
                    score += 1
                else:
                    score -= 1
            elif (cur_word in self.sentiment and self.sentiment[cur_word] == 'pos') or cur_word in self.pos_words:
                if negation_on: 
                    score -= 1
                else:
                    score += 1
            
        if score > 0:
            if intensifer_on:
                return 2
            return 1
        elif score < 0:
            if intensifer_on:
                return -2
            return -1
        else:
            return 0
 

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        movies = self.extract_titles(preprocessed_input)

        self.similar = ["and", "or", "nor"]
        self.opposite = ["but", "yet"]
        processed_input = re.sub('"([^"]*)"', "[movie]", preprocessed_input)

        for conjunction in self.similar:
            processed_input = re.sub(conjunction, "*", processed_input)
        
        for conjunction in self.opposite:
            processed_input = re.sub(conjunction, "*|", processed_input)

        processed_input = re.split('\*', processed_input)

        sentiments = []
        oppositeSentiment = False
        for index in range(len(processed_input)):
            if processed_input[index][0] == "|":
                oppositeSentiment = True
            score = self.extract_sentiment(processed_input[index])
            if score == 0 and index != 0:
                if oppositeSentiment:
                    score = sentiments[index - 1][1] * -1
                else:
                    score = sentiments[index - 1][1]
            
            sentiments.append((movies[index], score))

        return sentiments

    def levenshtein(self, seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros ((size_x, size_y))
        for x in range(size_x):
            matrix [x, 0] = x
        for y in range(size_y):
            matrix [0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix [x,y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1], matrix[x, y-1] + 1)
                else:
                    matrix [x,y] = min(matrix[x-1,y] + 1, matrix[x-1,y-1] + 2, matrix[x,y-1] + 1)
                    
        return matrix[size_x - 1, size_y - 1]
    
    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """
        potentials = [[] for i in range(max_distance + 1)]
        title = title.capitalize()
        m = re.search("^(An|A|The)?\s?(.+?)\s?(\((?:\d{4})\))?$", title)
        
        if m.group(1) != None:
            title += ", " + m.group(1)

        for mov in range(len(self.titles)):
            m2 = re.search("(.+?)(?:\s(\((?:\d{4})\)))?$", self.titles[mov][0]) 
            edit_distance = int(self.levenshtein(m.group(2), m2.group(1)))
            if edit_distance <= max_distance:
                potentials[edit_distance].append(mov)
        
        for i in range(max_distance + 1):
            if potentials[i] != []:
                return potentials[i]

        return []

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        potentials = []

        for movie in candidates:
            if re.search(r'\b({0})\b'.format(clarification), self.titles[movie][0]) != None:
                potentials.append(movie)
        
        if potentials == []:
            if clarification.isdigit():
                potentials.append(candidates[int(clarification) - 1])
        
        return potentials
        

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.zeros_like(ratings)
        binarized_ratings[ratings > threshold] = 1
        binarized_ratings[ratings <= threshold] = -1
        binarized_ratings[ratings == 0] = 0
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = np.dot(u, v)/(np.linalg.norm(u) * np.linalg.norm(v))
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.

        recommendations = []
        pred_ratings = []
        
        rated_movies = np.where(user_ratings != 0)[0]

        for movie in range(len(ratings_matrix)):
            if not movie in rated_movies:
                score = 0
                if ratings_matrix[movie].any():
                    for rated_movie in rated_movies:
                        score += user_ratings[rated_movie] * self.similarity(ratings_matrix[movie], ratings_matrix[rated_movie])
                    pred_ratings.append((movie, score))
                else:
                    pred_ratings.append((movie, 0.0))
        
        pred_ratings.sort(key = lambda x: x[1], reverse=True)
        
        for i in range(k):
            recommendations.append(pred_ratings[i][0])

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')

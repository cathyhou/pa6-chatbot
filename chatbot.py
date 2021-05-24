# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import numpy as np
import re
from porter_stemmer import PorterStemmer
import random


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
        self.yeses = ['yes', 'yep', 'yah', 'yeah', 'sure', 'yup', 'yea', 'yuh', 'definitely', 'correct', 'mmhm', 'i did', 'y']
        self.nos = ['no', 'nope', 'nah', 'i didn\'t', 'i did not', 'wrong', 'n']
        self.neg_emotions = ['angry', 'mad', 'sad', 'disappointed', 'frustrated', 'jealous', 'depressed', 'scared', 'afraid', 'fearful', 'bad mood']
        self.pos_emotions = ['happy', 'joyful', 'excited', 'good mood', 'relaxed', 'loving', 'hopeful']
        self.min_num_of_movies = 5
        self.num_ratings = 0
        self.error_count = 0

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

        self.new_input = True
        self.spell_check = False
        self.disambiguation = False
        self.recommendation = False
        self.cur_recommendations = []
        self.cur_input = []
        self.original_line = ""
        self.processed_line = ""
        self.catch_alls = ["let's go back to talking about movies!!", "ok, got it, but can we talk about movies again? tell me about a movie you watched recently", 
        "why don't we return to movies!!"]

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

       response = ""

       # candidates = possible movie indexes
       # sentiment = sentiment from movie input line
       
       if self.recommendation:
           if line.lower() in self.yeses:
               recommendation = self.cur_recommendations.pop(0)
               response = "\"{}\" would be a good movie for you too. ".format(self.titles[recommendation][0])
               self.recs_given += 1
               if self.recs_given >= 10:
                   self.recommendation = False
                   self.new_input = True
                   response += "\n i'm all out of recommendations. please tell me about other movies you've watched so i can give you more!"
                   self.num_ratings = 0
               else:
                   response += "want another? "
               return response
           elif line.lower() in self.nos:
               self.recommendation = False
               self.new_input = True
               return "okay thank you! you can tell me about more movies or type ':quit' to leave."
           else:
               return "sorry, i didn't get that. did you want another recommendation? you can reply with a 'yes' or 'no'."
               
    
       # responding to disabiguite (date, movies, spell check w multiple options)
       if self.disambiguation:
           original_len = len(self.candidates)
           self.candidates = self.disambiguate(line, self.candidates)
           if len(self.candidates) > 1:
               if len(self.candidates) < original_len:
                   return "that narrows it down a bit. which of these movies did you mean? {}".format(self.index_to_title(self.candidates))
               else:
                   return "sorry, i didn't get that! try giving me the year of the movie you meant."
           elif len(self.candidates) == 0:
               self.disambiguation = False
               self.new_input = True
               return "hm, i wasn't able to get that. can you try telling me about another movie?"
           else:
               self.disambiguation = False
               response += "thank you! "
               self.original_line = re.sub(self.cur_title, self.titles[self.candidates[0]][0], self.original_line)
               self.cur_input.append(self.candidates[0])

       # responding to spell check (only 1 option)
       if self.spell_check:
           if line.lower() in self.yeses:
               response += "got it! "
               self.original_line = re.sub(self.cur_title, self.titles[self.candidates[0]][0], self.original_line)
               self.cur_input.append(self.candidates[0])
               self.spell_check = False
           elif line.lower() in self.nos:
               self.spell_check = False
               return "hmm it seems like i haven’t seen that movie yet (´･_･`). try checking your spelling or tell me about another movie." 
           else:
               return "can you try again? did you mean \"{}\"?".format(self.titles[self.candidates[0]][0])
       
       if self.new_input:
           self.cur_input = []
           self.original_line = line
           self.processed_line = line
           self.new_input = False

       titles = self.extract_titles(self.processed_line)  

       if len(titles) == 0 and len(self.cur_input) == 0:
           if self.questions(line) != "":
               return self.questions(line) + " sorry, i can't help with that, but i can recommend you a movie! tell me about a movie you watched"
           if any(neg_emotion in line for neg_emotion in self.neg_emotions):
               self.new_input = True
               return "i'm sorry you feel that way. why don't you tell me about a movie and maybe i can help you feel better"
           elif any(pos_emotion in line for pos_emotion in self.pos_emotions):
               self.new_input = True
               return "i'm so glad you feel that way! can you tell me about a movie that made you also feel like that?"
           elif self.error_count > 0:
               self.new_input = True
               response = self.catch_alls[random.randint(0,2)]
               return response
           else:
               self.error_count += 1
               self.new_input = True
               return "sorry, i didn't get that. can you put the movie title in quotation marks. for example, \"Twilight\" or \"Twilight (2008)\"." 
       else:
           self.error_count = 0
           
       for title in titles:
           if title == "":
               self.new_input = True
               return "please include a movie between the quotes!"

           self.cur_title = title
           self.processed_line = re.sub(r'\"\b({0})\b\"'.format(title), "[movie]", self.processed_line)
           self.candidates = self.find_movies_by_title(title) 

           if len(self.candidates) > 1:
               self.disambiguation = True
               return "looks like there's more than one movie matching \"{}\". can you specify which one you mean? {}.".format(title, self.index_to_title(self.candidates))

           if len(self.candidates) == 0:
               self.candidates = self.find_movies_closest_to_title(title)  
               if len(self.candidates) > 1:
                   self.disambiguation = True
                   return "i couldn't find \"{}\" unfortunately. did you mean one of these instead, and if so, which one? {}".format(title, self.index_to_title(self.candidates))
               elif len(self.candidates) == 1:
                   self.spell_check = True
                   return "did you mean \"{}\" instead of \"{}\"?".format(self.titles[self.candidates[0]][0], title)
               else:
                   return "oops! i haven’t seen \"{}\" yet (´･_･`). please try another movie!".format(title)
           
           self.original_line = re.sub(self.cur_title, self.titles[self.candidates[0]][0], self.original_line)
           self.cur_input.append(self.candidates[0])
       
       sentiments = self.extract_sentiment_for_movies(self.original_line)

       likes = []
       dislikes = []
       unsures = []

       for i in range(len(sentiments)):
           if sentiments[i][1] > 0:
               likes.append(self.titles[self.cur_input[i]][0])
               self.user_ratings[self.cur_input[i]] = sentiments[i][1]
               self.num_ratings += 1
           elif sentiments[i][1] < 0:
               dislikes.append(self.titles[self.cur_input[i]][0])
               self.user_ratings[self.cur_input[i]] = sentiments[i][1]
               self.num_ratings += 1
           else:
               unsures.append(self.titles[self.cur_input[i]][0])
       
       if len(likes) > 0:
           response += "so you liked {} ヽ(ヅ)ノ ".format(', '.join(likes))
       if len(dislikes) > 0:
           response += "you disliked {} (｡•́︿•̀｡) ".format(', '.join(dislikes))
       if len(unsures) > 0:
           self.new_input = True
           return "i'm not sure if you liked {}, so please tell me more! ".format(', '.join(unsures))

       if self.num_ratings >= self.min_num_of_movies:
           response +=  "\n i’ve gathered enough information to recommend you a movie. hang tight! ヽ(＾Д＾)ﾉ"
           self.cur_recommendations = self.recommend(self.user_ratings, self.ratings)
           recommendation = self.cur_recommendations.pop(0)
           response += "\n i think you’ll really enjoy watching \"{}\". ".format(self.titles[recommendation][0])
           self.recs_given = 1    
           response += "want another?"
           self.recommendation = True
       else:
           response += "tell me another!"
           self.new_input = True
                          

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

    def questions(self, line):
        self.answers = {"can you":"i don't know if i can{}.", "can i":"i don't know if you can{}.",
        "what is":"i don't know what{} is.", "where is":"i don't know where{} is.", "how do i":"i don't know how to{}.",
        "how do you":"i'm not sure how i{}.", "who is":"hm, i don't know who{} is."}

        for key in self.answers:
            if line[0: len(key)].lower() == key:
                response = self.answers[key].format(line[len(key):])
                return response

        return ""
    
    def index_to_title(self, candidates):  
        movies = []
        for index in candidates:
            movies.append(self.titles[index][0])
        
        return movies

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

    def levenshtein(self, seq1, seq2, max):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1

        if abs(size_x-size_y) > max:
            return -1

        matrix = np.zeros ((size_x, size_y))
        for x in range(size_x):
            matrix[x][0] = x
        for y in range(size_y):
            matrix[0][y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix[x,y] = min(matrix[x-1, y] + 1, matrix[x-1, y-1], matrix[x, y-1] + 1)
                else:
                    matrix[x][y] = min(matrix[x-1,y] + 1, matrix[x-1,y-1] + 2, matrix[x,y-1] + 1)
  
        return matrix[size_x - 1][size_y - 1]
    
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
            edit_distance = int(self.levenshtein(m.group(2), m2.group(1), max_distance))
            if edit_distance >= 0 and edit_distance <= max_distance:
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
            title = " " + self.titles[movie][0] + " "
            if re.search(r'\b({})\b'.format(clarification.lower()), title.lower()) != None:
                potentials.append(movie)
        
        if potentials == []:
            if clarification.isdigit() and int(clarification) <= len(candidates):
                potentials.append(candidates[int(clarification) - 1])

        ordinal = {"first":0, "second":1, "third":2, "fourth":3, "fifth":4,"sixth":5, "seventh":6, "eighth":7, "ninth":8, "tenth":9}
        
        if potentials == []:
            for key in ordinal:
                if key in clarification:
                    potentials.append(candidates[ordinal[key]])

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

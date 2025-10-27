## Proyect 4. Uncertainty.



## Proyect 5. Learning.

  Shopping

When users are shopping online, not all will end up purchasing something. Most visitors to an online shopping website, in fact, likely don’t end up going through with a purchase during that web browsing session.
It might be useful, though, for a shopping website to be able to predict whether a user intends to make a purchase or not: perhaps displaying different content to the user, like showing the user a discount offer
if the website believes the user isn’t planning to complete the purchase. How could a website determine a user’s purchasing intent? That’s where machine learning will come in.

Your task in this problem is to build a nearest-neighbor classifier to solve this problem. Given information about a user — how many pages they’ve visited, whether they’re shopping on a weekend, what web browser
they’re using, etc. — your classifier will predict whether or not the user will make a purchase. Your classifier won’t be perfectly accurate — perfectly modeling human behavior is a task well beyond the scope of
this class — but it should be better than guessing randomly. To train your classifier, we’ll provide you with some data from a shopping website from about 12,000 users sessions.

How do we measure the accuracy of a system like this? If we have a testing data set, we could run our classifier on the data, and compute what proportion of the time we correctly classify the user’s intent. This
would give us a single accuracy percentage. But that number might be a little misleading. Imagine, for example, if about 15% of all users end up going through with a purchase. A classifier that always predicted
that the user would not go through with a purchase, then, we would measure as being 85% accurate: the only users it classifies incorrectly are the 15% of users who do go through with a purchase. And while 85%
accuracy sounds pretty good, that doesn’t seem like a very useful classifier.

Instead, we’ll measure two values: sensitivity (also known as the “true positive rate”) and specificity (also known as the “true negative rate”). Sensitivity refers to the proportion of positive examples that were
correctly identified: in other words, the proportion of users who did go through with a purchase who were correctly identified. Specificity refers to the proportion of negative examples that were correctly
identified: in this case, the proportion of users who did not go through with a purchase who were correctly identified. So our “always guess no” classifier from before would have perfect specificity (1.0) but no
sensitivity (0.0). Our goal is to build a classifier that performs reasonably on both metrics.

  Nim

Recall that in the game Nim, we begin with some number of piles, each with some number of objects. Players take turns: on a player’s turn, the player removes any non-negative number of objects from any one non-empty
pile. Whoever removes the last object loses.

There’s some simple strategy you might imagine for this game: if there’s only one pile and three objects left in it, and it’s your turn, your best bet is to remove two of those objects, leaving your opponent with
the third and final object to remove. But if there are more piles, the strategy gets considerably more complicated. In this problem, we’ll build an AI to learn the strategy for this game through reinforcement
learning. By playing against itself repeatedly and learning from experience, eventually our AI will learn which actions to take and which actions to avoid.

In particular, we’ll use Q-learning for this project. Recall that in Q-learning, we try to learn a reward value (a number) for every (state, action) pair. An action that loses the game will have a reward of -1, an
action that results in the other player losing the game will have a reward of 1, and an action that results in the game continuing has an immediate reward of 0, but will also have some future reward.

How will we represent the states and actions inside of a Python program? A “state” of the Nim game is just the current size of all of the piles. A state, for example, might be [1, 1, 3, 5], representing the state
with 1 object in pile 0, 1 object in pile 1, 3 objects in pile 2, and 5 objects in pile 3. An “action” in the Nim game will be a pair of integers (i, j), representing the action of taking j objects from pile i. So
the action (3, 5) represents the action “from pile 3, take away 5 objects.” Applying that action to the state [1, 1, 3, 5] would result in the new state [1, 1, 3, 0] (the same state, but with pile 3 now empty).

Recall that the key formula for Q-learning is below. Every time we are in a state s and take an action a, we can update the Q-value Q(s, a) according to:
Q(s, a) <- Q(s, a) + alpha * (new value estimate - old value estimate)
In the above formula, alpha is the learning rate (how much we value new information compared to information we already have). The new value estimate represents the sum of the reward received for the current action
and the estimate of all the future rewards that the player will receive. The old value estimate is just the existing value for Q(s, a). By applying this formula every time our AI takes a new action, over time our
AI will start to learn which actions are better in any state.

## Proyect 6. Neural Networks.

  Traffic

As research continues in the development of self-driving cars, one of the key challenges is computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular,
this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more. In this project, you’ll use TensorFlow to build a neural network to classify road signs based
on an image of those signs. To do so, you’ll need a labeled dataset: a collection of images that have already been categorized by the road sign represented in them. Several such data sets exist, but for this project,
we’ll use the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.

## Proyect 7. Language.

  Parser

A common task in natural language processing is parsing, the process of determining the structure of a sentence. This is useful for a number of reasons: knowing the structure of a sentence can help a computer to
better understand the meaning of the sentence, and it can also help the computer extract information out of a sentence. In particular, it’s often useful to extract noun phrases out of a sentence to get an
understanding for what the sentence is about.

In this problem, we’ll use the context-free grammar formalism to parse English sentences to determine their structure. Recall that in a context-free grammar, we repeatedly apply rewriting rules to transform symbols
into other symbols. The objective is to start with a nonterminal symbol S (representing a sentence) and repeatedly apply context-free grammar rules until we generate a complete sentence of terminal symbols (i.e.,
words). The rule S -> N V, for example, means that the S symbol can be rewritten as N V (a noun followed by a verb). If we also have the rule N -> "Holmes" and the rule V -> "sat", we can generate the complete
sentence "Holmes sat.".

Of course, noun phrases might not always be as simple as a single word like "Holmes". We might have noun phrases like "my companion" or "a country walk" or "the day before Thursday", which require more complex rules
to account for. To account for the phrase "my companion", for example, we might imagine a rule like: NP -> N | Det N
In this rule, we say that an NP (a “noun phrase”) could be either just a noun (N) or a determiner (Det) followed by a noun, where determiners include words like "a", "the", and "my". The vertical bar (|) just
indicates that there are multiple possible ways to rewrite an NP, with each possible rewrite separated by a bar.

To incorporate this rule into how we parse a sentence (S), we’ll also need to modify our S -> N V rule to allow for noun phrases (NPs) as the subject of our sentence. See how? And to account for more complex types
of noun phrases, we may need to modify our grammar even further.

  Attention

One way to create language models is to build a Masked Language Model, where a language model is trained to predict a “masked” word that is missing from a sequence of text. BERT is a transformer-based language
model developed by Google, and it was trained with this approach: the language model was trained to predict a masked word based on the surrounding context words.

BERT uses a transformer architecture and therefore uses an attention mechanism for understanding language. In the base BERT model, the transformer uses 12 layers, where each layer has 12 self-attention heads, for
a total of 144 self-attention heads.

This project will involve two parts:
 - First, we’ll use the transformers Python library, developed by AI software company Hugging Face, to write a program that uses BERT to predict masked words. The program will also generate diagrams visualizing
attention scores, with one diagram generated for each of the 144 attention heads.
 - Second, we’ll analyze the diagrams generated by our program to try to understand what BERT’s attention heads might be paying attention to as it attempts to understand our natural language.

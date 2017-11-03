--------------------------------------------

Author: Shahan Ali Memon

Mentors: Rita Singh, Bhiksha Raj

Copyright (c) 2017 Carnegie Mellon University

--------------------------------------------


--------------------------------------------

REQUIREMENTS:

To run the voice2age.py program, you would need the following libraries:
- keras
- numpy
- sox (python library via pip install sox)
- scipy

Once you have installed the above requirements (using pip install), you can run the 
program as follows:

$ python voice2age.py <full path to the wav file>

The above program will print an age range that the classifier would predict.

PS. If the model does not load due to the error "Can't load_model with error “Optimizer weight shape [..]", then you would need to run the convert_to_old.py program as most probably this issue has to do with the version of keras you are using.

Now you should try to run the program again.

--------------------------------------------

--------------------------------------------

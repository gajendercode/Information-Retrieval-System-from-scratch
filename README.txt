Programming Language: Python 3.9.7
Environment: Anaconda 4.11.0

Libraries Needed:
  numpy
  pandas
  glob
  nltk
  glob
  re
  os
  sys
  math
  pathlib
  collections
  pickle

Steps to execute:

Step 1: Make sure you install all the libraries mentioned above before running the assignment.
Step 2: Unzip 21111028-assignment1.zip in your local folder.
Step 3: Open the terminal window in the same folder where you extracted
Step 4: Issue the command, "make run path={path to the input query file}", and wait for ~2/3 minutes
Step 5: The output of all the three IR Systems should be present in the Output directory in the format as specified

NOTE: 1. The entire preprocessing is already performed on the ~8500 text files and the output files
      are saved in the folder "processed_docs". The corresponding code is contained in "q1.py". If you wish
      to recreate the processed documents, please use the command, "make setup". HOWEVER, THIS WILL TAKE
      AROUND ~15 MINUTES.

      2. The corpus is not included due to its huge size. If you want to do the preprocessing as mentioned above,
      please download the corpus from "https://www.cse.iitk.ac.in/users/arnabb/ir/english/" and extract it in the
      same folder as Makefile. At the end there should be a folder named "english-corpora" containing all
      the raw text files

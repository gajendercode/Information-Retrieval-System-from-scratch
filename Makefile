PYTHON = python3

# .PHONY defines parts of the makefile that are not dependant on any specific file
# This is most often used to store functions
.PHONY = help setup boolean tfidf bm25


# Defines the default target that `make` will to try to make, or in the case of a phony target, execute the specified commands
# This target is executed whenever we just type `make`
.DEFAULT_GOAL = help

# The @ makes sure that the command itself isn't echoed in the terminal
help:
	@echo "---------------HELP-----------------"
	@echo "To run all three systems and generate the output type: make run"
	@echo "To run the Boolean Model, type: make boolean"
	@echo "To run the TF-IDF Model, type: make tfidf"
	@echo "To run the BM-25 Model, type: make bm25"
#	@echo "To test the project type make test"
	@echo "To setup the assignment type: make setup"
	@echo "------------------------------------"

# This generates the desired project file structure
# A very important thing to note is that macros (or makefile variables) are referenced in the target's code with a single dollar sign ${}, but all script variables are referenced with two dollar signs $${}
setup:
	@rm -rf processed_docs
	@mkdir processed_docs	
	@echo "This will recreate the Pickle files after doing all sorts of preprocesing like tokenisation, stemming etc."
	@echo ""
	@echo "This should take around 15 minutes."
	@python q1.py

# The ${} notation is specific to the make syntax and is very similar to bash's $() 
# This function uses pytest to test our source files
boolean:
	@${PYTHON} boolean.py

tfidf:
	@${PYTHON} tfidf.py	

bm25:
	@${PYTHON} bm25.py
	
run:
	@unzip -o 21111063-ir-systems.zip
	@unzip -o 21111063-qrels.zip
	@rm -rf Output
	@mkdir Output
	@${PYTHON} run.py $(path)



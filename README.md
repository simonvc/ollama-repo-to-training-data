# ollama-repo-to-training-data

This tool is built in python. It creates many Training Data files numbered randomly ending in .td e.g. 956309819349.td

It is run by running `python ortd.py [path to git repo] "[preamble for each training sample created]"`

e.g. python ortd.py ../src/github.com/pavebank/pavebank "PaveBank core banking system"

The basic loop is to recurse through a code base over each source file (.go and .py only for now). for each file found:
  * create a training data file containing the whole source
  * create a training data file for each function
  * propose 5 hypothetical questions about the content of the file for example
  ** "This authentication handler uses the oauth2 middleware, what other handlers use this?"
  ** "This function uses the sql orm, where is the up/down schema migration defined?"

The ollama llm system with a model called "devstral" is running locally. All questions are provided to the local llm to answer.

Tools are defined and implemented by this system including:
* list_files - lists files in a directory
* list_files_recursively - lists files recursively in a directory
* find_in_files - searches in a file or deep search if given a directory

* implement other tools as necessesary

A queue is implemented to handle questions to be sent to the llm. Breadth first search rather than depth first should be preferred.

The goal is to create man training data files, each answering a sinle question about the code base. Each file will begin with the pre-amble provided on the code base and look like this: (simplified example)
```
** Pave Bank Core banking system **
Implemented in pavebank/auth/handler.go is a function called get_user.
How does this work?
The get_user function defined in pavebank/auth/handler.go makes an RPC call to ...

```
td files should contain extensive code snippits and comments and explain assumed business logic.

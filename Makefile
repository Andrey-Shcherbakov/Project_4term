#Python makefile
all: runproject

some_file: runproject

runproject:
	python data_prepro.py clean.py train.py predict.py

clean:
	rm -f runproject
#some questions according to cleaning and using Non_L_methods

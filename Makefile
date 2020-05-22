#Python makefile
all: runproject

some_file: runproject

runproject:
	python data_prepro.py models.py train.py clean.py predict.py

clean:
	rm -f runproject
#some questions according to cleaning and using Non_L_methods

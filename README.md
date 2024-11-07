# NLP-Project
Stock Movement Prediction

Base paper: https://www.sciencedirect.com/science/article/pii/S0957417423021565

- Run causality_new.py.
  
	Input: News data and stock data (datasets available)

	Output: Json file: Causal clusters

		Json file: Cluster strengths

- Run merger.py.
  
	Input: News data and stock data

	Output: CSV: Merged data based on matching dates

- Run feature_building.py.
  
	Input: Merged data from merger.py

	Output: CSV: Feature set for training

- Run training.py.
  
	Input: Merged data from merger.py

	       Feature set from feature_building.py
  
	Output: Predictions and performance metrics

- Run Stock_CNN.py.
  
	Input: Same as training.py

	Output: Same as training.py

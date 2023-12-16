Instructions to use:-

1. conda install requirements.txt
2. Populate your movie review dataset into the review column of the TAPE.xlsx file
3. Run main.py to get the enhanced text and pseudo labels (automatically populated in TAPE.xlsx at appropriate columns), change the API key if required
4. Run llm_gm.py for to run training and testing. Change test and train split appropriately in the file.

All the above files have by default 180 Movie reviews and can run based on it. Change the hyperparams as you feel necessary for larger dataset. 

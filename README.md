Unofficial implementation of https://github.com/CurryTang/Graph-LLM

Check these slides for more info: https://docs.google.com/presentation/d/1ewERkOaraOuJlZd6DtJM7aPXnE1Zb-bOmKzb2vgT4BA/edit?usp=sharing

Instructions to use for LLM as enhancer:-

1. conda install requirements.txt
2. Populate your movie review dataset into the review column of the TAPE.xlsx file
3. Run main.py to get the enhanced text and pseudo labels (automatically populated in tape.xlsx at appropriate columns), change the API key if required
4. Run llm_gm.py for to run training and testing. Change test and train split appropriately in the file.

Instructions to use LLM as predictor:-
1. conda install requirements.txt
2. Run create_prompt.py (will populate tape.csv)



All the above files have by default 180 Movie reviews and can run based on it. Change the hyperparams as you feel necessary for larger dataset. 

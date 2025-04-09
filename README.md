# Detecting and disambiguating language instructions for a robot

This repository contains a slight modification to the original [AmbiK repository](https://github.com/cog-model/AmbiK-dataset) - a possibility to get uncertainty scores of LLM's prediction using one of estimation methods taken from [LM-polygraph](https://github.com/IINemo/lm-polygraph). Currently works only with [KnowNo](https://arxiv.org/abs/2307.01928) framework.

## Project structure

- 🐍`llm_ue.py` - modified version of [original class](https://github.com/cog-model/AmbiK-dataset/blob/main/utils/llm.py) that return uncertainty scores in addition to logits and model answers.
- 🐍`knownoconfig.py` - file needed to configure [KnowNo](https://arxiv.org/abs/2307.01928) method (taken from [original repo](https://github.com/cog-model/AmbiK-dataset/blob/main/orig_knowno_pipeline.ipynb)).
- 🐍`knownopipeline.py` - modified [class](https://github.com/cog-model/AmbiK-dataset/blob/main/orig_knowno_pipeline.ipynb) so that it could be run together with `llm-ue.py` class.
- 📁`notebooks/baseline-example.ipynb` - example notebook that shows how to run a custom LLM model and get uncertainty scores along with the predictions on KnowNo dataset for multiple-choice question-answering problem.
- 📁`notebooks/lm-test.ipynb` - notebook that I used to see what different uncertainty estimation methods from [LM-polygraph framework](https://github.com/IINemo/lm-polygraph) would return for different open-source models

## How to use

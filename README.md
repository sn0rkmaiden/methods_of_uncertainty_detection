# Detecting and disambiguating language instructions for a robot

This repository contains a slight modification to the original [AmbiK repository](https://github.com/cog-model/AmbiK-dataset) - a possibility to get uncertainty scores of LLM's prediction using one of estimation methods taken from [LM-polygraph](https://github.com/IINemo/lm-polygraph). Currently works only with [KnowNo](https://arxiv.org/abs/2307.01928) framework.

## Project structure

- 🐍`llm_ue.py` - modified version of [original class](https://github.com/cog-model/AmbiK-dataset/blob/main/utils/llm.py) that return uncertainty scores in addition to logits and model answers.
- 🐍`knownoconfig.py` - file needed to configure [KnowNo](https://arxiv.org/abs/2307.01928) method (taken from [original repo](https://github.com/cog-model/AmbiK-dataset/blob/main/orig_knowno_pipeline.ipynb)).
- 🐍`knownopipe.py` - modified [class](https://github.com/cog-model/AmbiK-dataset/blob/main/orig_knowno_pipeline.ipynb) so that it could be run together with `llm-ue.py` class.
- 📁`notebooks/baseline-example.ipynb` - example notebook that shows how to run a custom LLM model and get uncertainty scores along with the predictions on KnowNo dataset for multiple-choice question-answering problem.
- 📁`notebooks/lm-test.ipynb` - notebook that I used to see what different uncertainty estimation methods from [LM-polygraph framework](https://github.com/IINemo/lm-polygraph) would return for different open-source models.

## How to use

Project requires files from all mentioned above repositories (I had to modify some files in them so I use forks here):

```python
!git clone -q "https://github.com/sn0rkmaiden/lm-polygraph.git"
!git clone -q "https://github.com/sn0rkmaiden/AmbiK-dataset.git"
!git clone -q "https://github.com/sn0rkmaiden/methods_of_uncertainty_detection.git"
```

Main methods for running LLM models are `KnowNoPipeline.run()` and `KnowNoPipeline.run_batch()`:

```python
options, logits, answers, right_answers, gen_scores, ans_scores = knowno.run_batch(option_prompts, tasks_for_ans)
```
Example output for `estimator=MaximumSequenceProbability()`:
- options: `{'A': 'A) use the bread knife to cut the vegetables into small pieces', 'B': 'B) use the paring knife to cut the vegetables into small pieces', 'C': 'do nothing', 'D': 'do nothing'}`
- answers: `['A', 'B']`
- right_answers: `['A) use the bread knife to cut the vegetables into small pieces', 'B) use the paring knife to cut the vegetables into small pieces']`
- ans_scores: `[17.31479]`

A detailed example can be found in `notebooks/baseline-example.ipynb`.

### Metrics and results

Next metrics were calculated during experiments (more details in the [paper](file:///C:/Users/%D0%90%D0%BB%D0%B8%D1%81%D0%B0/Desktop/MIPT-internship/papers-to-read/AmbiK_dataset.pdf)):
- Success rate (SR) 
- Help rate (HR)
- Correct help rate (CHR)
- Set size correctness (SSC)

I tried to use these models for prompt and answer generation: `google/gemma-2b`, `microsoft/phi-2`, `google/flan-t5-base` and some smaller others. According to subjective assessment and metric scores, `google/gemma-2b` performed best. For future work it would be interesting to try bigger models.

[Link](https://wandb.ai/snork_maiden-_/my-knowno-project?nw=nwusersnork_maiden) to my experiments (a few of those that I managed to run).

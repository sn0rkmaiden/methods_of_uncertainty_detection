import tqdm
from methods_of_uncertainty_detection.llm_ue import LLModel

class KnowNoPipeline():
    def __init__(self, title_prompt, title_answer, model_prompt, model_answer, tokenizer_prompt, tokenizer_answer, estimator, cpvalue, examples, adapter_prompt, adapter_ans, config=None):        
        self.config = config
        self.cp = cpvalue #You can use calibrate.py to recalculate 0.9 - llama 0.7 gemma
        self.mapping_1 = ['A', 'B', 'C', 'D']
        self.model_prompt = model_prompt
        self.model_answer = model_answer
        self.tokenizer_prompt = tokenizer_prompt
        self.tokenizer_answer = tokenizer_answer
        self.title_prompt = title_prompt
        self.title_answer = title_answer
        self.estimator = estimator
        self.examples_generation = examples    
        self.adapter_prompt = adapter_prompt
        self.adapter_ans = adapter_ans

    def options_prompt(self, description, task, prefix, action):
        #creating prompt for generating options (base prompt is taken from knowno/prompts/generation.txt)
        prompt = self.examples_generation.replace('<DESCRIPTION>', description)
        prompt = prompt.replace('<TASK>', task)
        prompt = prompt.replace('<PREFIX>', prefix)
        prompt = prompt.replace('<ACT>', action)
        return prompt

    def format_options(self, prompt, options):
        examples = options.replace(prompt, "")
        options, options_str = format_examples(examples)
        return options

    def predict_examples(self, description, task, prefix, action):
        llm = LLModel(self.title_prompt, self.model_prompt, self.tokenizer_prompt, self.estimator, self.adapter_prompt)
        prompt = self.options_prompt(description, task, prefix, action)
        options, ue_score = llm.generate(prompt)
        llm = None

        return self.format_options(prompt, options), ue_score

    def predict_examples_batch(self, prompts, batch_size=2): #generate examples for batch
        llm = LLModel(self.title_prompt, self.model_prompt, self.tokenizer_prompt, self.estimator, self.adapter_prompt)
        options_full = []
        scores_full = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size)):
            options, score = llm.generate_batch(prompts[i:i+batch_size])
            options_full += options
            scores_full.append(score)

        formated_options = []
        for i in range(len(prompts)):
            prompt = prompts[i]
            options = options_full[i]
            formated_options.append(self.format_options(prompt, options))
        llm = None
        gc.collect()
        return formated_options, scores_full

    def answer_with_cp(self, tokens_logits): #take in CP set only options whose logits are greater then CP value. CP value is defined through running calibration.py script
        possible_options = []
        for key in tokens_logits.keys():
            if tokens_logits[key] > self.cp:
                possible_options.append(key)
      #  print(possible_options)
        formated_options = []
        for option in possible_options:
            if option.isdigit():
                option_formated = self.mapping_1[int(option)-1]
            else:
                option_formated = option.upper()
            if option_formated not in formated_options:
                formated_options.append(option_formated)
        return formated_options

    def answer_with_estimator(self, tokens_logits, ue_scores):
      pass

    def answer_prompt(self, prompts_id, description, task, prefix, action):
        #prompt for getting logprobs of A, B, C, D variants. base prompt is in knowno/prompts/choising.txt
        #prompt = prompt.replace("You", "Options")
        prompt = ''

        if isinstance(prompts_id, tuple):
          prompts_id = prompts_id[0]

        for key, value in prompts_id.items():
            prompt += key
            prompt += ') '
            prompt += value[3:]
            prompt += '\n'
        prompt = answer_generation.replace('<OPTIONS>', prompt)
        prompt = prompt.replace('<TASK>', task)
        prompt = prompt.replace('<PREFIX>', prefix)
        prompt = prompt.replace('<ACT>', action)
        prompt = prompt.replace('<DESCRIPTION>', description)
        return prompt

    def generate_answer_batch(self, prompts, batch_size=1): #choosing CP set for batch
        llm = LLModel(self.title_answer, self.model_answer, self.tokenizer_answer, self.estimator, self.adapter_ans)
        full_texts = []
        full_logits = []
        full_scores = []
        for i in tqdm.tqdm(range(0, len(prompts), batch_size)):
            texts, logits, ue_score = llm.generate_batch(prompts[i:i+batch_size], return_logits=True)
            full_texts += texts
            full_logits += logits
            full_scores.append(ue_score)

        filtered_logits_batch = []
        answers = []
        for i in range(len(full_texts)):
            filtered_logits = llm.filter_logits(full_logits[i][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])
            #print(filtered_logits)
            filtered_logits_batch.append(filtered_logits)
            answer = self.answer_with_cp(filtered_logits)
            answers.append(answer)
        llm = None
        gc.collect()
        return filtered_logits_batch, answers, full_scores

    def generate_answer(self, prompt, description, task, prefix, action): #choosing CP set for single example
        llm = LLModel(self.title_answer, self.model_answer, self.tokenizer_answer, self.estimator, self.adapter_ans)

        prompt = self.answer_prompt(prompt, description, task, prefix, action)
        text, logits, ue_score = llm.generate(prompt, return_logits=True)
        filtered_logits = llm.filter_logits(logits[-1][0], words=["A", "B", "C", "D", 'a', 'b', 'c', 'd', '1', '2', '3', '4'])

        llm = None
        return filtered_logits, self.answer_with_cp(filtered_logits), ue_score

    def run(self, description, task, prefix, action):
        #generating multiple options
        options, predict_score = self.predict_examples(description, task, prefix, action)

        #getting logits of options and choosing the options with logits greater than CP value. constricting a CP set.
        answers_letter, answer_score = self.generate_answer(options, description, task, prefix, action)[1:]

        answers = [options[letter] for letter in answers_letter]

        if len(answers)==0:
            return [] #if no options are left in the set, it is impossible to answer
        return options, answers, predict_score, answer_score #else there 1 or many answers

    def run_batch(self, option_prompts, tasks_for_ans): #run, but for batch
        options, gen_scores = self.predict_examples_batch(option_prompts)
        answer_prompts = []
        for i in range(len(options)):
            answer_prompts.append(self.answer_prompt(options[i], tasks_for_ans[i]['description'], tasks_for_ans[i]['task'], tasks_for_ans[i]['prefix'], tasks_for_ans[i]['action']))
        logits, answers, ans_scores = self.generate_answer_batch(answer_prompts)
        right_answers = []

        for i in range(len(answers)):
            option = options[i]
            answers_letter = answers[i]
            if len(answers_letter) > 0:
                answers_ = [option[letter] for letter in answers_letter]
            else:
                answers_ = []
            right_answers.append(answers_)
        return options, logits, answers, right_answers, gen_scores, ans_scores

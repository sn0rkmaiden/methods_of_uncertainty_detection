class KnowNoConfig():
    def __init__(self, config):
        config_mapping = {
            'examples_generation': ['model', 'generation_kwargs'],
            'answering': ['model', 'generation_kwargs'],
        }
        self.config = config
        self.examples_generation = config['examples_generation']
        self.answering = config['answering']

def format_examples(examples):
    #checking whether the answer prompt is in the correct format: A) <option A>\nB) <option B> ... D) <option D>. If something is wrong, replacing the variant with 'do nothing'
    lines = examples.split("\n")
    if "\n" in examples:
        lines = examples.split("\n")
        if len(lines) < 4:
            lines = [x for x in re.split(r'(\d. [\w\s]*.)', examples) if len(x) > 2]
    else:
        lines = [x for x in re.split(r'(\d. [\w\s]*.)', examples) if len(x) > 2]

    options = ""
    mapping = {"A": "1", "B":"2", "C": "3", "D": "4"}
    variants = {"A":[], "B":[], "C":[], "D":[]}
    for line in lines:
        for key in variants.keys():
            if line.startswith(f"{key})") or line.startswith(f"{mapping[key]}.") or line.startswith(f"{mapping[key]})"):
                variants[key].append(line)

    for key in variants.keys():
        variants[key] = list(set(variants[key]))
        if len(variants[key]) > 0:
            options+=variants[key][0] +"\n"
            variants[key] = variants[key][0]
        else:
            options += 'do nothing' +"\n"
            variants[key] = 'do nothing'
    return variants, options

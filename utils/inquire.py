import inquirer

def inquire(question:str,options:list[str],default:str|None=None):
    questions = [
    inquirer.List('choice',
                    message=question,
                    choices=options,
                    default=default,
                ),
    ]
    answers = inquirer.prompt(questions)
    return answers["choice"]

if __name__ == "__main__":
    inquire("does this work?",["Yes","No"])
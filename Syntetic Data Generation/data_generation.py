import json

import openai


def generate_text_with_open_ai():
    # Set up the OpenAI API key
    openai.api_key = "YOUR_OPENAI_API_KEY"
    # Set the GPT-3 model to use
    model = "text-davinci-002"
    # Set the prompt for the GPT-3 model
    prompt = "Generate a review with positive sentiment about AWS"
    # Generate 100 samples of synthetic text data
    generated_text = list()
    for i in range(100):
        completions = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=60,
            n=1,
            temperature=0.75,
        )
        generated_text.append(completions.choices[0].text)

    # Save generated text to a JSON file
    with open("generated_text_positive.json", "w") as write_file:
        json.dump(generated_text, write_file, indent=4)


def main():
    generate_text_with_open_ai()


if __name__ == '__main__':
    main()

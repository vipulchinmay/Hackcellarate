import openai
openai.api_key = "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"
def chat(prompt):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user","content": prompt}]
    )
    return response.choices[0].message.content.strip()
if __name__ == "__main__":
    while True:
        user_input = ("You: ")
        if user_input.lower() in ["bye","Quit","exit","quit"]:
            break
        response = chat(user_input)
        print("ChatBot: ",response)
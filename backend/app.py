from flask import Flask, request, jsonify
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))

# Load the model and tokenizer, and move model to the selected device
model = T5ForConditionalGeneration.from_pretrained('model/model').to(device)
tokenizer = T5Tokenizer.from_pretrained('model/tokenizer')

def generate_qa_pairs(context, num_questions=1):
    try:
        # Prepare the input for the model
        input_text = f"context: {context}"
        input_ids = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True).input_ids.to(device)

        # Generate questions
        question_outputs = model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=num_questions,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=1.0
        )
        questions = tokenizer.batch_decode(question_outputs, skip_special_tokens=True)

        # Prepare to generate answers for each question
        qa_pairs = []
        for question in questions:
            answer_input = f"question: {question} context: {context}"
            answer_ids = tokenizer(answer_input, return_tensors='pt', max_length=512, truncation=True).input_ids.to(device)

            # Generate answers
            answer_output = model.generate(answer_ids, max_length=50)
            answer = tokenizer.decode(answer_output[0], skip_special_tokens=True)

            qa_pairs.append({"question": question, "answer": answer})

        return qa_pairs
    except Exception as e:
        print("Error generating QA pairs:", e)
        return []

@app.route('/generate', methods=['POST'])
def predict():
    # Receive JSON data from frontend
    data = request.get_json()
    context = data.get("context")
    
    # Check if context is provided
    if not context:
        return jsonify({"error": "No context provided"}), 400

    # Generate question-answer pairs
    qa_pairs = generate_qa_pairs(context, num_questions=5)

    # Send back the prediction
    return jsonify({'QA': qa_pairs})

if __name__ == '__main__':
    app.run(debug=True)
